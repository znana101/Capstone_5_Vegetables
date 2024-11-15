 #%%
import streamlit as st
import requests
import os
from ultralytics import solutions

# Constant take from Langflow API > Python API > c&p
BASE_API_URL = "http://127.0.0.1:7860"
FLOW_ID = "b580a291-ab51-46d9-a943-d1046b451353"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

solutions.inference(model=r"D:\YPAI09\Capstone\Capstone_5_Vegetable_Zaryth\runs\detect\train8\weights\best.pt")

def run_flow(message: str) -> dict:
    """
    Run a flow with a given message.

    :param message: The message to send to the flow
    :return: The JSON respons from the flow
    """

    api_url = f"{BASE_API_URL}/api/v1/run/{ENDPOINT or FLOW_ID}"

    payload = {
        "input_value": message,
        'output_type': 'chat',
        'input_type': 'chat',
    }

    response = requests.post(api_url, json=payload)
    return response.json()

# Function to extract the desired message
def extract_message(response: dict)-> str:
    try:
        #NAvigate to the message inside the response structure
        return response ['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        return 'No valid message found in response.'

# Streamlit app
def main():

    with st.sidebar:
        st.image(r"D:\YPAI09\Capstone\Capstone_5_Vegetable_Zaryth\vegetables.jpg",use_column_width="auto")
        st.write("""
        ## :grey-background[:orange[About This Project 🥔]]
        This chatbot provides detailed information about the project, its implementation, and various vegetable characteristics. It is an AI-powered object detection system designed to automatically identify and classify 5 types of vegetables (cabbage, cauliflower, cucumber, potato and tomato) in images. 
        Developed for an agricultural technology company, this project aims to enhance automation and accuracy in vegetable identification, making it a valuable tool for smart farming and food processing industries.
        """) 

        st.write("""
        _Visit my LinkedIn profile to see my progress:_
        """)
        st.link_button("LinkedIn", "https://www.linkedin.com/in/zaryth-n-497013283")

    st.title(":green[AI-Powered Vegetable Detection System] _with Intelligent Chatbot_")
    # st.imag('user.jpg', width=50)

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages =[]

    #Display previous messages into avatars
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message['avatar']):
            st.write(message['content'])

    #input box for user message
    if query := st.chat_input("Ask me anything..."):
        #Add user message to session state
        st.session_state.messages.append(
            {
                'role':'user',
                'content': query,
                'avatar': '⭐', #emoji for user
            }
        )
        with st.chat_message('user', avatar='⭐'): #Display user message
            st.write(query)

        #Call the langflow API and get thhe assistant's response
        with st.chat_message('assistant', avatar='😶‍🌫️'): #Emoji for assistant
            message_placeholder = st.empty() #placeholder for assistant response
            with st.spinner("Thinking... 💭"):
                #Fetch response from Langflow
                assistant_response = extract_message(run_flow(query))
                message_placeholder.write(assistant_response)

        #Add assistant response to session state
        st.session_state.messages.append(
            {
                'role':'assistant',
                'content': assistant_response,
                'avatar': '😶‍🌫️' 
                , #emoji for assistant
            }
        )

if __name__ =='__main__':
    main()

# %%
