import streamlit as st
import time
from core.cosmos import cosmos

st.set_page_config(page_title="cosmos AI", layout="wide", page_icon="⚡")

# Custom CSS for "Premium Aesthetics"
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cosmos():
    return cosmos()

ai = get_cosmos()

# Sidebar: System Status
with st.sidebar:
    st.header("🧠 Neural Status")
    
    # Memory Stats
    mem_count = ai.memory.index.ntotal if hasattr(ai.memory, 'index') else 0
    st.metric("LTM Vectors", mem_count)
    
    node_count = ai.memory.graph.number_of_nodes()
    edge_count = ai.memory.graph.number_of_edges()
    st.metric("Knowledge Nodes", node_count)
    st.metric("Synaptic Connections", edge_count)
    
    st.divider()
    
    st.header("📥 Ingestion")
    url = st.text_input("Ingest URL")
    if st.button("Absorb Knowledge"):
        if url:
            status = ai.ingest_data(url)
            st.success(status)
    
    if st.button("Ingest Local Data"):
        status = ai.ingest_data("local")
        st.success(status)

# Main Chat
st.title("cosmos AI v1.0")
st.caption("Autonomous • Recursive • Evolving")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Command the swarm..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream response (simulated for now, as Swarm runs atomically)
        with st.spinner("Swarm Processing..."):
            response = ai.chat(prompt)
        
        # Typewriter effect
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
