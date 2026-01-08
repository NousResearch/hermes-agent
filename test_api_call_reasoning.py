#!/usr/bin/env python3
"""
Test script to see how minimax-m2.1 responds to a tool-calling request via OpenRouter.
"""
import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded .env from {env_path}")

# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ OPENROUTER_API_KEY not found in environment")
    exit(1)
print(f"🔑 Using API key: {api_key[:12]}...{api_key[-4:]}")

# Initialize client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# Define a single simple tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on any topic. Returns relevant results with titles and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Messages
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to tools. Use the web_search tool when you need to find information."
    },
    {
        "role": "user", 
        "content": "What is the current price of Bitcoin?"
    }
]

print("\n" + "="*60)
print("📤 SENDING REQUEST")
print("="*60)
print(f"Model: minimax/minimax-m2.1")
print(f"Messages: {len(messages)}")
print(f"Tools: {len(tools)}")
print(f"User query: {messages[-1]['content']}")

# Make the request
try:
    response = client.chat.completions.create(
        model="minimax/minimax-m2.1",
        messages=messages,
        tools=tools,
        extra_body={
            "provider": {
                "only": ["minimax"]
            }
        },
        timeout=120.0
    )
    
    print("\n" + "="*60)
    print("📥 RESPONSE RECEIVED")
    print("="*60)
    
    # Print raw response info
    print(f"\nModel: {response.model}")
    print(f"ID: {response.id}")
    print(f"Created: {response.created}")
    
    if response.usage:
        print(f"\n📊 Usage:")
        print(f"   Prompt tokens: {response.usage.prompt_tokens}")
        print(f"   Completion tokens: {response.usage.completion_tokens}")
        print(f"   Total tokens: {response.usage.total_tokens}")
    
    # Print the message
    msg = response.choices[0].message
    print(f"\n🤖 Assistant Response:")
    print(f"   Role: {msg.role}")
    print(f"   Content: {msg.content}")
    print(f"   Tool calls: {msg.tool_calls}")
    
    if msg.tool_calls:
        print(f"\n🔧 Tool Calls Detail:")
        for i, tc in enumerate(msg.tool_calls):
            print(f"   [{i}] ID: {tc.id}")
            print(f"       Function: {tc.function.name}")
            print(f"       Arguments: {tc.function.arguments}")
    
    # Print full raw response as JSON
    print("\n" + "="*60)
    print("📝 RAW RESPONSE (JSON)")
    print("="*60)
    print(json.dumps(response.model_dump(), indent=2, default=str))

except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

