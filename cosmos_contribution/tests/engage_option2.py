import requests
import json
import time

BASE_URL = "http://127.0.0.1:8081"

def print_section(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def check_status():
    print_section("Checking System Status")
    try:
        resp = requests.get(f"{BASE_URL}/api/status")
        if resp.status_code == 200:
            print("Usage: success")
            data = resp.json()
            # Summary only to avoid spam
            print(f"Version: {data.get('version')}")
            print(f"Features: {list(data.get('features', {}).keys())}")
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Connection failed: {e}")

def chat_personal():
    print_section("Engaging Personal Chat")
    payload = {
        "message": "Hello cosmos, are you online?",
        "model": "cosmos",
        "chat_mode": "personal"
    }
    try:
        # Use simple chat endpoint
        resp = requests.post(f"{BASE_URL}/api/chat", json=payload)
        if resp.status_code == 200:
            print(f"Response: {resp.json().get('response', '')}")
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Failed: {e}")

def chat_cosmos():
    print_section("Engaging Cosmos Swarm")
    payload = {
        "message": "Briefly explain the current state of the swarm.",
        "model": "cosmos-12d"
    }
    try:
        # Try specific cosmos swarm endpoint
        resp = requests.post(f"{BASE_URL}/api/cosmos-swarm", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get('content', '')
            print(f"Cosmo's Synthesis: {content[:200]}..." if len(content) > 200 else f"Cosmo's Synthesis: {content}")
            print(f"Models Used: {len(data.get('individual_responses', []))}")
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Failed: {e}")

def check_evolution():
    print_section("Checking Evolution Status")
    try:
        resp = requests.get(f"{BASE_URL}/api/evolution/status")
        if resp.status_code == 200:
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"Error: {resp.status_code}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    print(f"Engaging cosmos on {BASE_URL}...")
    check_status()
    chat_personal()
    chat_cosmos()
    check_evolution()
    print("\nEngagement Complete.")
