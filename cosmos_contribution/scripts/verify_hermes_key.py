import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_nous_inference():
    print("=== Testing Nous Research Inference API ===")
    api_key = "sk-192nnl320tcvnbmgd053"
    url = "https://inference-api.nousresearch.com/v1/chat/completions"
    model = "nous-hermes-4-70b" # Or "nous-hermes-4-405b"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            print("  ✅ Nous Inference SUCCESS!")
            return True
        else:
            print(f"  ❌ Nous Inference FAILED ({response.status_code}): {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ Nous Inference ERROR: {e}")
        return False

if __name__ == "__main__":
    test_nous_inference()
