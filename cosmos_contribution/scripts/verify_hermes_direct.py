import os
import json
import urllib.request
import urllib.error
from pathlib import Path

def verify_hermes_key():
    # Load .env manually to be safe
    env_path = Path("D:/Cosmos/.env")
    api_key = None
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("HERMES_API_KEY="):
                    api_key = line.split("=")[1].strip()
                    break
    
    if not api_key:
        api_key = os.environ.get("HERMES_API_KEY")
        
    if not api_key:
        print("❌ Error: HERMES_API_KEY not found in .env or environment")
        return

    print(f"Testing HERMES_API_KEY: {api_key[:5]}...")
    
    url = "https://inference-api.nousresearch.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try both common names if one fails, but we'll start with what we set in .env
    model_id = "Hermes-4-70B"
    
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello!"}],
        "max_tokens": 10
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)
    
    try:
        print(f"Sending request to Nous Research Direct API ({model_id})...")
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            if "choices" in res_data:
                content = res_data["choices"][0]["message"]["content"]
                print(f"✅ SUCCESS! Response: {content}")
            else:
                print(f"❓ Unexpected response: {res_data}")
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error: {e.code} - {e.reason}")
        err_body = e.read().decode()
        print(err_body)
        if "model" in err_body.lower():
            print("Tip: If model is not found, we may need to try 'nous-hermes-4-70b'")
    except urllib.error.URLError as e:
        print(f"❌ URL Error: {e.reason}")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

if __name__ == "__main__":
    verify_hermes_key()
