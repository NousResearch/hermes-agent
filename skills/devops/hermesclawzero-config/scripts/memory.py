import requests
import sys
import os
from pathlib import Path

# Update these to match your HermesClawZero setup
BASE_URL = os.getenv("MEM_PUBLIC_URL") or os.getenv("OPENCLAW_URL") or "http://localhost:8000"
API_KEY = os.getenv("API_KEY") or os.getenv("OPENCLAW_KEY")
SYNC_DIR = os.getenv("MEM_SYNC_DIR") or os.getenv("OPENCLAW_SYNC_DIR") or str(Path.cwd() / "sync")

if not API_KEY:
    raise RuntimeError("API_KEY is required. Set API_KEY (or OPENCLAW_KEY for compatibility).")

def capture(text):
    url = f"{BASE_URL}/capture"
    params = {"key": API_KEY}
    data = {"text": text}
    try:
        response = requests.post(url, params=params, json=data)
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")

def search(query, limit=5):
    url = f"{BASE_URL}/search"
    params = {"query": query, "limit": limit, "key": API_KEY}
    try:
        response = requests.get(url, params=params)
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")

def autosave(content, filename):
    if not filename.endswith(('.txt', '.md', '.json')):
        filename += ".txt"
    filepath = os.path.join(SYNC_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Autosaved to {filepath}")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: memory.py [capture|search|autosave] [args...]")
        sys.exit(1)
        
    cmd = sys.argv[1]
    if cmd == "capture":
        capture(sys.argv[2])
    elif cmd == "search":
        search(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    elif cmd == "autosave":
        autosave(sys.argv[2], sys.argv[3])
