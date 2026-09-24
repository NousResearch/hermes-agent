"""Start the chatroom server."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from server.main import run_server

if __name__ == "__main__":
    print("🚀 Multi-Agent Chatroom Server starting on ws://localhost:8765 ...")
    run_server()
