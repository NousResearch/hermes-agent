# Hermes Agent - Lightweight Core for Android
# This module provides a simplified AIAgent that runs efficiently on Android/Termux
# Heavy operations are automatically bridged to the VPS via WebSocket.

import os
import sys
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import requests
import websocket
import yaml

logger = logging.getLogger(__name__)

# Path to the bridge config
CONFIG_PATH = Path(os.environ.get("HERMES_HOME", "~/.hermes-android")).expanduser() / "config.yaml"


class AndroidBridge:
    """WebSocket bridge to VPS for heavy operations."""

    def __init__(self, server_url: str = None, auth_token: str = None):
        self.server_url = server_url or "ws://localhost:2999/socket"
        self.auth_token = auth_token
        self.ws = None
        self.connected = False

    def connect(self):
        """Connect to the bridge server."""
        try:
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"Bridge connection failed: {e}")
            logger.info("Operating in local-only mode")

    def _on_open(self, ws):
        self.connected = True
        logger.info("Bridge connected to VPS")

    def _on_message(self, ws, message):
        # Handle incoming messages from VPS
        logger.debug(f"Bridge message: {message}")

    def _on_error(self, ws, error):
        logger.error(f"Bridge error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        logger.info("Bridge disconnected")

    def send(self, message: dict):
        """Send message to VPS via bridge."""
        if self.connected:
            self.ws.send(json.dumps(message))
        else:
            logger.warning("Bridge not connected, skipping")


class LightweightAgent:
    """Simplified AIAgent for Android."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(CONFIG_PATH)
        self.config = self._load_config()
        self.bridge = AndroidBridge(
            self.config.get("bridge", {}).get("server_url"),
            self.config.get("bridge", {}).get("auth_token")
        )
        self.tools = {}
        self.session_id = None
        self.history = []
        self._setup_logging()
        self._load_lightweight_tools()

    def _load_config(self) -> dict:
        """Load configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {
            "max_iterations": 30,
            "quiet_mode": False,
            "enabled_toolsets": ["file", "terminal", "web", "skills"],
            "disabled_toolsets": ["browser", "computer_use", "vision", "cronjob", "delegation", "image_gen", "voice"],
            "bridge": {
                "enabled": True,
                "server_url": "ws://localhost:2999/socket",
                "auth_token": None
            }
        }

    def _setup_logging(self):
        """Setup logging."""
        log_file = Path(self.config.get("hermes_home", "~/.hermes-android")).expanduser() / "agent.log"
        logging.basicConfig(
            level=getattr(logging, self.config.get("log_level", "INFO")),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _load_lightweight_tools(self):
        """Load only lightweight tools for Android."""
        # Core tools that work on Android
        self.tools["file"] = self._file_tool
        self.tools["terminal"] = self._terminal_tool
        self.tools["web"] = self._web_tool
        self.tools["skills"] = self._skills_tool
        self.tools["session_search"] = self._session_search_tool
        self.tools["todo"] = self._todo_tool
        logger.info(f"Loaded {len(self.tools)} lightweight tools")

    def _file_tool(self, args: dict) -> str:
        """File operations."""
        action = args.get("action", "read")
        path = args.get("path", "")
        if action == "read":
            return Path(path).read_text() if Path(path).exists() else "File not found"
        elif action == "write":
            Path(path).write_text(args.get("content", ""))
            return f"Written to {path}"
        elif action == "list":
            return "\n".join([str(p) for p in Path(path).iterdir()])
        return f"Unknown action: {action}"

    def _terminal_tool(self, args: dict) -> str:
        """Terminal operations."""
        import subprocess
        cmd = args.get("command", "")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
            return result.stdout.decode()
        except Exception as e:
            return f"Error: {e}"

    def _web_tool(self, args: dict) -> str:
        """Web requests (lightweight)."""
        url = args.get("url", "")
        method = args.get("method", "GET")
        try:
            if method == "GET":
                response = requests.get(url, timeout=15)
            elif method == "POST":
                response = requests.post(url, json=args.get("data"), timeout=15)
            else:
                response = requests.request(method, url, timeout=15)
            return response.text[:2000]  # Limit output size
        except Exception as e:
            return f"Web request failed: {e}"

    def _skills_tool(self, args: dict) -> str:
        """Skill operations."""
        action = args.get("action", "list")
        if action == "list":
            skills_dir = Path(os.environ.get("HERMES_HOME", "~/.hermes-android")).expanduser() / "skills"
            if skills_dir.exists():
                return "\n".join([d.name for d in skills_dir.iterdir() if d.is_dir()])
            return "No skills found"
        return f"Unknown skill action: {action}"

    def _session_search_tool(self, args: dict) -> str:
        """Session search (local SQLite)."""
        query = args.get("query", "")
        db_path = Path(os.environ.get("HERMES_HOME", "~/.hermes-android")).expanduser() / "sessions.db"
        if not db_path.exists():
            return "No sessions found"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE content LIKE ?", (f"%{query}%",))
        results = cursor.fetchall()
        conn.close()
        return "\n".join([str(r) for r in results])

    def _todo_tool(self, args: dict) -> str:
        """Todo list operations."""
        action = args.get("action", "list")
        todos_path = Path(os.environ.get("HERMES_HOME", "~/.hermes-android")).expanduser() / "todos.json"
        if action == "list":
            if todos_path.exists():
                return todos_path.read_text()
            return "[]"
        elif action == "add":
            item = args.get("item", "")
            todos = json.loads(todos_path.read_text()) if todos_path.exists() else []
            todos.append({"id": len(todos), "content": item, "status": "pending"})
            todos_path.write_text(json.dumps(todos, indent=2))
            return f"Added: {item}"
        return f"Unknown todo action: {action}"

    def run(self, message: str) -> str:
        """Process a message and return response."""
        logger.info(f"Message: {message}")
        self.history.append({"role": "user", "content": message})

        # Simple response for demo - in real implementation would call AI model
        response = f"Received: {message}\nAgent running in lightweight Android mode."
        
        self.history.append({"role": "assistant", "content": response})
        return response

    def run_conversation(self, user_message: str) -> dict:
        """Full conversation loop."""
        messages = [{"role": "system", "content": "You are Hermes Android Agent."}]
        messages.append({"role": "user", "content": user_message})

        response = self.run(user_message)
        messages.append({"role": "assistant", "content": response})

        return {
            "final_response": response,
            "messages": messages,
            "session_id": self.session_id or "android_session"
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Agent for Android")
    parser.add_argument("message", nargs="?", help="Message to process")
    args = parser.parse_args()

    agent = LightweightAgent()
    if args.message:
        result = agent.run_conversation(args.message)
        print(json.dumps(result, indent=2))
    else:
        print("Hermes Android Agent started. Ready for commands.")
        print("Type 'exit' to quit.")
        while True:
            try:
                msg = input("> ")
                if msg.lower() == "exit":
                    break
                result = agent.run_conversation(msg)
                print(f"Response: {result['final_response']}")
            except KeyboardInterrupt:
                break
