"""
Hermes control client — zero-dependency stdlib client for the local control API.

Auto-discovers port from ~/.hermes/control_api.port.
All mutations go through send_message() which accepts any text including /commands.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional

DEFAULT_PORT = 47823  # must match gateway.control_api.DEFAULT_PORT
PORT_FILE = os.path.expanduser("~/.hermes/control_api.port")


class HermesControl:
    """Client for the Hermes control API."""

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None):
        if port is None:
            port = self._discover_port()
        self.base_url = f"http://{host}:{port}"

    @staticmethod
    def _discover_port() -> int:
        try:
            with open(PORT_FILE) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return DEFAULT_PORT

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"X-Hermes-Control": "1"}
        if data:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())
        except urllib.error.URLError as e:
            return {"error": f"Cannot reach Hermes control API: {e.reason}"}

    def health(self) -> dict:
        return self._request("GET", "/health")

    def list_sessions(self) -> dict:
        return self._request("GET", "/sessions")

    def get_session(self, session_key: str = "_any") -> dict:
        return self._request("GET", f"/sessions/{session_key}")

    def list_commands(self) -> dict:
        return self._request("GET", "/commands")

    def send_message(self, text: str, session_key: str = "_any",
                     mode: str = "interrupt") -> dict:
        """Send a message or /command into a running session.

        mode: "interrupt" (default) stops current work; "queue" waits.
        """
        return self._request("POST", f"/sessions/{session_key}/message", {
            "text": text,
            "mode": mode,
        })
