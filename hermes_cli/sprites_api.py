"""Guest-only Sprites API; the account credential never enters the VM."""
from __future__ import annotations

import http.client
import json
import socket

SOCKET_PATH = "/.sprite/api.sock"


class _Connection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(SOCKET_PATH)


def request(method: str, path: str, payload: dict | None = None):
    connection = _Connection("sprite", timeout=15)
    try:
        connection.request(method, "/v1" + path, body=json.dumps(payload) if payload is not None else None,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read(1024 * 1024 + 1)
        if response.status == 404:
            raise FileNotFoundError("Sprites resource not found")
        if not 200 <= response.status < 300 or len(body) > 1024 * 1024:
            raise RuntimeError("Sprites guest operation failed")
        if not body:
            return None
        # Mutations stream newline-delimited events. Require terminal completion.
        if "ndjson" in (response.getheader("Content-Type") or ""):
            events = [json.loads(line) for line in body.splitlines() if line.strip()]
            if not events or events[-1].get("type") != "complete" or any(e.get("type") == "error" for e in events):
                raise RuntimeError("Sprites guest operation was not acknowledged")
            return events[-1]
        return json.loads(body)
    finally:
        connection.close()
