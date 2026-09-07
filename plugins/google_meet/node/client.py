"""Gateway-side RPC client for a remote meet node: one short-lived sync WebSocket per call, so
non-async tool handlers need no persistent connection. ``websockets`` is imported lazily."""

from __future__ import annotations

from typing import Any, Dict, Optional

from plugins.google_meet.node import protocol as _proto


class NodeClient:
    """Thin synchronous WS client matching the server's request surface."""

    def __init__(self, url: str, token: str, timeout: float = 10.0) -> None:
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")
        self.url = url
        self.token = token
        self.timeout = float(timeout)

    def _rpc(self, type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one request, return its payload dict; RuntimeError on error envelope / id mismatch."""
        try:
            from websockets.sync.client import connect  # type: ignore
        except ImportError as exc:
            raise RuntimeError("NodeClient requires the 'websockets' package. "
                               "Install it with: pip install websockets") from exc
        req = _proto.make_request(type, self.token, payload)
        with connect(self.url, open_timeout=self.timeout, close_timeout=self.timeout) as ws:
            ws.send(_proto.encode(req))
            resp = _proto.decode(ws.recv(timeout=self.timeout))
        if resp.get("type") == "error":
            raise RuntimeError(f"node error: {resp.get('error', '<unknown>')}")
        if resp.get("id") != req["id"]:
            raise RuntimeError(f"response id mismatch: sent {req['id']}, got {resp.get('id')!r}")
        payload_out = resp.get("payload")
        if not isinstance(payload_out, dict):  # pong envelopes also carry a dict payload
            raise RuntimeError("response missing payload dict")
        return payload_out

    def start_bot(self, url: str, guest_name: str = "Hermes Agent", duration: Optional[str] = None,
                  headed: bool = False, mode: str = "transcribe") -> Dict[str, Any]:
        payload: Dict[str, Any] = {"url": url, "guest_name": guest_name, "headed": bool(headed), "mode": mode}
        if duration is not None:
            payload["duration"] = duration
        return self._rpc("start_bot", payload)

    def stop(self) -> Dict[str, Any]:
        return self._rpc("stop", {})

    def status(self) -> Dict[str, Any]:
        return self._rpc("status", {})

    def transcript(self, last: Optional[int] = None) -> Dict[str, Any]:
        return self._rpc("transcript", {} if last is None else {"last": int(last)})

    def say(self, text: str) -> Dict[str, Any]:
        return self._rpc("say", {"text": str(text)})

    def ping(self) -> Dict[str, Any]:
        return self._rpc("ping", {})
