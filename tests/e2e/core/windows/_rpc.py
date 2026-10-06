"""Minimal stdio JSON-RPC client for ``python -m tui_gateway.entry`` — the process the Ink
TUI spawns on every OS. Frames are UTF-8 JSON lines with non-ASCII left literal, exactly
what Node's ``JSON.stringify`` + ``stdin.write`` produce, so the pipe encoding the gateway
applies on Windows is part of what is under test."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from tests.e2e.core.windows._helpers import TURN_TIMEOUT, WinHome

READY_TIMEOUT = 120.0


class StdioGateway:
    def __init__(self, home: WinHome) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tui_gateway.entry"], cwd=home.project, env=home.env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._frames: "queue.Queue[bytes | None]" = queue.Queue()
        self._stderr: list[bytes] = []
        self.events: list[dict[str, Any]] = []
        self._responses: dict[str, dict[str, Any]] = {}
        self._next = 0
        threading.Thread(target=self._pump_out, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    def _pump_out(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._frames.put(line)
        self._frames.put(None)

    def _pump_err(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line)

    @property
    def stderr(self) -> str:
        return b"".join(self._stderr).decode("utf-8", errors="replace")

    def send(self, obj: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def _pump(self, timeout: float) -> None:
        try:
            raw = self._frames.get(timeout=timeout)
        except queue.Empty:
            return
        if raw is None:
            raise AssertionError(f"tui_gateway closed stdout (rc={self.proc.poll()}):\n{self.stderr[-3000:]}")
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return  # stray non-protocol output; the protocol stream is what matters
        if msg.get("method") == "event":
            self.events.append(msg.get("params") or {})
        elif "id" in msg and "method" in msg:
            self.send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "unsupported"}})
        elif "id" in msg:
            self._responses[str(msg["id"])] = msg

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = TURN_TIMEOUT) -> Any:
        self._next += 1
        rid = f"win-{self._next}"
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while rid not in self._responses:
            if time.monotonic() > deadline:
                raise AssertionError(f"{method}: no response within {timeout}s\n{self.stderr[-2000:]}")
            self._pump(0.25)
        resp = self._responses.pop(rid)
        assert "error" not in resp, f"{method} failed: {resp['error']}"
        return resp.get("result")

    def wait_event(self, etype: str, pred: Callable[[dict], bool] = lambda _e: True,
                   timeout: float = TURN_TIMEOUT) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        seen = 0
        while True:
            for ev in self.events[seen:]:
                if ev.get("type") == etype and pred(ev):
                    return ev
            seen = len(self.events)
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"no {etype!r} within {timeout}s; saw {[e.get('type') for e in self.events][-20:]}\n"
                    f"{self.stderr[-2000:]}")
            self._pump(0.25)

    def turn(self, text: str) -> str:
        self.wait_event("gateway.ready", timeout=READY_TIMEOUT)
        sid = self.call("session.create", {})["session_id"]
        self.call("prompt.submit", {"session_id": sid, "text": text})
        done = self.wait_event("message.complete", lambda e: e.get("session_id") == sid)
        payload = done.get("payload") or {}
        text = payload.get("text")
        return text if isinstance(text, str) else json.dumps(payload)

    def close(self) -> int | None:
        """Normal stop for a stdio gateway: the client closes the pipe."""
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
            return None
