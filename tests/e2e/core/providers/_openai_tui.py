"""Minimal stdio JSON-RPC driver for a REAL ``python -m tui_gateway.entry`` process —
the backend the Ink TUI and Desktop talk to (one long-lived agent across turns)."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from tests.e2e.core.providers._openai_helpers import Home


class TuiGateway:
    def __init__(self, h: Home) -> None:
        env = h.env({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(h.project), "HERMES_YOLO_MODE": "1"})
        self.stderr_path = h.root / f"tui-gateway-{time.monotonic_ns()}.log"
        self._stderr = open(self.stderr_path, "wb")  # noqa: SIM115 - closed in close()
        self.proc = subprocess.Popen([sys.executable, "-m", "tui_gateway.entry"], cwd=h.project, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0)
        self._ids = itertools.count(1)
        self._cv = threading.Condition()
        self._responses: dict[int, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        threading.Thread(target=self._read, name="tui-rpc", daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                continue
            with self._cv:
                if frame.get("method") == "event":
                    self.events.append(dict(frame.get("params") or {}))
                elif "id" in frame:
                    self._responses[frame["id"]] = frame
                self._cv.notify_all()
        with self._cv:
            self._cv.notify_all()

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 90.0) -> dict[str, Any]:
        rid = next(self._ids)
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                                          "params": params or {}}).encode() + b"\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        with self._cv:
            while rid not in self._responses:
                left = deadline - time.monotonic()
                if left <= 0 or self.proc.poll() is not None:
                    raise AssertionError(f"no reply to {method} (rc={self.proc.poll()}){self.tail()}")
                self._cv.wait(min(left, 0.5))
            frame = self._responses.pop(rid)
        if "error" in frame:
            raise AssertionError(f"{method} -> {frame['error']}{self.tail()}")
        return frame["result"]

    def wait_event(self, pred: Callable[[dict[str, Any]], bool], timeout: float, start: int = 0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        seen = start
        with self._cv:
            while True:
                while seen < len(self.events):
                    ev = self.events[seen]
                    seen += 1
                    if pred(ev):
                        return {**ev, "_i": seen - 1}
                left = deadline - time.monotonic()
                if left <= 0 or self.proc.poll() is not None:
                    raise AssertionError(f"event never arrived within {timeout}s{self.tail()}")
                self._cv.wait(min(left, 0.5))

    def turn(self, sid: str, text: str, timeout: float = 120.0) -> str:
        """Submit one prompt and return the completed assistant text once the session settles."""
        start = len(self.events)
        reply = self.call("prompt.submit", {"session_id": sid, "text": text})
        assert reply.get("status") == "streaming", reply
        done = self.wait_event(lambda e: e.get("type") == "message.complete" and e.get("session_id") == sid,
                               timeout, start)
        self.wait_event(lambda e: e.get("type") == "session.info" and e.get("session_id") == sid
                        and (e.get("payload") or {}).get("running") is False, 60.0, done["_i"] + 1)
        payload = done.get("payload") or {}
        text_out = payload.get("text")
        return text_out if isinstance(text_out, str) else json.dumps(text_out)

    def close(self, timeout: float = 30.0) -> int | None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            return self.proc.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait(timeout=10)
            return None
        finally:
            self._stderr.close()

    def tail(self, n: int = 3000) -> str:
        try:
            self._stderr.flush()
            return "\n--- tui_gateway stderr ---\n" + Path(self.stderr_path).read_bytes()[-n:].decode(errors="replace")
        except (OSError, ValueError):
            return ""
