"""Entrypoint driver for the parity matrix: the ACP stdio server (``hermes acp``).

Follows the driver contract in ``_drive_cli``. The client side speaks
newline-delimited JSON-RPC 2.0 itself (no SDK client) so the driver controls
every byte and the shutdown path: ``initialize`` -> ``session/new`` (ACP's
documented cwd channel) -> ``session/prompt``, collecting
``agent_message_chunk`` text from ``session/update`` notifications, then stdin
EOF — the way an editor host closes the server.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from typing import Any

from tests.e2e.core.parity._helpers import TURN_TIMEOUT, DriveResult, ParityHome, hermes_argv, terminate
from tests.fakes.fake_llm_provider import FakeLLMServer

ACP_TOOLSET = "hermes-acp"  # acp_adapter/session.py::_expand_acp_enabled_toolsets default
EOF_EXIT_TIMEOUT = 30.0


class _AcpClient:
    """Minimal ACP client over a subprocess's stdio (one reader thread)."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self._next_id = 0
        self._inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.chunks: list[str] = []
        self.updates: list[dict[str, Any]] = []
        self.server_requests: list[str] = []
        self.stray_stdout: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except json.JSONDecodeError:
                # stdout must stay pure JSON-RPC; keep evidence instead of crashing.
                self.stray_stdout.append(line)
        self._inbox.put(None)

    def _send(self, msg: dict[str, Any]) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()  # type: ignore[union-attr]

    def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        self._next_id += 1
        rid = self._next_id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"ACP {method} got no response within {timeout}s")
            try:
                msg = self._inbox.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg is None:
                raise AssertionError(f"ACP server closed stdout during {method} (rc={self.proc.poll()})")
            if "method" in msg:
                self._handle_incoming(msg)
            elif msg.get("id") == rid:
                if "error" in msg:
                    raise AssertionError(f"ACP {method} failed: {msg['error']}")
                return msg.get("result") or {}

    def _handle_incoming(self, msg: dict[str, Any]) -> None:
        method, params = msg["method"], msg.get("params") or {}
        if "id" not in msg:  # notification
            if method == "session/update":
                update = params.get("update") or {}
                self.updates.append(update)
                content = update.get("content") or {}
                if update.get("sessionUpdate") == "agent_message_chunk" and content.get("type") == "text":
                    self.chunks.append(content.get("text", ""))
            return
        self.server_requests.append(method)
        if method == "session/request_permission":
            # An editor user clicking "allow once" (fall back to the first offered option).
            options = params.get("options") or []
            pick = next((o for o in options if o.get("kind") == "allow_once"), options[0] if options else None)
            outcome = ({"outcome": "selected", "optionId": pick["optionId"]} if pick
                       else {"outcome": "cancelled"})
            self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}})
            return
        # We advertise no fs/terminal client capabilities, so anything else is unexpected.
        self._send({"jsonrpc": "2.0", "id": msg["id"],
                    "error": {"code": -32601, "message": f"client does not implement {method}"}})


def drive_acp(ph: ParityHome, srv: FakeLLMServer, prompt: str) -> DriveResult:
    stderr_path = ph.root / "acp_stderr.log"
    with open(stderr_path, "w", encoding="utf-8") as stderr_fh:
        proc = subprocess.Popen(
            hermes_argv("acp"), cwd=ph.project, env=ph.env(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr_fh, text=True, bufsize=1,
        )
        client = _AcpClient(proc)
        graceful = True
        stop_reason = None
        try:
            init = client.request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
                "clientInfo": {"name": "parity-suite", "version": "1.0"},
            }, timeout=60)
            session = client.request("session/new", {"cwd": str(ph.project), "mcpServers": []},
                                     timeout=120)
            session_id = session["sessionId"]
            result = client.request("session/prompt", {
                "sessionId": session_id, "prompt": [{"type": "text", "text": prompt}],
            }, timeout=TURN_TIMEOUT)
            stop_reason = result.get("stopReason")
        finally:
            # Normal host shutdown: close the pipe and let the server exit on EOF.
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            try:
                proc.wait(timeout=EOF_EXIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                graceful = False
                terminate(proc)
    return DriveResult(
        final_text="".join(client.chunks),
        toolset=ACP_TOOLSET,
        cwd_channel="session/new cwd",
        graceful_exit=graceful,
        extra={
            "stop_reason": stop_reason,
            "returncode": proc.returncode,
            "agent_info": (init or {}).get("agentInfo"),
            "server_requests": client.server_requests,
            "stray_stdout": client.stray_stdout[:20],
            "stderr_log": str(stderr_path),
        },
    )
