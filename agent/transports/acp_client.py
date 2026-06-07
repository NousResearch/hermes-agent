"""ACP client — JSON-RPC 2.0 stdio transport for ACP-compliant agents.

Speaks the Agent Client Protocol (ACP spec v1, schema ref v0.11.2).
Transport is newline-delimited JSON-RPC 2.0 over stdio: spawn an ACP-compliant
agent process, do an ``initialize`` handshake, then drive ``session/new`` +
``session/prompt`` and consume streaming ``session/update`` notifications until
``session/prompt`` returns a PromptResponse.

Wire method names (from acp.meta.AGENT_METHODS / CLIENT_METHODS):
  agent-side requests:  initialize, session/new, session/prompt, session/close
  agent-side notifs:    session/cancel
  server-initiated:     session/update  (streaming chunks, fs/*, terminal/*)

This module is the wire-level speaker only. Higher-level concerns (session
lifecycle, event projection into Hermes messages, retry policy) live in sibling
modules.

Status: optional opt-in runtime gated behind ``api_mode == "acp_client"``.
Hermes' default tool dispatch is unchanged when this runtime is not selected.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ACPClientError(RuntimeError):
    """Raised on JSON-RPC errors from the ACP agent."""

    code: int
    message: str
    data: Optional[Any] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"ACP agent error {self.code}: {self.message}"


@dataclass
class _Pending:
    queue: queue.Queue
    method: str
    sent_at: float = field(default_factory=time.time)


class ACPClient:
    """Minimal JSON-RPC 2.0 client for ACP-compliant agents over stdio.

    Threading model:
      - Spawning thread (caller) drives request/response pairs synchronously.
      - One reader thread parses stdout, dispatches replies to the right
        pending future, and routes server-initiated requests and notifications
        to bounded queues that the caller drains on their own cadence.
      - One reader thread captures stderr for diagnostics.

    Intentionally NOT async. AIAgent.run_conversation() is synchronous and
    runs on the main thread; layering asyncio just to drive a stdio child
    creates surprising interrupt semantics. We use blocking queues with
    timeouts for cancellation.

    Three-arm dispatch (mirrors codex_app_server.py design):
      _pending          — our outgoing requests, awaiting responses
      _server_requests  — server-initiated requests (fs/*, terminal/*, permission)
      _notifications    — server push notifications (session/update streaming chunks)
    """

    def __init__(
        self,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        self._command = command
        spawn_env = os.environ.copy()
        if env:
            spawn_env.update(env)

        cmd = [command] + list(args or [])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=spawn_env,
        )
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._notifications: queue.Queue = queue.Queue()
        self._server_requests: queue.Queue = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._initialized = False

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    # ---------- lifecycle ----------

    # Minimal capabilities Hermes declares to ACP agents.  We explicitly
    # opt out of fs and terminal proxying because Hermes drives its own tool
    # executor — the agent must not expect to call back into us for those.
    # This mirrors what _handle_server_request() declines in acp_client_session.py
    # and lets conformant agents skip their fs/terminal request flow entirely.
    _DEFAULT_CLIENT_CAPABILITIES: dict = {
        "fs": {"readTextFile": False, "writeTextFile": False},
        "terminal": False,
    }

    def initialize(
        self,
        client_name: str = "hermes",
        client_version: str = "0.1",
        capabilities: Optional[dict] = None,
        timeout: float = 10.0,
    ) -> dict:
        """Send ACP ``initialize`` + ``initialized`` handshake.

        Returns the server's InitializeResponse dict
        (agent_info, agent_capabilities, protocol_version).

        ``capabilities`` defaults to ``_DEFAULT_CLIENT_CAPABILITIES`` when not
        supplied, explicitly advertising that Hermes does not proxy fs or
        terminal requests to the calling client.
        """
        if self._initialized:
            raise RuntimeError("already initialized")
        if capabilities is None:
            capabilities = self._DEFAULT_CLIENT_CAPABILITIES
        params = {
            "protocolVersion": 1,
            "clientInfo": {
                "name": client_name,
                "version": client_version,
            },
            "clientCapabilities": capabilities,
        }
        result = self.request("initialize", params, timeout=timeout)
        self._initialized = True
        return result

    def close(self, timeout: float = 3.0) -> None:
        """Close stdin and wait for the subprocess to exit, escalating to kill."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
            except Exception:
                pass

    def __enter__(self) -> "ACPClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- send/receive ----------

    def request(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: float = 30.0,
    ) -> dict:
        """Send a JSON-RPC request and block on the response.

        Returns the ``result`` dict on success.
        Raises ACPClientError on a JSON-RPC ``error`` response.
        Raises TimeoutError if no response arrives within ``timeout`` seconds.
        """
        rid = self._take_id()
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = _Pending(queue=q, method=method)
        self._send({"id": rid, "method": method, "params": params or {}})
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(
                f"ACP method {method!r} timed out after {timeout}s"
            )
        if "error" in msg:
            err = msg["error"]
            raise ACPClientError(
                code=err.get("code", -1),
                message=err.get("message", ""),
                data=err.get("data"),
            )
        return msg.get("result") or {}

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        self._send({"method": method, "params": params or {}})

    def respond(self, request_id: Any, result: dict) -> None:
        """Reply to a server-initiated request (fs/read, fs/write, permission)."""
        self._send({"id": request_id, "result": result})

    def respond_error(
        self, request_id: Any, code: int, message: str, data: Optional[Any] = None
    ) -> None:
        """Reply to a server-initiated request with an error."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"id": request_id, "error": err})

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next streaming notification (session/update), or None on timeout.

        timeout=0.0 means non-blocking. Use small positive timeouts inside the
        AIAgent turn loop to interleave reads with interrupt checks.
        """
        try:
            if timeout <= 0:
                return self._notifications.get_nowait()
            return self._notifications.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next server-initiated request (fs/*, terminal/*, permission)."""
        try:
            if timeout <= 0:
                return self._server_requests.get_nowait()
            return self._server_requests.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------- diagnostics ----------

    def stderr_tail(self, n: int = 20) -> list[str]:
        """Return last n lines of the agent's stderr (for error reports)."""
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    # ---------- internals ----------

    def _take_id(self) -> int:
        with self._pending_lock:
            rid = self._next_id
            self._next_id += 1
            return rid

    def _send(self, obj: dict) -> None:
        if self._closed:
            raise RuntimeError("ACP client is closed")
        if self._proc.stdin is None:
            raise RuntimeError("ACP agent stdin not available")
        # _send_lock serialises writes from multiple threads. Unlike Codex
        # (where session/prompt returns immediately), ACP session/prompt blocks
        # for the whole turn, so callers wrap it in a background thread while
        # the main thread drains notifications — two concurrent writers on the
        # same BufferedWriter would interleave mid-JSON-line and corrupt the wire.
        with self._send_lock:
            try:
                self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise RuntimeError(
                    f"ACP agent stdin closed unexpectedly: {exc}"
                ) from exc

    def _read_stdout(self) -> None:
        if self._proc.stdout is None:
            return
        try:
            for line in iter(self._proc.stdout.readline, b""):
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    with self._stderr_lock:
                        self._stderr_lines.append(
                            f"<non-json on stdout> {line[:200]!r}"
                        )
                    continue
                self._dispatch(msg)
        except Exception as exc:
            with self._stderr_lock:
                self._stderr_lines.append(f"<stdout reader error> {exc}")

    def _dispatch(self, msg: dict) -> None:
        """Three-arm dispatch:
        1. Reply (id + result/error, no method) → unblock the pending request future.
        2. Server-initiated request (id + method) → _server_requests queue.
        3. Notification (no id, method present) → _notifications queue.
        """
        # Reply
        if "id" in msg and ("result" in msg or "error" in msg):
            with self._pending_lock:
                pending = self._pending.pop(msg["id"], None)
            if pending is not None:
                try:
                    pending.queue.put_nowait(msg)
                except queue.Full:  # pragma: no cover - defensive
                    pass
            return
        # Server-initiated request (has id + method)
        if "id" in msg and "method" in msg:
            self._server_requests.put(msg)
            return
        # Notification (no id, has method)
        if "method" in msg:
            self._notifications.put(msg)

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        try:
            for line in iter(self._proc.stderr.readline, b""):
                if not line:
                    break
                with self._stderr_lock:
                    self._stderr_lines.append(
                        line.decode("utf-8", "replace").rstrip()
                    )
                    # Bound memory: keep last 500 lines.
                    if len(self._stderr_lines) > 500:
                        self._stderr_lines = self._stderr_lines[-500:]
        except Exception:  # pragma: no cover
            pass
