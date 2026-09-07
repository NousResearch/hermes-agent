"""Codex app-server JSON-RPC client (newline-delimited JSON-RPC 2.0 over stdio, codex 0.125+).

``initialize`` handshake, then ``thread/start`` + ``turn/start`` with streaming
``item/*`` notifications until ``turn/completed``. Wire-level speaker only —
projection, approvals and transcript handling live in sibling modules.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Optional

from tools.environments.local import hermes_subprocess_env

MIN_CODEX_VERSION = (0, 125, 0)


@dataclass
class CodexAppServerError(RuntimeError):
    """Raised on JSON-RPC errors from the app-server."""

    code: int
    message: str
    data: Optional[Any] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"codex app-server error {self.code}: {self.message}"


class CodexAppServerClient:
    """Minimal synchronous JSON-RPC 2.0 client for ``codex app-server`` over stdio.

    One reader thread routes replies to pending queues and notifications / server
    requests to queues; another captures stderr. Deliberately NOT async:
    AIAgent.run_conversation() is synchronous and cancels via ``turn/interrupt``.
    """

    def __init__(
        self, codex_bin: str = "codex", codex_home: Optional[str] = None,
        extra_args: Optional[list[str]] = None, env: Optional[dict[str, str]] = None,
    ) -> None:
        self._codex_bin = codex_bin
        # codex needs LLM provider creds but must not receive Tier-1 Hermes secrets (gateway/GitHub/infra tokens).
        # codex app-server is a model-driving CLI executor: it runs a model-chosen agentic loop that
        # executes shell commands, so it legitimately needs LLM provider credentials
        # (inherit_credentials=True) to authenticate against the model endpoint. But the previous
        # `os.environ.copy()` also handed it every Tier-1 Hermes secret — gateway bot tokens, GitHub auth,
        # Modal/Daytona infra tokens, the dashboard session token, AUXILIARY_* side-LLM keys,
        # GATEWAY_RELAY_* auth — none of which a coding subprocess has any use for. Route through the
        # centralized helper so Tier-1 + dynamic-internal secrets are always stripped while provider creds
        # still flow, matching copilot_acp_client (#29157 sibling spawn-site gap).
        spawn_env = hermes_subprocess_env(inherit_credentials=True)
        if env:
            spawn_env.update(env)
        if codex_home:
            spawn_env["CODEX_HOME"] = codex_home

        cmd = [codex_bin, "app-server", *(extra_args or [])]
        # Kanban workers must write handoff/status to the board DB outside the
        # workspace: keep the sandbox on, add the Kanban root as writable.
        if spawn_env.get("HERMES_KANBAN_TASK"):
            kanban_db = spawn_env.get("HERMES_KANBAN_DB")
            default_root = os.path.join(spawn_env.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "kanban")
            kanban_root = os.path.dirname(kanban_db) if kanban_db else spawn_env.get("HERMES_KANBAN_ROOT", default_root)
            cmd += [
                "-c", 'sandbox_mode="workspace-write"',
                "-c", f'sandbox_workspace_write.writable_roots=["{kanban_root}"]',
                "-c", "sandbox_workspace_write.network_access=false",
            ]
        # Codex emits tracing to stderr; default WARN keeps it quiet for users.
        spawn_env.setdefault("RUST_LOG", "warn")

        # Hide the console the codex child would otherwise flash on Windows (#56747).
        # Hide-only — stdio pipes stay intact for the app-server wire.
        # See #56747.
        from hermes_cli._subprocess_compat import windows_hide_flags

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env=spawn_env, creationflags=windows_hide_flags(),
        )
        self._next_id = 1
        self._pending: dict[int, queue.Queue] = {}  # request id -> single-slot reply queue
        self._pending_lock = threading.Lock()
        self._notifications: queue.Queue = queue.Queue()
        self._server_requests: queue.Queue = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        self._initialized = False

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    def initialize(
        self, client_name: str = "hermes", client_title: str = "Hermes Agent",
        client_version: str = "0.1", capabilities: Optional[dict] = None, timeout: float = 10.0,
    ) -> dict:
        """Send ``initialize`` + ``initialized``; return the server's InitializeResponse."""
        if self._initialized:
            raise RuntimeError("already initialized")
        params = {
            "clientInfo": {"name": client_name, "title": client_title, "version": client_version},
            "capabilities": capabilities or {},
        }
        result = self.request("initialize", params, timeout=timeout)
        self.notify("initialized")
        self._initialized = True
        return result

    def close(self, timeout: float = 3.0) -> None:
        """Close stdin and wait for the subprocess to exit, escalating to kill."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._proc.kill()
                self._proc.wait(timeout=1.0)

    def __enter__(self) -> "CodexAppServerClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
        """Send a request and block for ``result``; raise CodexAppServerError on ``error``."""
        rid, self._next_id = self._next_id, self._next_id + 1
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = q
        self._send({"id": rid, "method": method, "params": params or {}})
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"codex app-server method {method!r} timed out after {timeout}s")
        if "error" in msg:
            err = msg["error"]
            raise CodexAppServerError(code=err.get("code", -1), message=err.get("message", ""), data=err.get("data"))
        return msg.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        self._send({"method": method, "params": params or {}})

    def respond(self, request_id: Any, result: dict) -> None:
        """Reply to a server-initiated request (e.g. approval prompts)."""
        self._send({"id": request_id, "result": result})

    def respond_error(self, request_id: Any, code: int, message: str, data: Optional[Any] = None) -> None:
        """Reply to a server-initiated request with an error."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"id": request_id, "error": err})

    @staticmethod
    def _take(q: queue.Queue, timeout: float) -> Optional[dict]:
        try:
            return q.get_nowait() if timeout <= 0 else q.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next streaming notification, or None on timeout (0 = non-blocking)."""
        return self._take(self._notifications, timeout)

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next server-initiated request (e.g. exec/applyPatch approval)."""
        return self._take(self._server_requests, timeout)

    def stderr_tail(self, n: int = 20) -> list[str]:
        """Return last n lines of codex's stderr (for error reports)."""
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def _send(self, obj: dict) -> None:
        if self._closed:
            raise RuntimeError("codex app-server client is closed")
        if self._proc.stdin is None:
            raise RuntimeError("codex app-server stdin not available")
        try:
            self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError(f"codex app-server stdin closed unexpectedly: {exc}") from exc

    def _append_stderr(self, line: str) -> None:
        with self._stderr_lock:
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 500:  # bound memory
                self._stderr_lines = self._stderr_lines[-500:]

    def _read_stdout(self) -> None:
        if self._proc.stdout is None:
            return
        try:
            for line in iter(self._proc.stdout.readline, b""):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON stdout is unexpected; surface it via the stderr buffer.
                    self._append_stderr(f"<non-json on stdout> {line[:200]!r}")
                    continue
                self._dispatch(msg)
        except Exception as exc:
            self._append_stderr(f"<stdout reader error> {exc}")

    def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):  # reply
            with self._pending_lock:
                pending = self._pending.pop(msg["id"], None)
            if pending is not None:
                with contextlib.suppress(queue.Full):  # pragma: no cover - defensive
                    pending.put_nowait(msg)
        elif "method" in msg:  # server-initiated request (has id) or notification
            (self._server_requests if "id" in msg else self._notifications).put(msg)

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover
            for line in iter(self._proc.stderr.readline, b""):
                self._append_stderr(line.decode("utf-8", "replace").rstrip())


def parse_codex_version(output: str) -> Optional[tuple[int, int, int]]:
    """Parse ``codex --version`` output ("codex-cli 0.130.0 ...") into (major, minor, patch)."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    return tuple(int(g) for g in match.groups()) if match else None


def check_codex_binary(
    codex_bin: str = "codex", min_version: tuple[int, int, int] = MIN_CODEX_VERSION
) -> tuple[bool, str]:
    """Verify codex CLI is installed and meets minimum version. Returns (ok, message)."""
    try:
        proc = subprocess.run(
            [codex_bin, "--version"], capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=10, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, f"codex CLI not found at {codex_bin!r}. Install with: npm i -g @openai/codex"
    except subprocess.TimeoutExpired:
        return False, "codex --version timed out"
    if proc.returncode != 0:
        return False, f"codex --version exited {proc.returncode}: {proc.stderr.strip()}"
    version = parse_codex_version(proc.stdout)
    if version is None:
        return False, f"could not parse codex version from: {proc.stdout!r}"
    have = ".".join(map(str, version))
    if version < min_version:
        return False, f"codex {have} is older than required {'.'.join(map(str, min_version))}. Run: npm i -g @openai/codex"
    return True, have


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from dataclasses import field  # noqa: F401,E402
import time  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
