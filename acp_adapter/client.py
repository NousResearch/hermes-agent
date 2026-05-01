"""Small JSON-RPC client for driving Hermes ACP sessions.

This module is intentionally transport-level and dependency-light: it speaks
newline-delimited JSON-RPC over ``hermes acp`` stdio without importing the ACP
Python package.  It is used by orchestration tools that need to command another
Hermes profile as a real persistent agent rather than as an in-process
``delegate_task`` child.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


_AUTH_METHOD_ENV = "HERMES_ACP_AUTH_METHOD"


def _select_auth_method_id(
    initialize_result: dict[str, Any],
    preferred_method: str | None = None,
) -> str | None:
    """Choose the ACP auth method id to acknowledge, if the server advertises one."""
    preferred = (preferred_method or os.environ.get(_AUTH_METHOD_ENV, "")).strip().lower()
    auth_methods = (
        initialize_result.get("authMethods")
        or initialize_result.get("auth_methods")
        or []
    )
    if not isinstance(auth_methods, list):
        return None

    method_ids: list[str] = []
    for method in auth_methods:
        if not isinstance(method, dict):
            continue
        raw_id = str(method.get("id") or "").strip()
        if raw_id:
            method_ids.append(raw_id)

    if not method_ids:
        return None

    if preferred:
        for method_id in method_ids:
            if method_id.lower() == preferred:
                return method_id

    return method_ids[0]


def resolve_hermes_bin() -> str:
    """Resolve the Hermes executable used for ACP subprocesses."""
    explicit = os.environ.get("HERMES_BIN", "").strip()
    if explicit:
        return explicit

    which_hermes = shutil.which("hermes")
    if which_hermes:
        return which_hermes

    # Developer checkout fallback: run the current interpreter against the CLI
    # package.  This keeps tests and editable installs useful even before the
    # console script is on PATH.
    return sys.executable


def build_hermes_acp_command(profile: str, hermes_bin: str | None = None) -> list[str]:
    """Return the argv used to start a Hermes ACP server for *profile*."""
    resolved = hermes_bin or resolve_hermes_bin()
    profile = (profile or "default").strip() or "default"

    if Path(resolved).name.startswith("python"):
        cmd = [resolved, "-m", "hermes_cli.main"]
    else:
        cmd = [resolved]

    cmd.extend(["-p", profile])
    cmd.append("acp")
    return cmd


def build_hermes_acp_env(
    *,
    hermes_bin: str | None = None,
    use_profile_toolsets: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the environment for an agent-control ACP subprocess."""
    env = os.environ.copy()
    if use_profile_toolsets:
        env["HERMES_ACP_USE_PROFILE_TOOLSETS"] = "1"

    # When running from a source checkout before the console script is
    # installed, the fallback command is ``python -m hermes_cli.main``.  The
    # controlled agent may run in an arbitrary project cwd, so preserve this
    # checkout on PYTHONPATH for that child process.
    resolved = hermes_bin or resolve_hermes_bin()
    if Path(resolved).name.startswith("python"):
        project_root = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        paths = [project_root]
        if existing:
            paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(paths)

    env.update(extra_env or {})
    return env


class HermesACPClient:
    """Line-oriented JSON-RPC client for one Hermes profile."""

    def __init__(
        self,
        profile: str,
        cwd: str | None = None,
        *,
        hermes_bin: str | None = None,
        approval_policy: str = "deny",
        use_profile_toolsets: bool = True,
        extra_env: dict[str, str] | None = None,
    ):
        self.profile = (profile or "default").strip() or "default"
        self.cwd = cwd or str(Path.home())
        self.hermes_bin = hermes_bin
        self.approval_policy = approval_policy
        self.use_profile_toolsets = use_profile_toolsets
        self.extra_env = dict(extra_env or {})
        self.proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stderr_tail: list[str] = []
        self.notifications: list[dict[str, Any]] = []
        self.session_id: str | None = None

    # ---- process and IO -------------------------------------------------

    def _start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        self.proc = None
        cmd = build_hermes_acp_command(self.profile, self.hermes_bin)
        env = build_hermes_acp_env(
            hermes_bin=self.hermes_bin,
            use_profile_toolsets=self.use_profile_toolsets,
            extra_env=self.extra_env,
        )
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
            env=env,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        if not self.proc or self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def _read_stderr(self) -> None:
        if not self.proc or self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            self._stderr_tail.append(line.rstrip())
            if len(self._stderr_tail) > 80:
                self._stderr_tail.pop(0)

    def _write_message(self, payload: dict[str, Any]) -> None:
        self._start()
        assert self.proc is not None and self.proc.stdin is not None
        with self._lock:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()

    def _respond_to_server_request(self, message: dict[str, Any]) -> None:
        """Best-effort response to server-initiated JSON-RPC requests.

        The important case is ACP permission requests.  If we do not answer,
        the target agent can hang waiting for a decision.  Agent-control runs
        default to fail-closed, so unknown permission requests are denied.
        """
        req_id = message.get("id")
        if req_id is None or not message.get("method"):
            return

        method = str(message.get("method") or "")
        lowered = method.replace("/", "_").replace("-", "_").lower()
        if "permission" in lowered:
            if str(self.approval_policy).lower() in {"allow", "allow_once", "once"}:
                result = {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
            else:
                result = {"outcome": {"outcome": "cancelled"}}
            self._write_message({"jsonrpc": "2.0", "id": req_id, "result": result})
            return

        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        )

    def _send_rpc_raw(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> Any:
        self._start()
        assert self.proc is not None

        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                continue

            if message.get("id") == req_id:
                if "error" in message:
                    raise RuntimeError(f"ACP error for {method}: {message['error']}")
                return message.get("result")

            if message.get("id") is not None and message.get("method"):
                self._respond_to_server_request(message)
                continue

            self.notifications.append(message)

        stderr_text = "\n".join(self._stderr_tail).strip()
        if self.proc.poll() is not None:
            raise RuntimeError(stderr_text or f"ACP process exited during {method}.")
        raise TimeoutError(f"Timeout waiting for {method} response.")

    def _send_rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        result = self._send_rpc_raw(method, params=params, timeout=timeout)
        return result if isinstance(result, dict) else {"value": result}

    # ---- ACP calls ------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        return self._send_rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "hermes-agent-control",
                    "version": "1.0",
                },
            },
            timeout=30.0,
        )

    def maybe_authenticate(self, initialize_result: dict[str, Any]) -> str | None:
        method_id = _select_auth_method_id(initialize_result)
        if not method_id:
            return None
        self._send_rpc(
            "authenticate",
            {"methodId": method_id, "args": {}},
            timeout=30.0,
        )
        return method_id

    def connect(self) -> str | None:
        init_result = self.initialize()
        return self.maybe_authenticate(init_result)

    def new_session(self, cwd: str | None = None) -> str:
        result = self._send_rpc(
            "session/new",
            {"cwd": cwd or self.cwd, "mcpServers": []},
            timeout=30.0,
        )
        session_id = str(result.get("sessionId") or result.get("session_id") or "").strip()
        if not session_id:
            raise RuntimeError("session/new returned no sessionId.")
        self.session_id = session_id
        return session_id

    def load_session(self, session_id: str, cwd: str | None = None) -> str:
        result = self._send_rpc_raw(
            "session/load",
            {"sessionId": session_id, "cwd": cwd or self.cwd, "mcpServers": []},
            timeout=30.0,
        )
        if result is None:
            raise RuntimeError(f"session/load could not find sessionId: {session_id}")
        self.session_id = session_id
        return session_id

    def resume_session(self, session_id: str, cwd: str | None = None) -> str:
        result = self._send_rpc(
            "session/resume",
            {"sessionId": session_id, "cwd": cwd or self.cwd, "mcpServers": []},
            timeout=30.0,
        )
        resumed_id = str(result.get("sessionId") or result.get("session_id") or session_id).strip()
        if not resumed_id:
            raise RuntimeError("session/resume returned no sessionId.")
        self.session_id = resumed_id
        return resumed_id

    def fork_session(self, session_id: str, cwd: str | None = None) -> str:
        result = self._send_rpc(
            "session/fork",
            {"sessionId": session_id, "cwd": cwd or self.cwd, "mcpServers": []},
            timeout=30.0,
        )
        new_id = str(result.get("sessionId") or result.get("session_id") or "").strip()
        if not new_id:
            raise RuntimeError("session/fork returned no sessionId.")
        self.session_id = new_id
        return new_id

    def cancel(self, session_id: str) -> None:
        self._send_rpc("session/cancel", {"sessionId": session_id}, timeout=15.0)

    def prompt(self, session_id: str, text: str, timeout: float = 600.0) -> dict[str, Any]:
        self._start()
        assert self.proc is not None and self.proc.stdin is not None

        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": text}],
                },
            }
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()

        text_parts: list[str] = []
        updates: list[dict[str, Any]] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                continue

            if message.get("id") is not None and message.get("method"):
                self._respond_to_server_request(message)
                continue

            if message.get("method") == "session/update":
                params = message.get("params") or {}
                update = params.get("update") or {}
                if isinstance(update, dict):
                    updates.append(update)
                    kind = update.get("sessionUpdate") or update.get("session_update")
                    if kind == "agent_message_chunk":
                        content = update.get("content") or {}
                        if isinstance(content, dict) and content.get("type") == "text":
                            text_parts.append(str(content.get("text") or ""))
                continue

            if message.get("id") != req_id:
                self.notifications.append(message)
                continue

            if "error" in message:
                raise RuntimeError(f"ACP error for session/prompt: {message['error']}")

            result = message.get("result", {}) or {}
            if isinstance(result, dict) and not text_parts:
                for block in result.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(str(block.get("text") or ""))
            return {
                "stop_reason": result.get("stopReason") or result.get("stop_reason"),
                "text": "".join(text_parts),
                "content": result.get("content", []) if isinstance(result, dict) else [],
                "usage": result.get("usage", {}) if isinstance(result, dict) else {},
                "updates": updates,
            }

        stderr_text = "\n".join(self._stderr_tail).strip()
        if self.proc.poll() is not None:
            raise RuntimeError(stderr_text or "ACP process exited during session/prompt.")
        raise TimeoutError(f"Timeout ({timeout}s) waiting for session/prompt response.")

    def close(self) -> None:
        if not self.proc:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        finally:
            self.proc = None
