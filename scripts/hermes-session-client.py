#!/usr/bin/env python3
"""Hermes ACP session client for external orchestration.

This script speaks Hermes Agent's ACP stdio protocol so shell scripts,
CI jobs, or other agents can create and drive Hermes sessions without
embedding Hermes internals in-process.

Usage:
    hermes-session-client.py <profile> <command> [args...]

Commands:
    init
    new <cwd>
    load <session_id> [cwd]
    resume <session_id> [cwd]
    fork <session_id> [cwd]
    prompt <session_id> <text> [timeout_seconds]
    list [cwd]
    cancel <session_id>

Environment variables:
    HERMES_HOME             Base Hermes home (default: ~/.hermes)
    HERMES_AGENT_DIR        Hermes checkout dir (default: $HERMES_HOME/hermes-agent)
    HERMES_BIN              Explicit hermes binary path
    HERMES_PROFILES_BASE    Profiles dir (default: $HERMES_HOME/profiles)
    HERMES_ACP_AUTH_METHOD  Preferred ACP auth method id override
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


_HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
_HERMES_AGENT_DIR = os.environ.get(
    "HERMES_AGENT_DIR",
    str(Path(_HERMES_HOME) / "hermes-agent"),
)
_HERMES_BIN = os.environ.get(
    "HERMES_BIN",
    str(Path(_HERMES_AGENT_DIR) / "venv" / "bin" / "hermes"),
)
_PROFILES_BASE = os.environ.get(
    "HERMES_PROFILES_BASE",
    str(Path(_HERMES_HOME) / "profiles"),
)
_AUTH_METHOD_ENV = "HERMES_ACP_AUTH_METHOD"


def _resolve_hermes_bin(profile: str) -> str:
    """Resolve the hermes binary path with a few common fallbacks."""
    if os.environ.get("HERMES_BIN"):
        return _HERMES_BIN

    venv_bin = Path(_HERMES_AGENT_DIR) / "venv" / "bin" / "hermes"
    if venv_bin.exists():
        return str(venv_bin)

    profile_bin = Path(_PROFILES_BASE) / profile / "bin" / "hermes"
    if profile_bin.exists():
        return str(profile_bin)

    which_hermes = shutil.which("hermes")
    if which_hermes:
        return which_hermes

    raise RuntimeError(
        "Cannot find hermes binary. "
        "Set HERMES_BIN or HERMES_AGENT_DIR. "
        f"Searched: {venv_bin}, {profile_bin}, PATH."
    )


def _select_auth_method_id(
    initialize_result: dict[str, Any],
    preferred_method: str | None = None,
) -> str | None:
    """Choose the ACP auth method id to acknowledge, if any."""
    preferred = (preferred_method or os.environ.get(_AUTH_METHOD_ENV, "")).strip().lower()
    auth_methods = initialize_result.get("authMethods") or []
    if not isinstance(auth_methods, list):
        return None

    normalized: list[str] = []
    for method in auth_methods:
        if not isinstance(method, dict):
            continue
        raw_id = str(method.get("id") or "").strip()
        if raw_id:
            normalized.append(raw_id)

    if not normalized:
        return None

    if preferred:
        for method_id in normalized:
            if method_id.lower() == preferred:
                return method_id

    return normalized[0]


class HermesACPSessionClient:
    """Small ACP JSON-RPC client over stdio for a single Hermes profile."""

    def __init__(self, profile: str, cwd: str | None = None):
        self.profile = profile
        self.cwd = cwd or str(Path.home())
        self.bin = _resolve_hermes_bin(profile)
        self.proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._next_id = 0
        self._messages: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stderr_tail: list[str] = []
        self.session_id: str | None = None

    def _start(self) -> None:
        if self.proc is not None:
            return
        self.proc = subprocess.Popen(
            [self.bin, "-p", self.profile, "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
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
            if len(self._stderr_tail) > 50:
                self._stderr_tail.pop(0)

    def _send_rpc(self, method: str, params: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        self._start()
        assert self.proc is not None and self.proc.stdin is not None

        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
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
            if message.get("id") != req_id:
                continue
            if "error" in message:
                raise RuntimeError(f"ACP error for {method}: {message['error']}")
            return message.get("result", {}) or {}

        stderr_text = "\n".join(self._stderr_tail).strip()
        if self.proc.poll() is not None:
            raise RuntimeError(stderr_text or f"ACP process exited during {method}.")
        raise TimeoutError(f"Timeout waiting for {method} response.")

    def initialize(self) -> dict[str, Any]:
        return self._send_rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "hermes-session-client",
                    "version": "1.0",
                },
            },
        )

    def maybe_authenticate(self, initialize_result: dict[str, Any]) -> str | None:
        method_id = _select_auth_method_id(initialize_result)
        if not method_id:
            return None
        self._send_rpc("authenticate", {"methodId": method_id, "args": {}}, timeout=30.0)
        return method_id

    def new_session(self, cwd: str | None = None) -> str:
        result = self._send_rpc(
            "session/new",
            {"cwd": cwd or self.cwd, "mcpServers": []},
            timeout=30.0,
        )
        session_id = str(result.get("sessionId") or "").strip()
        if not session_id:
            raise RuntimeError("session/new returned no sessionId.")
        self.session_id = session_id
        return session_id

    def load_session(self, session_id: str, cwd: str | None = None) -> str:
        result = self._send_rpc(
            "session/load",
            {"sessionId": session_id, "cwd": cwd or self.cwd, "mcpServers": []},
            timeout=30.0,
        )
        if not isinstance(result, dict):
            raise RuntimeError("session/load returned no result.")
        self.session_id = session_id
        return session_id

    def resume_session(self, session_id: str, cwd: str | None = None) -> str:
        result = self._send_rpc(
            "session/resume",
            {"sessionId": session_id, "cwd": cwd or self.cwd, "mcpServers": []},
            timeout=30.0,
        )
        resumed_id = str(result.get("sessionId") or session_id).strip()
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
        new_id = str(result.get("sessionId") or "").strip()
        if not new_id:
            raise RuntimeError("session/fork returned no sessionId.")
        self.session_id = new_id
        return new_id

    def list_sessions(self, cwd: str | None = None) -> dict[str, Any]:
        return self._send_rpc(
            "session/list",
            {"cwd": cwd or self.cwd} if cwd else {},
            timeout=15.0,
        )

    def cancel(self, session_id: str) -> None:
        self._send_rpc("session/cancel", {"sessionId": session_id}, timeout=15.0)

    def prompt(self, session_id: str, text: str, timeout: float = 180.0) -> dict[str, Any]:
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
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                continue

            if message.get("method") == "session/update":
                update = message.get("params", {}).get("update", {})
                if update.get("sessionUpdate") == "agent_message_chunk":
                    content = update.get("content") or {}
                    if content.get("type") == "text":
                        text_parts.append(str(content.get("text") or ""))
                continue

            if message.get("id") != req_id:
                continue
            if "error" in message:
                raise RuntimeError(f"ACP error for session/prompt: {message['error']}")

            result = message.get("result", {}) or {}
            if not text_parts:
                for block in result.get("content", []) or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(str(block.get("text") or ""))
            return {
                "stop_reason": result.get("stopReason"),
                "text": "".join(text_parts),
                "content": result.get("content", []),
                "usage": result.get("usage", {}),
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


def _initialize_session(profile: str, cwd: str | None = None) -> tuple[HermesACPSessionClient, str | None]:
    client = HermesACPSessionClient(profile, cwd=cwd)
    init_result = client.initialize()
    auth_method = client.maybe_authenticate(init_result)
    return client, auth_method


def _print_usage(prog: str) -> None:
    print(
        f"Usage: {prog} <profile> <command> [args...]\n"
        "\n"
        "Commands:\n"
        "  init\n"
        "  new <cwd>\n"
        "  load <session_id> [cwd]\n"
        "  resume <session_id> [cwd]\n"
        "  fork <session_id> [cwd]\n"
        "  prompt <session_id> <text> [timeout_seconds]\n"
        "  list [cwd]\n"
        "  cancel <session_id>\n"
        "\n"
        "Environment:\n"
        "  HERMES_HOME             Base Hermes home (default: ~/.hermes)\n"
        "  HERMES_AGENT_DIR        Hermes checkout dir\n"
        "  HERMES_BIN              Explicit hermes binary path\n"
        "  HERMES_PROFILES_BASE    Profiles dir\n"
        "  HERMES_ACP_AUTH_METHOD  Preferred ACP auth method id override\n",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        _print_usage(Path(sys.argv[0]).name if sys.argv else "hermes-session-client.py")
        return 1

    profile, command, *rest = args

    try:
        if command == "init":
            client, auth_method = _initialize_session(profile)
            try:
                if auth_method:
                    print(f"INIT_OK profile={profile} auth_method={auth_method}")
                else:
                    print(f"INIT_OK profile={profile} auth_method=none")
            finally:
                client.close()
            return 0

        if command == "new":
            cwd = rest[0] if rest else str(Path.home())
            client, _ = _initialize_session(profile, cwd=cwd)
            try:
                print(f"SESSION_ID:{client.new_session(cwd)}")
            finally:
                client.close()
            return 0

        if command == "load":
            if not rest:
                raise RuntimeError("load requires <session_id> [cwd].")
            session_id = rest[0]
            cwd = rest[1] if len(rest) > 1 else str(Path.home())
            client, _ = _initialize_session(profile, cwd=cwd)
            try:
                print(f"SESSION_ID:{client.load_session(session_id, cwd)}")
            finally:
                client.close()
            return 0

        if command == "resume":
            if not rest:
                raise RuntimeError("resume requires <session_id> [cwd].")
            session_id = rest[0]
            cwd = rest[1] if len(rest) > 1 else str(Path.home())
            client, _ = _initialize_session(profile, cwd=cwd)
            try:
                print(f"SESSION_ID:{client.resume_session(session_id, cwd)}")
            finally:
                client.close()
            return 0

        if command == "fork":
            if not rest:
                raise RuntimeError("fork requires <session_id> [cwd].")
            session_id = rest[0]
            cwd = rest[1] if len(rest) > 1 else str(Path.home())
            client, _ = _initialize_session(profile, cwd=cwd)
            try:
                print(f"SESSION_ID:{client.fork_session(session_id, cwd)}")
            finally:
                client.close()
            return 0

        if command == "prompt":
            if len(rest) < 2:
                raise RuntimeError("prompt requires <session_id> <text> [timeout_seconds].")
            session_id = rest[0]
            text = rest[1]
            timeout = float(rest[2]) if len(rest) > 2 else 180.0
            client, _ = _initialize_session(profile)
            try:
                result = client.prompt(session_id, text, timeout=timeout)
                response = str(result.get("text") or "").strip()
                print(response or str(result.get("stop_reason") or "(empty response)"))
            finally:
                client.close()
            return 0

        if command == "list":
            cwd = rest[0] if rest else None
            client, _ = _initialize_session(profile, cwd=cwd)
            try:
                print(json.dumps(client.list_sessions(cwd), indent=2))
            finally:
                client.close()
            return 0

        if command == "cancel":
            if not rest:
                raise RuntimeError("cancel requires <session_id>.")
            client, _ = _initialize_session(profile)
            try:
                client.cancel(rest[0])
                print(f"CANCELLED:{rest[0]}")
            finally:
                client.close()
            return 0

        raise RuntimeError(f"Unknown command: {command}")
    except Exception as exc:
        print(f"ERROR:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
