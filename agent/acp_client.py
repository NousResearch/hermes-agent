"""OpenAI-compatible shim that forwards Hermes requests to any ACP agent.

Generalised from the Copilot-specific ``copilot_acp_client.py``.  Each request
starts a short-lived ACP session, sends the formatted conversation as a single
prompt, collects text chunks, and converts the result back into the minimal
shape Hermes expects from an OpenAI client.

Usage:
    client = ACPClient(agent_name="claude")
    resp = client.chat.completions.create(messages=[...], model="claude-acp")
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from agent.acp_agent_registry import resolve_agent_command, resolve_agent_env, split_agent_command

logger = logging.getLogger(__name__)

ACP_MARKER_PREFIX = "acp://"
_DEFAULT_TIMEOUT_SECONDS = 900.0


# ---------------------------------------------------------------------------
# Helpers (carried over from copilot_acp_client.py)
# ---------------------------------------------------------------------------

def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    agent_name: str = "ACP agent",
) -> str:
    sections: list[str] = [
        f"You are being used as the active ACP agent backend for Hermes (agent: {agent_name}).",
        "Use your own ACP capabilities and respond directly in natural language.",
        "Do not emit OpenAI tool-call JSON.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


def extract_agent_from_url(base_url: str) -> Optional[str]:
    """Extract agent name from an ``acp://<agent>`` URL, or return ``None``."""
    if not base_url or not base_url.startswith(ACP_MARKER_PREFIX):
        return None
    return base_url[len(ACP_MARKER_PREFIX):].strip().lower() or None


# ---------------------------------------------------------------------------
# OpenAI-compatible namespace shims
# ---------------------------------------------------------------------------

class _ACPChatCompletions:
    def __init__(self, client: "ACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "ACPClient"):
        self.completions = _ACPChatCompletions(client)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class ACPClient:
    """Minimal OpenAI-client-compatible facade for any ACP agent."""

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        # Resolve agent name from explicit param or base_url
        self._agent_name = agent_name or extract_agent_from_url(base_url or "") or "unknown"

        # Resolve the ACP command to spawn
        if acp_command or command:
            self._acp_argv = shlex.split(acp_command or command or "")
            if acp_args or args:
                self._acp_argv.extend(acp_args or args or [])
        else:
            resolved = resolve_agent_command(self._agent_name)
            if resolved is None:
                from agent.acp_agent_registry import list_agents
                raise ValueError(
                    f"Unknown ACP agent '{self._agent_name}'. "
                    f"Set HERMES_ACP_{self._agent_name.upper().replace('-', '_')}_COMMAND or "
                    f"choose from: {', '.join(list_agents())}"
                )
            self._acp_argv = split_agent_command(resolved)

        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())

        # Resolve auth env vars to inject into the subprocess
        self._auth_env = resolve_agent_env(self._agent_name)

        self.api_key = api_key or f"{self._agent_name}-acp"
        self.base_url = base_url or f"acp://{self._agent_name}"
        self._default_headers = dict(default_headers or {})
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

    @property
    def agent_name(self) -> str:
        return self._agent_name

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [], model=model, agent_name=self._agent_name,
        )
        response_text, reasoning_text = self._run_prompt(
            prompt_text,
            timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS),
        )

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=[],
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or f"{self._agent_name}-acp",
        )

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        # Build subprocess env: inherit parent env + inject auth vars
        spawn_env: dict[str, str] | None = None
        if self._auth_env:
            spawn_env = {**os.environ, **self._auth_env}

        try:
            proc = subprocess.Popen(
                self._acp_argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._acp_cwd,
                env=spawn_env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start ACP agent '{self._agent_name}' "
                f"(command: {' '.join(self._acp_argv)}). "
                f"Ensure the agent is installed or set HERMES_ACP_{self._agent_name.upper().replace('-', '_')}_COMMAND."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError(f"ACP agent '{self._agent_name}' did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _stdout_reader() -> None:
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(
            method: str,
            params: dict[str, Any],
            *,
            text_parts: list[str] | None = None,
            reasoning_parts: list[str] | None = None,
        ) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(
                        f"ACP agent '{self._agent_name}' {method} failed: {err.get('message') or err}"
                    )
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                raise RuntimeError(f"ACP agent '{self._agent_name}' process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for ACP agent '{self._agent_name}' response to {method}.")

        try:
            _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": True,
                            "writeTextFile": True,
                        }
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            )
            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError(f"ACP agent '{self._agent_name}' did not return a sessionId.")

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self.close()

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "outcome": {
                        "outcome": "allow_once",
                    }
                },
            }
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                file_content = path.read_text() if path.exists() else ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = file_content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    file_content = "".join(lines[start:end])
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {"content": file_content},
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""))
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True
