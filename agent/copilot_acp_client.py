"""OpenAI-compatible shim that forwards Hermes requests to ACP servers.

This adapter lets Hermes treat ACP-compatible servers (GitHub Copilot, etc.) as
a chat-style backend.  Two transport modes are supported:

* **stdio** (default): Hermes spawns the ACP command as a subprocess and
  communicates via stdin/stdout newline-delimited JSON-RPC.
* **streamable-http**: Hermes connects to a running ACP HTTP server (e.g.
  started with ``--acp --acp-transport streamable-http`` or via a daemon)
  and communicates via SSE over HTTP.

The transport is auto-detected from ``base_url``:
  - ``acp://…`` or absent  → stdio subprocess
  - ``acp+http://…``, ``acp+https://…``, or ``acp+tcp://…`` → streamable-http
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import httpx  # Optional — only needed for streamable-http transport
except ImportError:
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
from agent.file_safety import get_read_block_error, is_write_denied
from agent.redact import redact_sensitive_text

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _is_http_base_url(base_url: str) -> bool:
    """Return True when base_url points to a running HTTP ACP server."""
    return base_url.startswith(("acp+http://", "acp+https://", "acp+tcp://"))


def _normalize_http_base_url(base_url: str) -> str:
    """Convert ``acp+http://`` / ``acp+https://`` / ``acp+tcp://`` to real HTTP URLs."""
    if base_url.startswith("acp+https://"):
        return ("https://" + base_url[len("acp+https://"):]).rstrip("/")
    if base_url.startswith("acp+http://"):
        return ("http://" + base_url[len("acp+http://"):]).rstrip("/")
    if base_url.startswith("acp+tcp://"):
        return ("http://" + base_url[len("acp+tcp://"):]).rstrip("/")
    return base_url.rstrip("/")


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""

    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            return profile_home
    except Exception:
        pass

    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "cancelled",
            }
        },
    }


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

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


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[SimpleNamespace] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type="function",
                function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned



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


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


class _HttpACPSession:
    """HTTP client for ACP streamable-http transport.

    Protocol (as implemented by ACP-compatible servers):

      1. ``POST <base>/api/v1/acp/connect``  → ``{connectionId, sessionToken}``
      2. ``POST <base>/api/v1/acp``
             ``acp-connection-id: <connectionId>``
             ``Accept: application/json, text/event-stream``
         Body: JSON-RPC 2.0 message.
         Response: chunked SSE stream — the server keeps the connection open;
         we stop reading once we receive the response event that matches our
         request id or an ``agent_turn_complete`` notification.
    """

    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for ACP streamable-http transport. "
                "Install it with: pip install httpx"
            )
        self._base = _normalize_http_base_url(base_url)
        self._timeout = timeout
        self._next_id = 0

    def _next_msg_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _connect(self) -> str:
        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{self._base}/api/v1/acp/connect", json={})
            resp.raise_for_status()
            data = resp.json()
        conn_id = data.get("connectionId")
        if not conn_id:
            raise RuntimeError(f"ACP HTTP connect did not return connectionId: {data}")
        logger.debug("ACP HTTP connected: connectionId=%s", conn_id)
        return conn_id

    def _rpc_sse(self, conn_id: str, method: str, params: dict[str, Any],
                 msg_id: int, *, collect_notifications: bool = False,
                 timeout: float = 30.0) -> list[dict[str, Any]]:
        """Send one JSON-RPC request and read SSE events until response arrives."""
        payload = {"jsonrpc": "2.0", "method": method, "id": msg_id, "params": params}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "acp-connection-id": conn_id,
        }
        events: list[dict[str, Any]] = []
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{self._base}/api/v1/acp",
                               json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    line = line.rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        msg = json.loads(data_str)
                    except Exception:
                        logger.debug("ACP HTTP: ignoring non-JSON SSE data: %s", data_str[:100])
                        continue
                    events.append(msg)
                    if msg.get("id") == msg_id and ("result" in msg or "error" in msg):
                        if not collect_notifications:
                            break
        # Check for JSON-RPC error in the response
        for ev in events:
            if ev.get("id") == msg_id and "error" in ev:
                err = ev["error"]
                logger.warning("ACP HTTP RPC error on %s: %s", method, err)
                raise RuntimeError(f"ACP HTTP {method} failed: {err.get('message', err)}")
        return events

    def prompt(self, prompt_text: str, model: str | None = None,
               timeout: float | None = None) -> tuple[str, str]:
        """Send a prompt via ACP HTTP and collect streamed chunks.

        Returns ``(text, reasoning)``.
        """
        t = timeout or self._timeout
        conn_id = self._connect()

        # 1. initialize
        init_id = self._next_msg_id()
        self._rpc_sse(conn_id, "initialize", {
            "protocolVersion": 1,
            "capabilities": {},
            "clientInfo": {"name": "hermes", "version": "1.0"},
        }, init_id, timeout=10)

        # 2. session/new — ``mcpServers`` is required (empty = no extra tools)
        new_id = self._next_msg_id()
        session_events = self._rpc_sse(conn_id, "session/new", {
            "mcpServers": [],
            "cwd": os.getcwd(),
        }, new_id, collect_notifications=True, timeout=15)

        # sessionId arrives in a session/update notification
        session_id: str | None = None
        for ev in session_events:
            params_ev = ev.get("params") or {}
            sid = params_ev.get("sessionId")
            if sid:
                session_id = sid
                break
            result_ev = ev.get("result") or {}
            sid = result_ev.get("sessionId") or result_ev.get("session_id")
            if sid:
                session_id = sid
                break

        if not session_id:
            raise RuntimeError("ACP HTTP: failed to obtain sessionId from session/new")
        logger.debug("ACP HTTP session created: %s", session_id)

        # 3. session/prompt — prompt must be an array of content blocks
        prompt_id = self._next_msg_id()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "acp-connection-id": conn_id,
        }
        payload = {"jsonrpc": "2.0", "method": "session/prompt", "id": prompt_id, "params": {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": prompt_text}],
        }}

        text_parts: list[str] = []
        reasoning_parts: list[str] = []

        with httpx.Client(timeout=t) as client:
            with client.stream("POST", f"{self._base}/api/v1/acp",
                               json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    line = line.rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        msg = json.loads(data_str)
                    except Exception:
                        logger.debug("ACP HTTP: ignoring non-JSON SSE data: %s", data_str[:100])
                        continue

                    # Notification: streamed chunks
                    if msg.get("method") == "session/update":
                        p = msg.get("params") or {}
                        update = p.get("update") or {}
                        kind = str(update.get("sessionUpdate") or "").strip()
                        content = update.get("content") or {}
                        chunk = str(content.get("text") or "") if isinstance(content, dict) else ""
                        if kind == "agent_message_chunk" and chunk:
                            text_parts.append(chunk)
                        elif kind == "agent_thought_chunk" and chunk:
                            reasoning_parts.append(chunk)
                        elif kind in ("agent_turn_complete", "session_update_complete"):
                            break
                        continue

                    # Response to session/prompt
                    if msg.get("id") == prompt_id:
                        if "error" in msg:
                            raise RuntimeError(f"ACP prompt error: {msg['error']}")
                        result = msg.get("result") or {}
                        direct = result.get("message") or result.get("text") or ""
                        if direct and not text_parts:
                            text_parts.append(str(direct))
                        break

        return "".join(text_parts), "".join(reasoning_parts)


class CopilotACPClient:
    """Minimal OpenAI-client-compatible facade for ACP servers.

    Supports two transport modes, auto-detected from ``base_url``:

    * **Subprocess (stdio)**: ``base_url`` is ``acp://copilot`` or similar
      marker — a child process is spawned for each request.
    * **HTTP (streamable-http)**: ``base_url`` starts with ``acp+http://``,
      ``acp+https://``, or ``acp+tcp://`` — connects to a running ACP server.
    """

    def __init__(
        self,
        *,
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
        self.api_key = api_key or "copilot-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text = self._run_prompt(
            prompt_text,
            timeout_seconds=_effective_timeout,
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "copilot-acp",
        )

    def _run_prompt(self, prompt_text: str, *, model: str | None = None, timeout_seconds: float) -> tuple[str, str]:
        # HTTP mode: base_url points to a running ACP streamable-http server
        if _is_http_base_url(self.base_url):
            session = _HttpACPSession(self.base_url, timeout=timeout_seconds)
            return session.prompt(prompt_text, model=model, timeout=timeout_seconds)

        # Subprocess (stdio) mode
        try:
            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Copilot ACP command '{self._acp_command}'. "
                "Install GitHub Copilot CLI or set HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Copilot ACP process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
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

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None) -> Any:
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
                        f"Copilot ACP {method} failed: {err.get('message') or err}"
                    )
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                raise RuntimeError(f"Copilot ACP process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for Copilot ACP response to {method}.")

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
                raise RuntimeError("Copilot ACP did not return a sessionId.")

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
            response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                content = path.read_text() if path.exists() else ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                if is_write_denied(str(path)):
                    raise PermissionError(
                        f"Write denied: '{path}' is a protected system/credential file."
                    )
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
