"""OpenAI-compatible shim that forwards Hermes requests to `cursor-agent acp`.

Cursor's HTTP OpenAI shim (`cur-*` on :8317/:8321) correctly rejects tool
schemas. This adapter is the execution path: it talks JSON-RPC ACP over
stdio, injects Hermes tool schemas into the prompt, and rebuilds OpenAI
`tool_calls` from `<tool_call>{...}</tool_call>` blocks.

Transport (verified 2026-08-16): `cursor-agent acp` — initialize →
authenticate(cursor_login) → session/new → session/set_model →
session/prompt. Do not fall back to `cursor-agent -p`; ACP works.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, get_write_denied_error, is_write_approval_required
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://cursor"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_DEFAULT_COMMAND = "cursor-agent"
_DEFAULT_ARGS = ["acp"]

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)

_CLI_SUFFIXES = (
    "-high-fast",
    "-xhigh-fast",
    "-low-fast",
    "-medium-fast",
    "-xhigh",
    "-high",
    "-medium",
    "-low",
    "-fast",
)


class CursorACPToolCallParseError(ValueError):
    """A `<tool_call>` block was present but could not be recovered."""


def _resolve_command() -> str:
    return _DEFAULT_COMMAND


def _resolve_args() -> list[str]:
    return list(_DEFAULT_ARGS)


def _resolve_home_dir() -> str:
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
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = hermes_subprocess_env(inherit_credentials=True)
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env

    apply_subprocess_home_env(env)
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {"outcome": {"outcome": "cancelled"}},
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
        tool_calls = message.get("tool_calls")
        if not rendered and isinstance(tool_calls, list) and tool_calls:
            rendered = json.dumps(tool_calls, ensure_ascii=False)
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


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


def _extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    open_count = text.count("<tool_call>")
    close_count = text.count("</tool_call>")
    if open_count != close_count:
        raise CursorACPToolCallParseError(
            "truncated <tool_call> block: opening and closing tags do not match"
        )

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _parse_tool_call(raw_json: str) -> ChatCompletionMessageToolCall:
        try:
            obj = json.loads(raw_json)
        except Exception as exc:
            raise CursorACPToolCallParseError(
                f"malformed <tool_call> JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise CursorACPToolCallParseError("malformed <tool_call> payload: expected object")
        fn = obj.get("function")
        if not isinstance(fn, dict):
            raise CursorACPToolCallParseError("malformed <tool_call> payload: missing function")
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            raise CursorACPToolCallParseError("malformed <tool_call> payload: missing function.name")
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted) + 1}"
        return _build_openai_tool_call(
            call_id=call_id,
            name=fn_name.strip(),
            arguments=fn_args,
        )

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        extracted.append(_parse_tool_call(match.group(1)))
        consumed_spans.append((match.start(), match.end()))

    if open_count and len(extracted) != open_count:
        raise CursorACPToolCallParseError(
            "malformed <tool_call> block: could not parse every tagged payload"
        )

    if not extracted:
        for match in _TOOL_CALL_JSON_RE.finditer(text):
            try:
                extracted.append(_parse_tool_call(match.group(0)))
            except CursorACPToolCallParseError:
                continue
            consumed_spans.append((match.start(), match.end()))

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


def resolve_cursor_acp_model_id(
    requested: str | None,
    available: list[dict[str, Any]] | None,
) -> str | None:
    """Map a Hermes/CLI model id onto a Cursor ACP `modelId`.

    Cursor ACP advertises parameterized ids such as
    ``gemini-3.7-flash[effort=high]``. Hermes seats ask for CLI names such as
    ``gemini-3.7-flash-high`` / ``gpt-5.6-sol-high``. Exact `modelId` and
    `name` win; otherwise a suffix-stripped stem is matched.
    """
    req = (requested or "").strip()
    if not req:
        return None
    rows: list[tuple[str, str]] = []
    for item in available or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("modelId") or "").strip()
        name = str(item.get("name") or "").strip()
        if model_id:
            rows.append((model_id, name))
            if model_id == req or name == req:
                return model_id

    stem = req
    for suffix in _CLI_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem:
        for model_id, name in rows:
            if name == stem or model_id == stem or model_id.startswith(stem + "["):
                return model_id
    return None


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
    def __init__(self, client: "CursorACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CursorACPClient"):
        self.completions = _ACPChatCompletions(client)


class CursorACPClient:
    """Minimal OpenAI-client-compatible facade for Cursor ACP."""

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
        self.api_key = api_key or "cursor-acp"
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
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        if timeout is None:
            effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text = self._run_prompt(
            prompt_text,
            timeout_seconds=effective_timeout,
            model=model,
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
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "cursor-acp",
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        timeout_seconds: float,
        model: str | None = None,
    ) -> tuple[str, str]:
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Cursor ACP command '{self._acp_command}'. "
                "Install the Cursor CLI and run `cursor-agent login`, then retry."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Cursor ACP process did not expose stdin/stdout pipes.")

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

        threading.Thread(target=_stdout_reader, daemon=True).start()
        threading.Thread(target=_stderr_reader, daemon=True).start()

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

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
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
                        f"Cursor ACP {method} failed: {err.get('message') or err}"
                    )
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                lower = stderr_text.lower()
                if "login" in lower or "unauthor" in lower or "auth" in lower:
                    raise RuntimeError(
                        "Cursor ACP process exited before completing the handshake. "
                        "Run `cursor-agent login` and retry.\n"
                        f"Original error:\n{stderr_text}"
                    )
                raise RuntimeError(f"Cursor ACP process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for Cursor ACP response to {method}.")

        try:
            init_result = _request(
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
            ) or {}
            auth_methods = init_result.get("authMethods") if isinstance(init_result, dict) else None
            method_id = "cursor_login"
            if isinstance(auth_methods, list):
                ids = [
                    str(item.get("id") or "").strip()
                    for item in auth_methods
                    if isinstance(item, dict)
                ]
                if "cursor_login" in ids:
                    method_id = "cursor_login"
                elif ids:
                    method_id = ids[0]
            try:
                _request("authenticate", {"methodId": method_id})
            except RuntimeError as exc:
                raise RuntimeError(
                    "Cursor ACP authentication failed. "
                    "Run `cursor-agent login` and retry. "
                    f"Original error: {exc}"
                ) from exc

            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("Cursor ACP did not return a sessionId.")

            available = []
            models = session.get("models") if isinstance(session, dict) else None
            if isinstance(models, dict):
                raw_available = models.get("availableModels")
                if isinstance(raw_available, list):
                    available = [item for item in raw_available if isinstance(item, dict)]
            resolved = resolve_cursor_acp_model_id(model, available)
            if model and available and resolved is None:
                raise RuntimeError(
                    f"Cursor ACP does not advertise model {model!r}. "
                    "Use a Cursor CLI id such as gpt-5.6-sol-high or gemini-3.7-flash-high."
                )
            if resolved:
                _request(
                    "session/set_model",
                    {"sessionId": session_id, "modelId": resolved},
                )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": prompt_text}],
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
                try:
                    content = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {"content": content},
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                denied = get_write_denied_error(str(path))
                if denied:
                    raise PermissionError(denied)
                if is_write_approval_required(str(path)):
                    raise PermissionError(
                        f"Write denied: '{path}' requires interactive approval "
                        "and cannot be written through the ACP file bridge."
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""), encoding="utf-8")
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
