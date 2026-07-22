"""OpenAI-compatible shim that forwards Hermes requests to `copilot --acp`.

This adapter lets Hermes treat the GitHub Copilot ACP server as a chat-style
backend. A long-lived ACP subprocess + session is reused across completions;
prompts stream `session/update` chunks back as OpenAI-style deltas when
requested.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, get_write_denied_error
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*", re.IGNORECASE)
_TOOL_CALL_CLOSE = "</tool_call>"

# Stderr fingerprint of the deprecated `gh copilot` CLI extension
# (https://github.blog/changelog/2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension).
_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = (
    "has been deprecated",
    "no commands will be executed",
)

_EXEC_PERMISSION_HINTS = (
    "exec",
    "terminal",
    "shell",
    "bash",
    "command",
    "run_command",
    "sudo",
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""

    lower = stderr_text.lower()
    if not any(req in lower for req in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lower for marker in _DEPRECATION_MARKERS)


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


def _permission_mode() -> str:
    mode = os.getenv("HERMES_COPILOT_ACP_PERMISSION_MODE", "cwd_safe").strip().lower()
    if mode in {"deny", "cwd_safe"}:
        return mode
    return "cwd_safe"


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    # Copilot ACP is a model-driving CLI executor: it legitimately needs LLM
    # provider credentials. Route through the central helper so Tier-1 secrets
    # (gateway bot tokens, GitHub auth, infra) are still stripped (#29157).
    env = hermes_subprocess_env(inherit_credentials=True)
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env, get_real_home

    apply_subprocess_home_env(env)
    # Copilot CLI auth lives under the real user HOME (~/.config/github-copilot).
    # Do not remap HOME to the Hermes profile home even in containers.
    real_home = get_real_home(env)
    if real_home:
        env["HERMES_REAL_HOME"] = real_home
        env["HOME"] = real_home
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


def _permission_allowed(message_id: Any, option_id: str = "allow_once") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {
            "outcome": {
                "outcome": "selected",
                "option_id": option_id,
            }
        },
    }


def _is_sensitive_path(path: Path) -> bool:
    try:
        from acp_adapter.edit_approval import _is_sensitive_auto_approve_path

        return _is_sensitive_auto_approve_path(str(path))
    except Exception:
        name = path.name.lower()
        if name in {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}:
            return True
        parts = {p.lower() for p in path.parts}
        return ".git" in parts or ".ssh" in parts


def _permission_looks_like_exec(params: dict[str, Any]) -> bool:
    blob = json.dumps(params, ensure_ascii=False).lower()
    return any(hint in blob for hint in _EXEC_PERMISSION_HINTS)


def _extract_permission_paths(params: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                if key_l in {"path", "filepath", "file_path", "uri", "target"} and isinstance(value, str):
                    if value.startswith("file://"):
                        value = value[len("file://"):]
                    paths.append(value)
                else:
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(params)
    return paths


def _decide_permission(params: dict[str, Any], cwd: str) -> bool:
    """Return True to auto-approve cwd-safe fs permissions; False to deny."""
    if _permission_mode() == "deny":
        return False
    if _permission_looks_like_exec(params):
        return False

    paths = _extract_permission_paths(params)
    if not paths:
        # No path + not exec → deny unknown permission kinds (fail closed).
        return False

    for path_text in paths:
        try:
            path = _ensure_path_within_cwd(path_text, cwd)
        except PermissionError:
            return False
        if _is_sensitive_path(path):
            return False
        if get_read_block_error(str(path)) or get_write_denied_error(str(path)):
            return False
    return True


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a Hermes tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
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
                "Available Hermes tools (OpenAI function schema). "
                "When using a Hermes tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string. "
                "Do not use Copilot shell/terminal for Hermes tool goals; cwd-safe ACP file reads are OK for context.\n"
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


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
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


def _extract_balanced_json(text: str, start: int) -> tuple[str | None, int]:
    """Extract one JSON object starting at ``start`` with brace balancing."""
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], idx + 1
    return None, start


def _extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> bool:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return False
        if not isinstance(obj, dict):
            return False
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return False
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return False
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=fn_name.strip(),
                arguments=fn_args,
            )
        )
        return True

    search_from = 0
    while True:
        open_match = _TOOL_CALL_OPEN_RE.search(text, search_from)
        if not open_match:
            break
        json_start = open_match.end()
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1
        raw_json, after_json = _extract_balanced_json(text, json_start)
        if raw_json is None:
            search_from = open_match.end()
            continue
        close_idx = text.lower().find(_TOOL_CALL_CLOSE, after_json)
        end = close_idx + len(_TOOL_CALL_CLOSE) if close_idx >= 0 else after_json
        if _try_add_tool_call(raw_json):
            consumed_spans.append((open_match.start(), end))
        search_from = end

    # Bare-JSON fallback only when no XML blocks were found.
    if not extracted:
        idx = 0
        while idx < len(text):
            brace = text.find("{", idx)
            if brace < 0:
                break
            raw_json, after = _extract_balanced_json(text, brace)
            if raw_json is None:
                idx = brace + 1
                continue
            if '"type"' in raw_json and '"function"' in raw_json and _try_add_tool_call(raw_json):
                consumed_spans.append((brace, after))
            idx = after

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


def _estimate_usage(
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
    response_text: str,
    reasoning_text: str,
) -> SimpleNamespace:
    try:
        from agent.model_metadata import (
            estimate_request_tokens_rough,
            estimate_tokens_rough,
        )

        prompt_tokens = int(
            estimate_request_tokens_rough(messages or [], tools=tools) or 0
        )
        completion_tokens = int(
            estimate_tokens_rough(response_text or "")
            + estimate_tokens_rough(reasoning_text or "")
        )
    except Exception:
        prompt_tokens = max(1, len(json.dumps(messages or [])) // 4)
        completion_tokens = max(0, (len(response_text) + len(reasoning_text)) // 4)
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


class CopilotACPClient:
    """Minimal OpenAI-client-compatible facade for Copilot ACP."""

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
        self._active_process_lock = threading.RLock()
        self._inbox: queue.Queue[dict[str, Any]] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._next_id = 0
        self._session_id: str | None = None
        self._initialized = False
        self._reader_threads: list[threading.Thread] = []

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
            self._session_id = None
            self._initialized = False
            self._inbox = None
            self._next_id = 0
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

    def _effective_timeout(self, timeout: Any) -> float:
        if timeout is None:
            return _DEFAULT_TIMEOUT_SECONDS
        if isinstance(timeout, (int, float)):
            return float(timeout)
        candidates = [
            getattr(timeout, attr, None)
            for attr in ("read", "write", "connect", "pool", "timeout")
        ]
        numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
        return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

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
        _effective_timeout = self._effective_timeout(timeout)

        if stream:
            return self._stream_chat_completion(
                prompt_text=prompt_text,
                model=model,
                messages=messages,
                tools=tools,
                timeout_seconds=_effective_timeout,
            )

        response_text, reasoning_text = self._run_prompt(
            prompt_text,
            timeout_seconds=_effective_timeout,
        )
        return self._build_completion(
            model=model,
            messages=messages,
            tools=tools,
            response_text=response_text,
            reasoning_text=reasoning_text,
        )

    def _build_completion(
        self,
        *,
        model: str | None,
        messages: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        response_text: str,
        reasoning_text: str,
    ) -> SimpleNamespace:
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        usage = _estimate_usage(messages, tools, response_text, reasoning_text)
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

    def _stream_chat_completion(
        self,
        *,
        prompt_text: str,
        model: str | None,
        messages: list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        timeout_seconds: float,
    ) -> Iterator[SimpleNamespace]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        pending_chunks: queue.Queue[SimpleNamespace | None] = queue.Queue()

        def on_message_chunk(chunk: str) -> None:
            text_parts.append(chunk)
            pending_chunks.put(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                role="assistant",
                                content=chunk,
                                tool_calls=None,
                                reasoning_content=None,
                                reasoning=None,
                            ),
                            finish_reason=None,
                        )
                    ],
                    model=model or "copilot-acp",
                    usage=None,
                )
            )

        def on_thought_chunk(chunk: str) -> None:
            reasoning_parts.append(chunk)
            pending_chunks.put(
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                role="assistant",
                                content=None,
                                tool_calls=None,
                                reasoning_content=chunk,
                                reasoning=chunk,
                            ),
                            finish_reason=None,
                        )
                    ],
                    model=model or "copilot-acp",
                    usage=None,
                )
            )

        def runner() -> None:
            try:
                response_text, reasoning_text = self._run_prompt(
                    prompt_text,
                    timeout_seconds=timeout_seconds,
                    on_message_chunk=on_message_chunk,
                    on_thought_chunk=on_thought_chunk,
                )
                # Mocks / non-chunking backends may return full text without
                # invoking live callbacks — emit one content delta then.
                if not text_parts and response_text:
                    on_message_chunk(response_text)
                if not reasoning_parts and reasoning_text:
                    on_thought_chunk(reasoning_text)
            except Exception as exc:
                pending_chunks.put(exc)  # type: ignore[arg-type]
            finally:
                pending_chunks.put(None)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

        while True:
            item = pending_chunks.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        completion = self._build_completion(
            model=model,
            messages=messages,
            tools=tools,
            response_text="".join(text_parts),
            reasoning_text="".join(reasoning_parts),
        )
        # Emit final finish + usage after live deltas (content already streamed).
        finish_delta = SimpleNamespace(
            role=None,
            content=None,
            tool_calls=None,
            reasoning_content=None,
            reasoning=None,
        )
        if completion.choices[0].message.tool_calls:
            finish_delta.tool_calls = []
            for index, tool_call in enumerate(completion.choices[0].message.tool_calls):
                finish_delta.tool_calls.append(
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
            # Suppress already-streamed plain text when tool calls dominate.
            # Live content may have included tool XML; Hermes extracts from final message path.
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=finish_delta,
                    finish_reason=completion.choices[0].finish_reason,
                )
            ],
            model=completion.model,
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            model=completion.model,
            usage=completion.usage,
        )

    def _process_alive(self) -> bool:
        proc = self._active_process
        return proc is not None and proc.poll() is None

    def _ensure_process(self) -> subprocess.Popen[str]:
        with self._active_process_lock:
            if self._process_alive() and self._inbox is not None:
                assert self._active_process is not None
                return self._active_process

            # Stale process/session — reset and spawn.
            if self._active_process is not None:
                try:
                    self._active_process.kill()
                except Exception:
                    pass
            self._active_process = None
            self._session_id = None
            self._initialized = False
            self._next_id = 0
            self._stderr_tail.clear()

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

            inbox: queue.Queue[dict[str, Any]] = queue.Queue()
            self._inbox = inbox
            self._active_process = proc
            self.is_closed = False

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
                    self._stderr_tail.append(line.rstrip("\n"))

            out_thread = threading.Thread(target=_stdout_reader, daemon=True)
            err_thread = threading.Thread(target=_stderr_reader, daemon=True)
            out_thread.start()
            err_thread.start()
            self._reader_threads = [out_thread, err_thread]
            return proc

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
        text_parts: list[str] | None = None,
        reasoning_parts: list[str] | None = None,
        on_message_chunk: Callable[[str], None] | None = None,
        on_thought_chunk: Callable[[str], None] | None = None,
    ) -> Any:
        proc = self._ensure_process()
        inbox = self._inbox
        if proc.stdin is None or inbox is None:
            raise RuntimeError("Copilot ACP process is not ready.")

        with self._active_process_lock:
            self._next_id += 1
            request_id = self._next_id

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
                on_message_chunk=on_message_chunk,
                on_thought_chunk=on_thought_chunk,
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

        stderr_text = "\n".join(self._stderr_tail).strip()
        if proc.poll() is not None and stderr_text:
            self.close()
            if _is_gh_copilot_deprecation_message(stderr_text):
                raise RuntimeError(
                    "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                    "(github.com/github/copilot-cli), but the binary it just "
                    "spawned is the deprecated `gh copilot` extension.\n\n"
                    "Install the new CLI:\n"
                    "  npm install -g @github/copilot\n"
                    "  # then verify with: copilot --help\n\n"
                    "If `copilot` already resolves to the new CLI but you still see this,\n"
                    "point Hermes at it explicitly:\n"
                    "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
                    "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\n"
                    "directly with a Copilot subscription token) via `hermes setup`.\n\n"
                    f"Original error:\n{stderr_text}"
                )
            raise RuntimeError(f"Copilot ACP process exited early: {stderr_text}")
        raise TimeoutError(f"Timed out waiting for Copilot ACP response to {method}.")

    def _ensure_session(self, *, timeout_seconds: float) -> str:
        with self._active_process_lock:
            if self._process_alive() and self._session_id and self._initialized:
                return self._session_id

        if not self._initialized:
            self._rpc(
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
                timeout_seconds=timeout_seconds,
            )
            self._initialized = True

        session = self._rpc(
            "session/new",
            {
                "cwd": self._acp_cwd,
                "mcpServers": [],
            },
            timeout_seconds=timeout_seconds,
        ) or {}
        session_id = str(session.get("sessionId") or "").strip()
        if not session_id:
            raise RuntimeError("Copilot ACP did not return a sessionId.")
        self._session_id = session_id
        return session_id

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        timeout_seconds: float,
        on_message_chunk: Callable[[str], None] | None = None,
        on_thought_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        with self._active_process_lock:
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            try:
                session_id = self._ensure_session(timeout_seconds=timeout_seconds)
                self._rpc(
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
                    timeout_seconds=timeout_seconds,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                    on_message_chunk=on_message_chunk,
                    on_thought_chunk=on_thought_chunk,
                )
                return "".join(text_parts), "".join(reasoning_parts)
            except Exception:
                # Force respawn on next call after hard failures.
                self.close()
                raise

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
        on_message_chunk: Callable[[str], None] | None = None,
        on_thought_chunk: Callable[[str], None] | None = None,
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
            if kind == "agent_message_chunk" and chunk_text:
                if text_parts is not None:
                    text_parts.append(chunk_text)
                if on_message_chunk is not None:
                    on_message_chunk(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text:
                if reasoning_parts is not None:
                    reasoning_parts.append(chunk_text)
                if on_thought_chunk is not None:
                    on_thought_chunk(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if method == "session/request_permission":
            if _decide_permission(params, cwd):
                response = _permission_allowed(message_id)
            else:
                response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text()
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
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                denied = get_write_denied_error(str(path))
                if denied:
                    raise PermissionError(denied)
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
