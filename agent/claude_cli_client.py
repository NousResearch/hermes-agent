"""OpenAI-compatible facade that routes Hermes requests through Claude Code CLI.

This adapter lets Hermes treat ``claude -p`` as a chat-style backend.
It disables Claude Code built-in tools and asks the model to emit OpenAI-style
tool calls inside ``<tool_call>{...}</tool_call>`` blocks when Hermes tools are
needed. The client remembers the Claude session id and resumes it on later
turns within the same Hermes session.
"""

from __future__ import annotations

import json
import os
import re
import select
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any


CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://local"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)
_STREAM_TOOL_START = "<tool_call>"
_STREAM_TOOL_END = "</tool_call>"
_EMPTY_MCP_CONFIG = json.dumps({"mcpServers": {}}, separators=(",", ":"))
_DEFAULT_STRIPPED_RUNTIME_ENV = {
    "CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
}


def _debug_log(message: str) -> None:
    path = os.getenv("HERMES_CLAUDE_CLI_DEBUG_LOG", "").strip()
    if not path:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except Exception:
        pass


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip()
        or os.getenv("CLAUDE_CLI_PATH", "").strip()
        or os.getenv("CLAUDE_CODE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_CLAUDE_CLI_ARGS", "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def _resolve_cwd(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return str(Path(explicit).expanduser().resolve())

    env_cwd = os.getenv("HERMES_CLAUDE_CLI_CWD", "").strip()
    if env_cwd:
        return str(Path(env_cwd).expanduser().resolve())

    if os.getenv("HERMES_CLAUDE_CLI_USE_PROCESS_CWD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return str(Path.cwd().resolve())

    neutral = Path.home() / ".hermes" / "claude-cli-runtime"
    neutral.mkdir(parents=True, exist_ok=True)
    return str(neutral.resolve())


def _normalize_model(model: str | None) -> str | None:
    if not model:
        return None
    normalized = model.strip()
    if normalized.startswith("claude-cli/"):
        normalized = normalized.split("/", 1)[1]
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    return normalized or None


def _resume_enabled() -> bool:
    raw = os.getenv("HERMES_CLAUDE_CLI_RESUME", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _normalize_effort(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized or normalized in {"none", "disabled", "off"}:
        return None
    return {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }.get(normalized)


def _extract_effort(extra_kwargs: dict[str, Any]) -> str | None:
    reasoning = extra_kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            return None
        normalized = _normalize_effort(reasoning.get("effort"))
        if normalized:
            return normalized

    extra_body = extra_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        body_reasoning = extra_body.get("reasoning")
        if isinstance(body_reasoning, dict):
            if body_reasoning.get("enabled") is False:
                return None
            normalized = _normalize_effort(body_reasoning.get("effort"))
            if normalized:
                return normalized

        if extra_body.get("think") is False:
            return None

    return None


def _system_prompt_flag() -> str:
    mode = os.getenv("HERMES_CLAUDE_CLI_SYSTEM_PROMPT_MODE", "").strip().lower()
    if mode in {"append", "append-file"}:
        return "--append-system-prompt-file"
    return "--system-prompt-file"


def _strip_runtime_enabled() -> bool:
    raw = os.getenv("HERMES_CLAUDE_CLI_STRIP_RUNTIME", "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _apply_runtime_env_defaults() -> None:
    if not _strip_runtime_enabled():
        return
    for key, value in _DEFAULT_STRIPPED_RUNTIME_ENV.items():
        os.environ.setdefault(key, value)


@contextmanager
def _system_prompt_file_args(system_prompt: str):
    text = str(system_prompt or "").strip()
    if not text:
        yield []
        return

    fd, path = tempfile.mkstemp(prefix="hermes-claude-system-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        yield [_system_prompt_flag(), path]
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


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
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("input_text") or ""
                parts.append(str(text).strip())
            elif item_type in {"image_url", "input_image"}:
                parts.append("[image omitted]")
            elif item_type in {"tool_result", "tool_use", "function"}:
                parts.append(f"[{item_type} omitted]")
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _render_assistant_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    rendered_calls: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        payload = {
            "id": tool_call.get("call_id") or tool_call.get("id") or "",
            "type": tool_call.get("type") or "function",
            "function": {
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "{}",
            },
        }
        rendered_calls.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(rendered_calls).strip()


def _split_system_prompt(
    messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role == "system":
            rendered = _render_message_content(message.get("content"))
            if rendered:
                system_parts.append(rendered)
            continue
        non_system.append(message)
    return "\n\n".join(part for part in system_parts if part).strip(), non_system


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "If you need a Hermes tool, emit exactly <tool_call>{...}</tool_call>.",
        "Each tool call JSON must match OpenAI function-call shape exactly: {\"id\":...,\"type\":\"function\",\"function\":{\"name\":...,\"arguments\":\"{...}\"}}.",
        "Transcript tool requests and tool results are literal prior Hermes tool traffic.",
    ]

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
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
                "Available Hermes tools (OpenAI function schema):\n"
                + json.dumps(tool_specs, ensure_ascii=False, separators=(",", ":"))
            )

    if tool_choice not in (None, "auto"):
        sections.append(
            f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False, separators=(',', ':'))}"
        )

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "assistant":
            rendered = _render_message_content(message.get("content"))
            rendered_tool_calls = _render_assistant_tool_calls(message.get("tool_calls"))
            if rendered:
                transcript.append(f"Assistant:\n{rendered}")
            if rendered_tool_calls:
                transcript.append(
                    "Assistant tool request(s):\n"
                    f"{rendered_tool_calls}"
                )
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            rendered = _render_message_content(message.get("content"))
            if rendered:
                label = f"Tool result ({tool_call_id})" if tool_call_id else "Tool result"
                transcript.append(f"{label}:\n{rendered}")
            continue

        label = {
            "user": "User",
        }.get(role, role.title())
        rendered = _render_message_content(message.get("content"))
        if rendered:
            transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(part.strip() for part in sections if part and part.strip())


def _build_tool_guidance(
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "If you need a Hermes tool, emit exactly <tool_call>{...}</tool_call>.",
        "Each tool call JSON must match OpenAI function-call shape exactly: {\"id\":...,\"type\":\"function\",\"function\":{\"name\":...,\"arguments\":\"{...}\"}}.",
        "Tool result messages arrive as plain user messages prefixed with 'Tool result (...)'.",
    ]

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
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
                "Available Hermes tools (OpenAI function schema):\n"
                + json.dumps(tool_specs, ensure_ascii=False, separators=(",", ":"))
            )

    if tool_choice not in (None, "auto"):
        sections.append(
            f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False, separators=(',', ':'))}"
        )

    return "\n\n".join(part.strip() for part in sections if part and part.strip())


def _combine_system_prompt(system_prompt: str, tool_guidance: str) -> str:
    parts = [str(system_prompt or "").strip(), str(tool_guidance or "").strip()]
    return "\n\n".join(part for part in parts if part)


def _make_user_event(text: str) -> str | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    return json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": cleaned,
            },
        },
        ensure_ascii=False,
    )


def _build_resume_delta_payload(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return None

    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if isinstance(message, dict) and str(message.get("role") or "").strip().lower() == "assistant":
            last_assistant_idx = idx
            break

    if last_assistant_idx is None:
        return None

    delta_messages = messages[last_assistant_idx + 1 :]
    if not delta_messages:
        return None

    parts: list[str] = []
    for message in delta_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role == "user":
            rendered = _render_message_content(message.get("content"))
            if rendered:
                parts.append(rendered)
            continue
        if role == "tool":
            rendered = _render_message_content(message.get("content"))
            if rendered:
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                label = f"Tool result ({tool_call_id})" if tool_call_id else "Tool result"
                parts.append(f"{label}:\n{rendered}")
            continue
        return None

    return _make_user_event("\n\n".join(part for part in parts if part))


def _build_initial_structured_payload(messages: list[dict[str, Any]]) -> str | None:
    if len(messages) != 1:
        return None
    message = messages[0]
    if not isinstance(message, dict):
        return None
    if str(message.get("role") or "").strip().lower() != "user":
        return None
    return _make_user_event(_render_message_content(message.get("content")))


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    _debug_log(
        "extract:start "
        f"text_len={len(text) if isinstance(text, str) else -1} "
        f"has_tag={'<tool_call>' in text if isinstance(text, str) else False}"
    )
    if not isinstance(text, str) or not text.strip():
        _debug_log("extract:empty")
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
            call_id = f"claude_cli_call_{len(extracted)+1}"

        extracted.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type="function",
                function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
            )
        )

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add_tool_call(match.group(1))
        consumed_spans.append((match.start(), match.end()))

    if not extracted:
        for match in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add_tool_call(match.group(0))
            consumed_spans.append((match.start(), match.end()))

    if not consumed_spans:
        cleaned = text.strip()
        if cleaned in {"</s>", "<|endoftext|>", "<|eot_id|>"}:
            cleaned = ""
        _debug_log(f"extract:none cleaned_len={len(cleaned)}")
        return extracted, cleaned

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

    cleaned = "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    if cleaned in {"</s>", "<|endoftext|>", "<|eot_id|>"}:
        cleaned = ""
    _debug_log(f"extract:done tool_calls={len(extracted)} cleaned_len={len(cleaned)}")
    return extracted, cleaned


def _split_partial_marker_tail(text: str, marker: str) -> tuple[str, str]:
    if not text or not marker:
        return text, ""
    max_keep = min(len(text), len(marker) - 1)
    for size in range(max_keep, 0, -1):
        if marker.startswith(text[-size:]):
            return text[:-size], text[-size:]
    return text, ""


def _extract_text_from_content_blocks(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _coerce_reasoning_delta(block_type: str | None, delta: dict[str, Any]) -> str:
    kind = str(delta.get("type") or "").strip().lower()
    if block_type in {"thinking", "redacted_thinking"} or "thinking" in kind:
        for key in ("thinking", "text"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class _ClaudeCLIChatCompletions:
    def __init__(self, client: "ClaudeCLIClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCLIChatNamespace:
    def __init__(self, client: "ClaudeCLIClient"):
        self.completions = _ClaudeCLIChatCompletions(client)


class _ClaudeCLIStreamChunk(SimpleNamespace):
    """Mimics an OpenAI ChatCompletionChunk with .choices[0].delta."""


def _make_stream_chunk(
    *,
    model: str,
    content: str = "",
    reasoning: str = "",
    tool_call_delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> _ClaudeCLIStreamChunk:
    delta_kwargs: dict[str, Any] = {
        "content": None,
        "tool_calls": None,
        "reasoning": None,
        "reasoning_content": None,
    }
    if content or tool_call_delta is not None or reasoning:
        delta_kwargs["role"] = "assistant"
    if content:
        delta_kwargs["content"] = content
    if reasoning:
        delta_kwargs["reasoning"] = reasoning
        delta_kwargs["reasoning_content"] = reasoning
    if tool_call_delta is not None:
        delta_kwargs["tool_calls"] = [
            SimpleNamespace(
                index=tool_call_delta.get("index", 0),
                id=tool_call_delta.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=SimpleNamespace(
                    name=tool_call_delta.get("name") or "",
                    arguments=tool_call_delta.get("arguments") or "",
                ),
            )
        ]
    delta = SimpleNamespace(**delta_kwargs)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return _ClaudeCLIStreamChunk(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


class ClaudeCLIClient:
    """Minimal OpenAI-client-compatible facade for Claude Code CLI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        claude_command: str | None = None,
        claude_args: list[str] | None = None,
        claude_cwd: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ):
        _apply_runtime_env_defaults()
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = claude_command or command or _resolve_command()
        self._args = list(claude_args or args or _resolve_args())
        self._cwd = _resolve_cwd(claude_cwd)
        self._timeout = (
            float(timeout) if isinstance(timeout, (int, float)) else _DEFAULT_TIMEOUT_SECONDS
        )
        self._last_session_id: str | None = None
        self._last_total_cost_usd: float | None = None
        self._last_stop_reason: str | None = None
        self.chat = _ClaudeCLIChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **extra_kwargs: Any,
    ) -> Any:
        system_prompt, prompt_messages = _split_system_prompt(messages or [])
        tool_guidance = _build_tool_guidance(tools=tools, tool_choice=tool_choice)
        prompt_text = _format_messages_as_prompt(
            prompt_messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        effort = _extract_effort(extra_kwargs)
        structured_stream_input: str | None = None
        structured_stream_system_prompt = system_prompt

        if self._last_session_id and _resume_enabled():
            structured_stream_input = _build_resume_delta_payload(prompt_messages)
            if structured_stream_input:
                structured_stream_system_prompt = _combine_system_prompt(
                    system_prompt,
                    tool_guidance,
                )
        else:
            structured_stream_input = _build_initial_structured_payload(prompt_messages)
            if structured_stream_input:
                structured_stream_system_prompt = _combine_system_prompt(
                    system_prompt,
                    tool_guidance,
                )
        _debug_log(
            "create:stream_prep "
            f"stream={stream} "
            f"resume_enabled={_resume_enabled()} "
            f"has_last_session={bool(self._last_session_id)} "
            f"last_session_id={self._last_session_id or ''} "
            f"prompt_messages={len(prompt_messages)} "
            f"roles={[str((m or {}).get('role') or '') for m in prompt_messages if isinstance(m, dict)]} "
            f"structured_input={bool(structured_stream_input)}"
        )

        if timeout is None:
            effective_timeout = self._timeout
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else self._timeout

        if stream:
            return self._stream_completion_live(
                model=model or "claude-cli",
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                structured_input=structured_stream_input,
                structured_system_prompt=structured_stream_system_prompt,
                effort=effort,
                timeout_seconds=effective_timeout,
            )

        result = self._run_prompt(
            prompt_text,
            system_prompt=system_prompt,
            model=model,
            effort=effort,
            timeout_seconds=effective_timeout,
        )
        _debug_log(
            "create:prompt_done "
            f"prompt_len={len(prompt_text)} "
            f"system_prompt_len={len(system_prompt)} "
            f"effort={effort or ''} "
            f"result_keys={sorted(result.keys())}"
        )
        response_text = str(result.get("result") or "").strip()
        _debug_log(f"create:result_text result_len={len(response_text)}")
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        _debug_log(
            "create:extract_done "
            f"tool_calls={len(tool_calls)} cleaned_len={len(cleaned_text)}"
        )
        usage_payload = result.get("usage") or {}

        prompt_tokens = int(
            usage_payload.get("input_tokens")
            or usage_payload.get("cache_creation_input_tokens")
            or 0
        )
        completion_tokens = int(usage_payload.get("output_tokens") or 0)
        cached_tokens = int(usage_payload.get("cache_read_input_tokens") or 0)

        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        )
        finish_reason = "tool_calls" if tool_calls else "stop"

        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "claude-cli",
            claude_session_id=self._last_session_id,
            claude_total_cost_usd=self._last_total_cost_usd,
            claude_stop_reason=self._last_stop_reason,
        )

    def _stream_completion(
        self,
        *,
        model: str,
        content: str,
        tool_calls: list[SimpleNamespace],
        finish_reason: str,
        usage: Any,
    ):
        def _generator():
            if content:
                yield _make_stream_chunk(model=model, content=content)

            for index, tool_call in enumerate(tool_calls):
                yield _make_stream_chunk(
                    model=model,
                    tool_call_delta={
                        "index": index,
                        "id": getattr(tool_call, "id", None),
                        "name": getattr(getattr(tool_call, "function", None), "name", ""),
                        "arguments": getattr(getattr(tool_call, "function", None), "arguments", ""),
                    },
                )

            yield _make_stream_chunk(
                model=model,
                finish_reason=finish_reason,
                usage=usage,
            )

        return _generator()

    def _stream_completion_live(
        self,
        *,
        model: str,
        prompt_text: str,
        system_prompt: str,
        structured_input: str | None,
        structured_system_prompt: str,
        effort: str | None,
        timeout_seconds: float,
    ):
        def _generator():
            normalized_model = _normalize_model(model)
            use_structured_input = bool(structured_input and structured_input.strip())
            effective_system_prompt = (
                structured_system_prompt if use_structured_input else system_prompt
            )
            stdin_payload = structured_input if use_structured_input else prompt_text

            with _system_prompt_file_args(effective_system_prompt) as system_args:
                command = [
                    self._command,
                    *self._args,
                    "-p",
                    "--verbose",
                    "--input-format",
                    "stream-json" if use_structured_input else "text",
                    "--output-format",
                    "stream-json",
                    "--include-partial-messages",
                    "--tools",
                    "",
                    "--disable-slash-commands",
                    "--strict-mcp-config",
                    "--mcp-config",
                    _EMPTY_MCP_CONFIG,
                    "--setting-sources",
                    "user",
                    *system_args,
                ]
                if normalized_model:
                    command.extend(["--model", normalized_model])
                if effort:
                    command.extend(["--effort", effort])
                if self._last_session_id and _resume_enabled():
                    command.extend(["--resume", self._last_session_id])

                resolved = shutil.which(command[0]) if command and command[0] else None
                if not resolved:
                    raise RuntimeError(
                        f"Could not find Claude CLI command '{command[0]}'. Install Claude Code or set "
                        "HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CLI_PATH."
                    )
                command[0] = resolved
                _debug_log(
                    "stream_prompt:start "
                    f"model={normalized_model or ''} "
                    f"structured={use_structured_input} "
                    f"effort={effort or ''} "
                    f"timeout={timeout_seconds:.1f} "
                    f"cwd={self._cwd} "
                    f"argv_len={sum(len(part) for part in command)} "
                    f"stdin_len={len(stdin_payload)} "
                    f"system_prompt_len={len(effective_system_prompt)}"
                )
                try:
                    proc = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=self._cwd,
                        bufsize=1,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to start Claude CLI: {exc}") from exc

                assert proc.stdin is not None
                assert proc.stdout is not None
                assert proc.stderr is not None

                try:
                    proc.stdin.write(stdin_payload)
                    proc.stdin.close()
                except Exception:
                    proc.kill()
                    proc.wait()
                    raise

                block_types: dict[int, str] = {}
                stderr_lines: list[str] = []
                raw_text_parts: list[str] = []
                fallback_assistant_text = ""
                pending_text = ""
                tool_buffer = ""
                inside_tool_block = False
                emitted_text = ""
                result_payload: dict[str, Any] | None = None
                usage = None
                finish_reason = "stop"
                start = time.monotonic()

                def _emit_visible(fragment: str, *, final: bool = False):
                    nonlocal pending_text, tool_buffer, inside_tool_block, emitted_text
                    if fragment:
                        pending_text += fragment
                    emitted_now: list[str] = []
                    while True:
                        if inside_tool_block:
                            end_idx = pending_text.find(_STREAM_TOOL_END)
                            if end_idx == -1:
                                tool_buffer += pending_text
                                pending_text = ""
                                break
                            tool_buffer += pending_text[:end_idx]
                            pending_text = pending_text[end_idx + len(_STREAM_TOOL_END):]
                            inside_tool_block = False
                            tool_buffer = ""
                            continue

                        start_idx = pending_text.find(_STREAM_TOOL_START)
                        if start_idx != -1:
                            visible = pending_text[:start_idx]
                            if visible:
                                emitted_now.append(visible)
                                emitted_text += visible
                            pending_text = pending_text[start_idx + len(_STREAM_TOOL_START):]
                            inside_tool_block = True
                            tool_buffer = ""
                            continue

                        if not pending_text:
                            break
                        if final:
                            visible = pending_text
                            pending_text = ""
                        else:
                            visible, pending_text = _split_partial_marker_tail(
                                pending_text,
                                _STREAM_TOOL_START,
                            )
                        if visible:
                            emitted_now.append(visible)
                            emitted_text += visible
                        break
                    return emitted_now

                while True:
                    elapsed = time.monotonic() - start
                    remaining = timeout_seconds - elapsed
                    if remaining <= 0:
                        proc.kill()
                        raise TimeoutError(f"Claude CLI timed out after {timeout_seconds:.0f}s")

                    ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], remaining)
                    if not ready:
                        proc.kill()
                        raise TimeoutError(f"Claude CLI timed out after {timeout_seconds:.0f}s")

                    if proc.stderr in ready:
                        err_line = proc.stderr.readline()
                        if err_line:
                            stderr_lines.append(err_line.rstrip("\n"))

                    if proc.stdout in ready:
                        line = proc.stdout.readline()
                        if line:
                            stripped = line.strip()
                            if stripped:
                                try:
                                    payload = json.loads(stripped)
                                except Exception:
                                    _debug_log(f"stream_prompt:json_error preview={stripped[:200]!r}")
                                    continue

                                payload_type = str(payload.get("type") or "").strip().lower()
                                if payload_type == "system":
                                    session_id = str(payload.get("session_id") or "").strip()
                                    if session_id:
                                        self._last_session_id = session_id
                                elif payload_type == "stream_event":
                                    event = payload.get("event") or {}
                                    if not isinstance(event, dict):
                                        event = {}
                                    event_type = str(event.get("type") or "").strip().lower()
                                    if event_type == "content_block_start":
                                        idx = int(event.get("index") or 0)
                                        block = event.get("content_block") or {}
                                        if not isinstance(block, dict):
                                            block = {}
                                        block_types[idx] = str(block.get("type") or "").strip().lower()
                                    elif event_type == "content_block_delta":
                                        idx = int(event.get("index") or 0)
                                        block_type = block_types.get(idx)
                                        delta = event.get("delta") or {}
                                        if not isinstance(delta, dict):
                                            delta = {}
                                        text_delta = str(delta.get("text") or "")
                                        if text_delta and (
                                            block_type == "text"
                                            or str(delta.get("type") or "").strip().lower() == "text_delta"
                                        ):
                                            raw_text_parts.append(text_delta)
                                            for visible in _emit_visible(text_delta):
                                                yield _make_stream_chunk(model=model, content=visible)
                                        reasoning_delta = _coerce_reasoning_delta(block_type, delta)
                                        if reasoning_delta:
                                            yield _make_stream_chunk(model=model, reasoning=reasoning_delta)
                                    elif event_type == "content_block_stop":
                                        idx = int(event.get("index") or 0)
                                        block_types.pop(idx, None)
                                    elif event_type == "message_delta":
                                        delta = event.get("delta") or {}
                                        if not isinstance(delta, dict):
                                            delta = {}
                                        stop = delta.get("stop_reason")
                                        if isinstance(stop, str) and stop.strip():
                                            finish_reason = "tool_calls" if stop == "tool_use" else "stop"
                                        usage_payload = event.get("usage") or {}
                                        if isinstance(usage_payload, dict):
                                            prompt_tokens = int(
                                                usage_payload.get("input_tokens")
                                                or usage_payload.get("cache_creation_input_tokens")
                                                or 0
                                            )
                                            completion_tokens = int(usage_payload.get("output_tokens") or 0)
                                            cached_tokens = int(
                                                usage_payload.get("cache_read_input_tokens") or 0
                                            )
                                            usage = SimpleNamespace(
                                                prompt_tokens=prompt_tokens,
                                                completion_tokens=completion_tokens,
                                                total_tokens=prompt_tokens + completion_tokens,
                                                prompt_tokens_details=SimpleNamespace(
                                                    cached_tokens=cached_tokens
                                                ),
                                            )
                                elif payload_type == "assistant" and not raw_text_parts:
                                    message = payload.get("message") or {}
                                    if not isinstance(message, dict):
                                        message = {}
                                    fallback_assistant_text = _extract_text_from_content_blocks(
                                        message.get("content")
                                    )
                                elif payload_type == "result":
                                    result_payload = payload
                                    session_id = str(payload.get("session_id") or "").strip()
                                    if session_id:
                                        self._last_session_id = session_id
                                        _debug_log(
                                            "stream_prompt:result "
                                            f"session_id={session_id} "
                                            f"stop_reason={payload.get('stop_reason') or ''}"
                                        )
                                    total_cost = payload.get("total_cost_usd")
                                    self._last_total_cost_usd = (
                                        float(total_cost)
                                        if isinstance(total_cost, (int, float))
                                        else None
                                    )
                                    stop_reason = payload.get("stop_reason")
                                    self._last_stop_reason = (
                                        str(stop_reason).strip()
                                        if isinstance(stop_reason, str)
                                        else None
                                    )
                        elif proc.poll() is not None:
                            break

                    if proc.poll() is not None:
                        break

                try:
                    rc = proc.wait(timeout=1)
                except Exception:
                    proc.kill()
                    rc = proc.wait()

                stderr = "\n".join(part for part in stderr_lines if part).strip()
                if rc != 0:
                    raise RuntimeError(
                        f"Claude CLI returned exit code {rc}: {stderr or 'unknown error'}"
                    )

                if fallback_assistant_text and not raw_text_parts:
                    raw_text_parts.append(fallback_assistant_text)

                raw_text = "".join(raw_text_parts).strip()
                tool_calls, cleaned_text = _extract_tool_calls_from_text(raw_text)
                for visible in _emit_visible("", final=True):
                    yield _make_stream_chunk(model=model, content=visible)

                if cleaned_text and len(cleaned_text) > len(emitted_text) and cleaned_text.startswith(
                    emitted_text
                ):
                    tail = cleaned_text[len(emitted_text):]
                    if tail:
                        emitted_text += tail
                        yield _make_stream_chunk(model=model, content=tail)

                if not usage and isinstance(result_payload, dict):
                    usage_payload = result_payload.get("usage") or {}
                    if isinstance(usage_payload, dict):
                        prompt_tokens = int(
                            usage_payload.get("input_tokens")
                            or usage_payload.get("cache_creation_input_tokens")
                            or 0
                        )
                        completion_tokens = int(usage_payload.get("output_tokens") or 0)
                        cached_tokens = int(usage_payload.get("cache_read_input_tokens") or 0)
                        usage = SimpleNamespace(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
                        )

                if tool_calls:
                    finish_reason = "tool_calls"
                    for index, tool_call in enumerate(tool_calls):
                        yield _make_stream_chunk(
                            model=model,
                            tool_call_delta={
                                "index": index,
                                "id": getattr(tool_call, "id", None),
                                "name": getattr(getattr(tool_call, "function", None), "name", ""),
                                "arguments": getattr(
                                    getattr(tool_call, "function", None),
                                    "arguments",
                                    "",
                                ),
                            },
                        )

                yield _make_stream_chunk(
                    model=model,
                    finish_reason=finish_reason,
                    usage=usage,
                )

        return _generator()

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        system_prompt: str,
        model: str | None,
        effort: str | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        normalized_model = _normalize_model(model)
        with _system_prompt_file_args(system_prompt) as system_args:
            command = [
                self._command,
                *self._args,
                "-p",
                "--output-format",
                "json",
                "--tools",
                "",
                "--disable-slash-commands",
                "--strict-mcp-config",
                "--mcp-config",
                _EMPTY_MCP_CONFIG,
                "--setting-sources",
                "user",
                *system_args,
            ]
            if normalized_model:
                command.extend(["--model", normalized_model])
            if effort:
                command.extend(["--effort", effort])
            if self._last_session_id and _resume_enabled():
                command.extend(["--resume", self._last_session_id])

            resolved = shutil.which(command[0]) if command and command[0] else None
            if not resolved:
                raise RuntimeError(
                    f"Could not find Claude CLI command '{command[0]}'. Install Claude Code or set "
                    "HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CLI_PATH."
                )
            command[0] = resolved
            _debug_log(
                "run_prompt:start "
                f"model={normalized_model or ''} "
                f"effort={effort or ''} "
                f"timeout={timeout_seconds:.1f} "
                f"cwd={self._cwd} "
                f"argv_len={sum(len(part) for part in command)} "
                f"prompt_len={len(prompt_text)} "
                f"system_prompt_len={len(system_prompt)}"
            )
            try:
                proc = subprocess.run(
                    command,
                    input=prompt_text,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                    cwd=self._cwd,
                )
            except subprocess.TimeoutExpired as exc:
                _debug_log("run_prompt:timeout")
                raise TimeoutError(f"Claude CLI timed out after {timeout_seconds:.0f}s") from exc

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        _debug_log(
            "run_prompt:done "
            f"rc={proc.returncode} stdout_len={len(stdout)} stderr_len={len(stderr)}"
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude CLI returned exit code {proc.returncode}: {stderr or stdout or 'unknown error'}"
            )

        try:
            payload = json.loads(stdout)
        except Exception as exc:
            _debug_log(f"run_prompt:json_error preview={stdout[:200]!r}")
            raise RuntimeError(f"Claude CLI did not return JSON: {stdout[:500]}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Claude CLI returned unexpected payload shape")

        session_id = str(payload.get("session_id") or "").strip()
        if session_id:
            self._last_session_id = session_id

        total_cost = payload.get("total_cost_usd")
        self._last_total_cost_usd = (
            float(total_cost) if isinstance(total_cost, (int, float)) else None
        )
        stop_reason = payload.get("stop_reason")
        self._last_stop_reason = str(stop_reason).strip() if isinstance(stop_reason, str) else None
        _debug_log(
            "run_prompt:parsed "
            f"session_id={self._last_session_id or ''} "
            f"stop_reason={self._last_stop_reason or ''}"
        )
        return payload
