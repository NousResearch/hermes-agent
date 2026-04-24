"""OpenAI-compatible facade that routes Hermes requests through Claude Code CLI.

This adapter lets Hermes treat ``claude -p`` as a chat-style backend.
It disables Claude Code built-in tools and asks the model to emit OpenAI-style
tool calls inside ``<tool_call>{...}</tool_call>`` blocks when Hermes tools are
needed. The client remembers the Claude session id and resumes it on later
turns within the same Hermes session.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://local"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)
_SHELL_FENCE_RE = re.compile(
    r"```(?:bash|sh|zsh|shell|terminal|console)\s*\n(?P<command>.*?)\n```",
    re.IGNORECASE | re.DOTALL,
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


def _persistent_worker_enabled() -> bool:
    raw = os.getenv("HERMES_CLAUDE_CLI_PERSISTENT", "").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def _broker_enabled() -> bool:
    raw = os.getenv("HERMES_CLAUDE_CLI_BROKER", "").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def _broker_socket_path() -> str:
    raw = os.getenv("HERMES_CLAUDE_CLI_BROKER_SOCKET", "").strip()
    if raw:
        return str(Path(raw).expanduser())
    try:
        uid = str(os.getuid())
    except Exception:
        uid = "nouid"
    return str(Path(tempfile.gettempdir()) / f"hermes-claude-cli-broker-{uid}.sock")


def _broker_log_path() -> str:
    raw = os.getenv("HERMES_CLAUDE_CLI_BROKER_LOG", "").strip()
    if raw:
        return str(Path(raw).expanduser())
    try:
        uid = str(os.getuid())
    except Exception:
        uid = "nouid"
    return str(Path(tempfile.gettempdir()) / f"hermes-claude-cli-broker-{uid}.log")


def _broker_startup_timeout() -> float:
    raw = os.getenv("HERMES_CLAUDE_CLI_BROKER_STARTUP_TIMEOUT", "").strip()
    try:
        return max(0.25, float(raw)) if raw else 2.0
    except Exception:
        return 2.0


def _broker_connect_timeout() -> float:
    raw = os.getenv("HERMES_CLAUDE_CLI_BROKER_CONNECT_TIMEOUT", "").strip()
    try:
        return max(0.05, float(raw)) if raw else 0.25
    except Exception:
        return 0.25


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


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _strip_runtime_enabled(explicit: Any = None) -> bool:
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str) and explicit.strip():
        return _truthy(explicit)
    raw = os.getenv("HERMES_CLAUDE_CLI_STRIP_RUNTIME", "").strip().lower()
    if not raw:
        return True
    return _truthy(raw)


def _build_process_env(*, strip_runtime: bool) -> dict[str, str]:
    env = dict(os.environ)
    if strip_runtime:
        env.update(_DEFAULT_STRIPPED_RUNTIME_ENV)
    else:
        for key in _DEFAULT_STRIPPED_RUNTIME_ENV:
            env.pop(key, None)
    return env


class _NonBlockingLineReader:
    """Read subprocess lines without TextIOWrapper/select buffering stalls."""

    def __init__(self, pipe: Any):
        self._pipe = pipe
        self._buffer = b""
        self._fd: int | None = None
        try:
            fd = pipe.fileno()
        except Exception:
            fd = None
        if isinstance(fd, int):
            try:
                os.set_blocking(fd, False)
                self._fd = fd
            except Exception:
                self._fd = None

    @property
    def selectable(self) -> Any:
        return self._fd if self._fd is not None else self._pipe

    def read_ready_lines(self) -> list[str]:
        if self._fd is None:
            try:
                line = self._pipe.readline()
            except Exception:
                return []
            if not line:
                return []
            if isinstance(line, bytes):
                return [line.decode("utf-8", errors="replace")]
            return [str(line)]

        lines: list[str] = []
        while True:
            try:
                chunk = os.read(self._fd, 65536)
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            except OSError:
                break
            if not chunk:
                if self._buffer:
                    lines.append(self._buffer.decode("utf-8", errors="replace"))
                    self._buffer = b""
                break
            self._buffer += chunk
            while True:
                newline_index = self._buffer.find(b"\n")
                if newline_index < 0:
                    break
                raw_line = self._buffer[: newline_index + 1]
                self._buffer = self._buffer[newline_index + 1 :]
                lines.append(raw_line.decode("utf-8", errors="replace"))
        return lines


def _slice_between(text: str, start_marker: str, end_marker: str | None = None) -> str:
    if not text:
        return ""
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = len(text)
    if end_marker:
        marker_end = text.find(end_marker, start)
        if marker_end >= 0:
            end = marker_end
    return text[start:end].strip()


def _lean_claude_cli_system_prompt(system_prompt: str) -> str:
    text = str(system_prompt or "").strip()
    if not text:
        return ""

    parts: list[str] = []

    persistent_memory_marker = "You have persistent memory across sessions."
    memory_section_marker = "MEMORY (your personal notes)"
    skills_section_marker = "## Skills (mandatory)"

    prefix_end = text.find(persistent_memory_marker)
    if prefix_end > 0:
        prefix = text[:prefix_end].strip()
        if prefix:
            parts.append(prefix)
    else:
        parts.append(text[:2000].strip())

    memory_and_profile = _slice_between(
        text,
        memory_section_marker,
        skills_section_marker,
    )
    if memory_and_profile:
        parts.append(memory_and_profile)

    parts.append(
        "Use Hermes tools when relevant. Keep the existing voice and user context, "
        "but do not self-update skills or persist new memories unless the user asks."
    )

    lean = "\n\n".join(part for part in parts if part).strip()
    return lean or text


def _summarize_tool_specs(
    tools: list[dict[str, Any]] | None,
    *,
    compact: bool = False,
) -> list[dict[str, Any]]:
    tool_specs: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return tool_specs

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        description = str(fn.get("description") or "").strip()
        parameters = fn.get("parameters", {})
        if compact:
            if description:
                description = description.splitlines()[0].strip()[:160]
            props: dict[str, Any] = {}
            required: list[str] = []
            if isinstance(parameters, dict):
                raw_props = parameters.get("properties")
                if isinstance(raw_props, dict):
                    props = raw_props
                raw_required = parameters.get("required")
                if isinstance(raw_required, list):
                    required = [str(item) for item in raw_required[:10]]
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": description,
                    "args": list(props.keys())[:10],
                    "required": required,
                }
            )
            continue

        tool_specs.append(
            {
                "name": name.strip(),
                "description": description,
                "parameters": parameters,
            }
        )

    return tool_specs



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


def _coerce_compact_tool_arguments(arguments: Any) -> Any:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except Exception:
            return stripped
    return arguments


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
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        payload = {
            "name": name,
            "arguments": _coerce_compact_tool_arguments(function.get("arguments")),
        }
        rendered_calls.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
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
    *,
    compact_tools: bool = False,
) -> str:
    sections: list[str] = [
        "If you need a Hermes tool, emit exactly <tool_call>{...}</tool_call>.",
        "Use the minimal tool-call JSON shape: {\"name\":\"tool_name\",\"arguments\":{...}}.",
        "Do not include optional id/type fields unless needed, and do not stringify the arguments object.",
        "Transcript tool requests and tool results are literal prior Hermes tool traffic.",
    ]

    tool_specs = _summarize_tool_specs(tools, compact=compact_tools)
    if tool_specs:
        label = "Available Hermes tools (compact schema):" if compact_tools else "Available Hermes tools (OpenAI function schema):"
        sections.append(label + "\n" + json.dumps(tool_specs, ensure_ascii=False, separators=(",", ":")))

    if tool_choice not in (None, "auto"):
        sections.append(
            "Tool choice hint: " + json.dumps(tool_choice, ensure_ascii=False, separators=(",", ":"))
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
    compact_tools: bool = False,
) -> str:
    sections: list[str] = [
        "If you need a Hermes tool, emit exactly <tool_call>{...}</tool_call>.",
        "Prefer the minimal tool-call JSON shape: {\"name\":\"tool_name\",\"arguments\":{...}}.",
        "Do not include optional id/type fields unless needed, and do not stringify the arguments object.",
        "Tool result messages arrive as plain user messages prefixed with 'Tool result (...)'.",
    ]

    tool_specs = _summarize_tool_specs(tools, compact=compact_tools)
    if tool_specs:
        label = "Available Hermes tools (compact schema):" if compact_tools else "Available Hermes tools (OpenAI function schema):"
        sections.append(label + "\n" + json.dumps(tool_specs, ensure_ascii=False, separators=(",", ":")))

    if tool_choice not in (None, "auto"):
        sections.append(
            "Tool choice hint: " + json.dumps(tool_choice, ensure_ascii=False, separators=(",", ":"))
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


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            obj, end_offset = decoder.raw_decode(text[start:])
        except Exception:
            index = start + 1
            continue
        end = start + end_offset
        yield obj, start, end
        index = max(end, start + 1)


def _looks_like_action_shell_fence(text: str, match: re.Match) -> bool:
    before = text[:match.start()].strip().lower()
    after = text[match.end():].strip().lower()
    if after:
        # If there is substantial prose after the fence, it is more likely an
        # explanatory answer than an intended terminal action.
        if len(after) > 80:
            return False
    if not before:
        return True
    cues = (
        "on it", "let me", "lemme", "i'll", "i will", "i’m", "i'm",
        "checking", "inspect", "look", "run", "try", "doing", "getting",
        "retry", "now", "for real", "first", "next",
    )
    return len(before) <= 240 and any(cue in before for cue in cues)


def _is_self_negated_tool_json(text: str) -> bool:
    lowered = text.lower()
    return (
        "not a real tool call" in lowered
        or "not real tool call" in lowered
        or "fake tool call" in lowered
    )


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

    def _try_add_tool_call(raw_json: str) -> bool:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return False
        return _try_add_tool_call_obj(obj)

    def _try_add_tool_call_obj(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        fn = obj.get("function")
        if not isinstance(fn, dict):
            fn = {
                "name": obj.get("name"),
                "arguments": obj.get("arguments", obj.get("input", {})),
            }
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return False
        fn_args = fn.get("arguments", fn.get("input", {}))
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False, separators=(",", ":"))
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"claude_cli_call_{len(extracted)+1}"

        extracted.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type=str(obj.get("type") or "function"),
                function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
            )
        )
        return True


    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        if _try_add_tool_call(match.group(1)):
            consumed_spans.append((match.start(), match.end()))

    if not extracted:
        for match in _TOOL_CALL_JSON_RE.finditer(text):
            if _try_add_tool_call(match.group(0)):
                consumed_spans.append((match.start(), match.end()))

    if not extracted and not _is_self_negated_tool_json(text):
        for obj, start, end in _iter_json_objects(text):
            if _try_add_tool_call_obj(obj):
                consumed_spans.append((start, end))
                break

    if not extracted:
        for match in _SHELL_FENCE_RE.finditer(text):
            command = (match.group("command") or "").strip()
            if not command or not _looks_like_action_shell_fence(text, match):
                continue
            _try_add_tool_call_obj({
                "name": "terminal",
                "arguments": {"command": command},
            })
            consumed_spans.append((match.start(), match.end()))
            break

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



def _tool_call_signature(tool_call: Any) -> tuple[str, str]:
    function = getattr(tool_call, "function", None)
    name = str(getattr(function, "name", "") or "").strip()
    arguments = str(getattr(function, "arguments", "") or "")
    return name, arguments


def _filter_already_emitted_tool_calls(
    tool_calls: list[SimpleNamespace],
    emitted_tool_calls: list[SimpleNamespace],
) -> list[SimpleNamespace]:
    if not emitted_tool_calls or not tool_calls:
        return tool_calls

    remaining: dict[tuple[str, str], int] = {}
    for tool_call in emitted_tool_calls:
        signature = _tool_call_signature(tool_call)
        remaining[signature] = remaining.get(signature, 0) + 1

    fresh: list[SimpleNamespace] = []
    for tool_call in tool_calls:
        signature = _tool_call_signature(tool_call)
        count = remaining.get(signature, 0)
        if count > 0:
            remaining[signature] = count - 1
            continue
        fresh.append(tool_call)
    return fresh


def _extract_tool_calls_from_closed_stream_block(block_text: str) -> list[SimpleNamespace]:
    wrapped = f"{_STREAM_TOOL_START}{block_text}{_STREAM_TOOL_END}"
    tool_calls, _ = _extract_tool_calls_from_text(wrapped)
    return tool_calls


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


def _chunk_hydration_text(text: str, max_chars: int = 22000) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    remaining = cleaned
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break
        window = remaining[:max_chars]
        minimum_split = int(max_chars * 0.65)
        split_at = window.rfind("\n\n")
        if split_at < minimum_split:
            split_at = window.rfind("\n")
        if split_at < minimum_split:
            split_at = max_chars
            split_at = max_chars
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return [chunk for chunk in chunks if chunk]


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


def _usage_to_broker_payload(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def _usage_from_broker_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    prompt_tokens = int(payload.get("prompt_tokens") or 0)
    completion_tokens = int(payload.get("completion_tokens") or 0)
    total_tokens = int(payload.get("total_tokens") or prompt_tokens + completion_tokens)
    cached_tokens = int(payload.get("cached_tokens") or 0)
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def _stream_chunk_to_broker_payload(chunk: Any) -> dict[str, Any]:
    choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
    delta = getattr(choice, "delta", None) if choice is not None else None
    tool_call_delta = None
    tool_calls = getattr(delta, "tool_calls", None) if delta is not None else None
    if tool_calls:
        tool_call = tool_calls[0]
        function = getattr(tool_call, "function", None)
        tool_call_delta = {
            "index": int(getattr(tool_call, "index", 0) or 0),
            "id": getattr(tool_call, "id", None),
            "name": getattr(function, "name", "") or "",
            "arguments": getattr(function, "arguments", "") or "",
        }
    return {
        "model": getattr(chunk, "model", None),
        "content": getattr(delta, "content", None) or "",
        "reasoning": (
            getattr(delta, "reasoning", None)
            or getattr(delta, "reasoning_content", None)
            or ""
        ),
        "tool_call_delta": tool_call_delta,
        "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
        "usage": _usage_to_broker_payload(getattr(chunk, "usage", None)),
    }


def _stream_chunk_from_broker_payload(payload: Any, *, model: str) -> _ClaudeCLIStreamChunk:
    if not isinstance(payload, dict):
        payload = {}
    return _make_stream_chunk(
        model=str(payload.get("model") or model),
        content=str(payload.get("content") or ""),
        reasoning=str(payload.get("reasoning") or ""),
        tool_call_delta=(
            payload.get("tool_call_delta")
            if isinstance(payload.get("tool_call_delta"), dict)
            else None
        ),
        finish_reason=payload.get("finish_reason"),
        usage=_usage_from_broker_payload(payload.get("usage")),
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
        strip_runtime: bool | None = None,
        timeout: float | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = claude_command or command or _resolve_command()
        self._args = list(claude_args or args or _resolve_args())
        self._cwd = _resolve_cwd(claude_cwd)
        self._strip_runtime = _strip_runtime_enabled(strip_runtime)
        self._process_env = _build_process_env(strip_runtime=self._strip_runtime)
        self._timeout = (
            float(timeout) if isinstance(timeout, (int, float)) else _DEFAULT_TIMEOUT_SECONDS
        )
        self._last_session_id: str | None = None
        self._last_total_cost_usd: float | None = None
        self._last_stop_reason: str | None = None
        self._hydrated_session_id: str | None = None
        self._hydrated_prompt_hash: str | None = None
        self._broker_enabled = _broker_enabled()
        self._persistent_enabled = _persistent_worker_enabled() or self._broker_enabled
        self._broker_process: subprocess.Popen | None = None
        self._worker_lock = threading.RLock()
        self._worker_process: subprocess.Popen | None = None
        self._worker_signature: tuple[str, str, str] | None = None
        self._worker_system_prompt_path: Path | None = None
        self.chat = _ClaudeCLIChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self._close_persistent_worker(reason="client_close")
        self.is_closed = True

    def import_transport_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict):
            return
        session_id = state.get("claude_session_id")
        self._last_session_id = (
            str(session_id).strip() or None
            if session_id is not None
            else None
        )
        hydrated_session_id = state.get("hydrated_session_id")
        self._hydrated_session_id = (
            str(hydrated_session_id).strip() or None
            if hydrated_session_id is not None
            else None
        )
        hydrated_prompt_hash = state.get("hydrated_prompt_hash")
        self._hydrated_prompt_hash = (
            str(hydrated_prompt_hash).strip() or None
            if hydrated_prompt_hash is not None
            else None
        )

    def export_transport_state(self) -> dict[str, Any] | None:
        if not self._last_session_id:
            return None
        state: dict[str, Any] = {"claude_session_id": self._last_session_id}
        if self._hydrated_session_id:
            state["hydrated_session_id"] = self._hydrated_session_id
        if self._hydrated_prompt_hash:
            state["hydrated_prompt_hash"] = self._hydrated_prompt_hash
        return state

    def _persistent_worker_signature(
        self,
        *,
        model: str | None,
        system_prompt: str,
        effort: str | None,
    ) -> tuple[str, str, str]:
        return (_normalize_model(model) or "", str(system_prompt or ""), effort or "")

    def _build_persistent_user_event(
        self,
        *,
        prompt_text: str,
        structured_input: str | None,
    ) -> str:
        payload = None
        if isinstance(structured_input, str) and structured_input.strip():
            payload = structured_input.strip()
        else:
            payload = _make_user_event(prompt_text)
        if not payload:
            raise RuntimeError("Claude CLI persistent worker requires a non-empty user event payload")
        return payload if payload.endswith("\n") else payload + "\n"

    def _create_persistent_system_prompt_file(self, system_prompt: str) -> Path | None:
        content = str(system_prompt or "").strip()
        if not content:
            return None
        fd, raw_path = tempfile.mkstemp(prefix="hermes-claude-system-", suffix=".txt")
        path = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def _close_persistent_worker(self, *, reason: str) -> None:
        with self._worker_lock:
            proc = self._worker_process
            prompt_path = self._worker_system_prompt_path
            self._worker_process = None
            self._worker_signature = None
            self._worker_system_prompt_path = None

        if proc is not None:
            _debug_log(
                "persistent:close "
                f"reason={reason} pid={getattr(proc, 'pid', 0)} session_id={self._last_session_id or ''}"
            )
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except Exception:
                        pass

        if prompt_path is not None:
            try:
                prompt_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _broker_config_payload(self) -> dict[str, Any]:
        return {
            "command": self._command,
            "args": list(self._args),
            "cwd": self._cwd,
            "strip_runtime": self._strip_runtime,
            "timeout": self._timeout,
        }

    def _connect_broker_socket(self, socket_path: str, *, timeout: float | None = None) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_broker_connect_timeout() if timeout is None else timeout)
        try:
            sock.connect(socket_path)
        except Exception:
            sock.close()
            raise
        return sock

    def _start_broker_process(self, socket_path: str) -> None:
        proc = self._broker_process
        if proc is not None and proc.poll() is None:
            return

        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["HERMES_CLAUDE_CLI_BROKER"] = "0"
        env["HERMES_CLAUDE_CLI_PERSISTENT"] = "1"
        env.setdefault("PYTHONUNBUFFERED", "1")

        log_path = Path(_broker_log_path())
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} broker:start socket={socket_path}\n"
            )
            log_handle.flush()
            self._broker_process = subprocess.Popen(
                [sys.executable, "-m", "agent.claude_cli_broker", socket_path],
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=log_handle,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )

    def _ensure_broker_socket(self) -> socket.socket:
        socket_path = _broker_socket_path()
        try:
            return self._connect_broker_socket(socket_path)
        except OSError as first_exc:
            _debug_log(f"broker:connect_miss socket={socket_path} error={type(first_exc).__name__}:{first_exc}")

        self._start_broker_process(socket_path)
        deadline = time.monotonic() + _broker_startup_timeout()
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self._connect_broker_socket(socket_path, timeout=0.2)
            except OSError as exc:
                last_exc = exc
                time.sleep(0.05)
        raise RuntimeError(
            f"Claude CLI broker did not become ready at {socket_path}: {last_exc or 'unknown error'}"
        )

    def _stream_completion_broker(
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
            sock = self._ensure_broker_socket()
            sock.settimeout(max(timeout_seconds + 10.0, 30.0))
            reader = None
            writer = None
            try:
                reader = sock.makefile("r", encoding="utf-8")
                writer = sock.makefile("w", encoding="utf-8")
                request = {
                    "action": "stream",
                    "request_id": uuid.uuid4().hex,
                    "config": self._broker_config_payload(),
                    "transport_state": self.export_transport_state(),
                    "stream": {
                        "model": model,
                        "prompt_text": prompt_text,
                        "system_prompt": system_prompt,
                        "structured_input": structured_input,
                        "structured_system_prompt": structured_system_prompt,
                        "effort": effort,
                        "timeout_seconds": timeout_seconds,
                    },
                }
                writer.write(json.dumps(request, separators=(",", ":"), ensure_ascii=False) + "\n")
                writer.flush()

                while True:
                    line = reader.readline()
                    if not line:
                        raise RuntimeError("Claude CLI broker closed the stream before done")
                    try:
                        payload = json.loads(line)
                    except Exception as exc:
                        raise RuntimeError(f"Claude CLI broker returned invalid JSON: {line[:200]!r}") from exc
                    if not isinstance(payload, dict):
                        continue
                    event_type = str(payload.get("type") or "").strip().lower()
                    if event_type == "state":
                        state = payload.get("state")
                        if isinstance(state, dict):
                            self.import_transport_state(state)
                    elif event_type == "chunk":
                        yield _stream_chunk_from_broker_payload(payload.get("chunk"), model=model)
                    elif event_type == "error":
                        raise RuntimeError(str(payload.get("error") or "Claude CLI broker error"))
                    elif event_type == "done":
                        state = payload.get("state")
                        if isinstance(state, dict):
                            self.import_transport_state(state)
                        return
            finally:
                for handle in (writer, reader):
                    try:
                        if handle is not None:
                            handle.close()
                    except Exception:
                        pass
                try:
                    sock.close()
                except Exception:
                    pass

        return _generator()

    def _ensure_persistent_worker(
        self,
        *,
        model: str | None,
        system_prompt: str,
        effort: str | None,
    ) -> subprocess.Popen:
        if not self._persistent_enabled:
            raise RuntimeError("Claude CLI persistent worker is disabled")

        signature = self._persistent_worker_signature(
            model=model,
            system_prompt=system_prompt,
            effort=effort,
        )
        with self._worker_lock:
            proc = self._worker_process
            if proc is not None and proc.poll() is None and self._worker_signature == signature:
                _debug_log(
                    "persistent:reuse "
                    f"pid={getattr(proc, 'pid', 0)} session_id={self._last_session_id or ''}"
                )
                return proc

        self._close_persistent_worker(reason="worker_recreate")

        normalized_model = _normalize_model(model)
        prompt_path = self._create_persistent_system_prompt_file(system_prompt)
        system_args: list[str] = []
        if prompt_path is not None:
            system_args = [_system_prompt_flag(), str(prompt_path)]

        command = [
            self._command,
            *self._args,
            "-p",
            "--verbose",
            "--input-format",
            "stream-json",
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
            if prompt_path is not None:
                prompt_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not find Claude CLI command '{command[0]}'. Install Claude Code or set "
                "HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CLI_PATH."
            )
        command[0] = resolved

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._cwd,
                env=self._process_env,
                bufsize=1,
            )
        except Exception:
            if prompt_path is not None:
                prompt_path.unlink(missing_ok=True)
            raise

        with self._worker_lock:
            self._worker_process = proc
            self._worker_signature = signature
            self._worker_system_prompt_path = prompt_path

        _debug_log(
            "persistent:start "
            f"pid={getattr(proc, 'pid', 0)} model={normalized_model or ''} effort={effort or ''} "
            f"session_id={self._last_session_id or ''} system_prompt_len={len(system_prompt)}"
        )
        return proc

    def _stream_completion_live(
        self,
        *,
        model: str,
        prompt_text: str | None,
        prompt_text_factory: Callable[[], str] | None = None,
        system_prompt: str,
        structured_input: str | None,
        structured_system_prompt: str,
        effort: str | None,
        timeout_seconds: float,
    ):
        def _generator():
            prompt_text_cache = prompt_text

            def _resolve_prompt_text() -> str:
                nonlocal prompt_text_cache
                if prompt_text_cache is None:
                    prompt_text_cache = prompt_text_factory() if prompt_text_factory else ""
                return prompt_text_cache

            persistent_prompt_text = lambda: (
                "" if (structured_input and structured_input.strip()) else _resolve_prompt_text()
            )

            if self._broker_enabled:
                yielded_any = False
                try:
                    for chunk in self._stream_completion_broker(
                        model=model,
                        prompt_text=persistent_prompt_text(),
                        system_prompt=system_prompt,
                        structured_input=structured_input,
                        structured_system_prompt=structured_system_prompt,
                        effort=effort,
                        timeout_seconds=timeout_seconds,
                    ):
                        yielded_any = True
                        yield chunk
                    return
                except Exception as exc:
                    _debug_log(
                        "broker:fallback_stream "
                        f"yielded_any={yielded_any} error={type(exc).__name__}:{exc}"
                    )
                    if yielded_any:
                        raise

            if self._persistent_enabled:
                yielded_any = False
                try:
                    for chunk in self._stream_completion_persistent(
                        model=model,
                        prompt_text=persistent_prompt_text(),
                        system_prompt=system_prompt,
                        structured_input=structured_input,
                        structured_system_prompt=structured_system_prompt,
                        effort=effort,
                        timeout_seconds=timeout_seconds,
                    ):
                        yielded_any = True
                        yield chunk
                    return
                except Exception as exc:
                    _debug_log(
                        "persistent:fallback_stream "
                        f"yielded_any={yielded_any} error={type(exc).__name__}:{exc}"
                    )
                    self._close_persistent_worker(reason="stream_error")
                    if yielded_any:
                        raise

            yield from self._stream_completion_oneshot(
                model=model,
                prompt_text=_resolve_prompt_text(),
                system_prompt=system_prompt,
                structured_input=structured_input,
                structured_system_prompt=structured_system_prompt,
                effort=effort,
                timeout_seconds=timeout_seconds,
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
        if self._persistent_enabled:
            try:
                chunks = self._stream_completion_persistent(
                    model=model or "claude-cli",
                    prompt_text=prompt_text,
                    system_prompt=system_prompt,
                    structured_input=None,
                    structured_system_prompt=system_prompt,
                    effort=effort,
                    timeout_seconds=timeout_seconds,
                )
                text_parts: list[str] = []
                usage_obj = None
                finish_reason = "stop"
                for chunk in chunks:
                    choice = chunk.choices[0] if chunk.choices else None
                    delta = getattr(choice, "delta", None) if choice is not None else None
                    content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        text_parts.append(str(content))
                    if choice is not None and getattr(choice, "finish_reason", None):
                        finish_reason = str(choice.finish_reason)
                    if getattr(chunk, "usage", None) is not None:
                        usage_obj = chunk.usage
                usage_payload = None
                if usage_obj is not None:
                    usage_payload = {
                        "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                        "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                        "cache_read_input_tokens": int(
                            getattr(getattr(usage_obj, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                        ),
                    }
                return {
                    "result": "".join(text_parts).strip(),
                    "session_id": self._last_session_id or "",
                    "stop_reason": self._last_stop_reason or finish_reason,
                    "total_cost_usd": self._last_total_cost_usd,
                    "usage": usage_payload or {},
                }
            except Exception as exc:
                _debug_log(f"persistent:fallback_prompt error={type(exc).__name__}:{exc}")
                self._close_persistent_worker(reason="prompt_error")

        return self._run_prompt_oneshot(
            prompt_text,
            system_prompt=system_prompt,
            model=model,
            effort=effort,
            timeout_seconds=timeout_seconds,
        )


    def _stream_completion_persistent(
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
            requested_structured = bool(structured_input and structured_input.strip())
            effective_system_prompt = (
                structured_system_prompt
                if requested_structured and structured_system_prompt.strip()
                else system_prompt
            )
            stdin_payload = self._build_persistent_user_event(
                prompt_text=prompt_text,
                structured_input=structured_input,
            )
            proc = self._ensure_persistent_worker(
                model=normalized_model,
                system_prompt=effective_system_prompt,
                effort=effort,
            )
            _debug_log(
                'persistent:turn_start '
                f"pid={getattr(proc, 'pid', 0)} model={normalized_model or ''} "
                f"requested_structured={requested_structured} stdin_len={len(stdin_payload)} "
                f"session_id={self._last_session_id or ''}"
            )

            with self._worker_lock:
                if proc.poll() is not None:
                    raise RuntimeError('Claude CLI persistent worker exited before turn start')
                assert proc.stdin is not None
                assert proc.stdout is not None
                assert proc.stderr is not None
                stdout_reader = _NonBlockingLineReader(proc.stdout)
                stderr_reader = _NonBlockingLineReader(proc.stderr)
                try:
                    proc.stdin.write(stdin_payload)
                    proc.stdin.flush()
                except Exception as exc:
                    raise RuntimeError(f'Failed to write to Claude CLI persistent worker: {exc}') from exc

                block_types: dict[int, str] = {}
                stderr_lines: list[str] = []
                raw_text_parts: list[str] = []
                fallback_assistant_text = ''
                pending_text = ''
                tool_buffer = ''
                inside_tool_block = False
                emitted_text = ''
                streamed_tool_calls: list[SimpleNamespace] = []
                result_payload: dict[str, Any] | None = None
                usage = None
                finish_reason = 'stop'
                start = time.monotonic()
                early_finish_emitted = False

                def _emit_visible(fragment: str, *, final: bool = False) -> tuple[list[str], list[SimpleNamespace]]:
                    nonlocal pending_text, tool_buffer, inside_tool_block, emitted_text
                    if fragment:
                        pending_text += fragment
                    emitted_now: list[str] = []
                    emitted_tool_calls: list[SimpleNamespace] = []
                    while True:
                        if inside_tool_block:
                            end_idx = pending_text.find(_STREAM_TOOL_END)
                            if end_idx == -1:
                                tool_buffer += pending_text
                                pending_text = ''
                                break
                            tool_buffer += pending_text[:end_idx]
                            pending_text = pending_text[end_idx + len(_STREAM_TOOL_END):]
                            inside_tool_block = False
                            parsed_tool_calls = _extract_tool_calls_from_closed_stream_block(tool_buffer)
                            if parsed_tool_calls:
                                emitted_tool_calls.extend(parsed_tool_calls)
                            tool_buffer = ''
                            continue

                        start_idx = pending_text.find(_STREAM_TOOL_START)
                        if start_idx != -1:
                            visible = pending_text[:start_idx]
                            if visible:
                                emitted_now.append(visible)
                                emitted_text += visible
                            pending_text = pending_text[start_idx + len(_STREAM_TOOL_START):]
                            inside_tool_block = True
                            tool_buffer = ''
                            continue

                        if not pending_text:
                            break
                        if final:
                            visible = pending_text
                            pending_text = ''
                        else:
                            visible, pending_text = _split_partial_marker_tail(
                                pending_text,
                                _STREAM_TOOL_START,
                            )
                        if visible:
                            emitted_now.append(visible)
                            emitted_text += visible
                        break
                    return emitted_now, emitted_tool_calls

                def _yield_stream_tool_call_chunks(tool_calls_now: list[SimpleNamespace]):
                    nonlocal streamed_tool_calls
                    if not tool_calls_now:
                        return
                    start_index = len(streamed_tool_calls)
                    streamed_tool_calls.extend(tool_calls_now)
                    for offset, tool_call in enumerate(tool_calls_now):
                        yield _make_stream_chunk(
                            model=model,
                            tool_call_delta={
                                'index': start_index + offset,
                                'id': getattr(tool_call, 'id', None),
                                'name': getattr(getattr(tool_call, 'function', None), 'name', ''),
                                'arguments': getattr(
                                    getattr(tool_call, 'function', None),
                                    'arguments',
                                    '',
                                ),
                            },
                        )

                def _handle_stdout_line(line: str):
                    nonlocal fallback_assistant_text, finish_reason, result_payload, usage
                    stripped = line.strip()
                    if not stripped:
                        return
                    try:
                        payload = json.loads(stripped)
                    except Exception:
                        _debug_log(f"persistent:json_error preview={stripped[:200]!r}")
                        return

                    payload_type = str(payload.get('type') or '').strip().lower()
                    if payload_type == 'system':
                        session_id = str(payload.get('session_id') or '').strip()
                        if session_id:
                            self._last_session_id = session_id
                    elif payload_type == 'stream_event':
                        event = payload.get('event') or {}
                        if not isinstance(event, dict):
                            event = {}
                        event_type = str(event.get('type') or '').strip().lower()
                        if event_type == 'content_block_start':
                            idx = int(event.get('index') or 0)
                            block = event.get('content_block') or {}
                            if not isinstance(block, dict):
                                block = {}
                            block_types[idx] = str(block.get('type') or '').strip().lower()
                        elif event_type == 'content_block_delta':
                            idx = int(event.get('index') or 0)
                            block_type = block_types.get(idx)
                            delta = event.get('delta') or {}
                            if not isinstance(delta, dict):
                                delta = {}
                            text_delta = str(delta.get('text') or '')
                            if text_delta and (
                                block_type == 'text'
                                or str(delta.get('type') or '').strip().lower() == 'text_delta'
                            ):
                                raw_text_parts.append(text_delta)
                                visible_fragments, tool_calls_now = _emit_visible(text_delta)
                                for visible in visible_fragments:
                                    yield _make_stream_chunk(model=model, content=visible)
                                for tool_chunk in _yield_stream_tool_call_chunks(tool_calls_now):
                                    yield tool_chunk
                            reasoning_delta = _coerce_reasoning_delta(block_type, delta)
                            if reasoning_delta:
                                yield _make_stream_chunk(model=model, reasoning=reasoning_delta)
                        elif event_type == 'content_block_stop':
                            idx = int(event.get('index') or 0)
                            block_types.pop(idx, None)
                        elif event_type == 'message_delta':
                            delta = event.get('delta') or {}
                            if not isinstance(delta, dict):
                                delta = {}
                            stop = delta.get('stop_reason')
                            if isinstance(stop, str) and stop.strip():
                                finish_reason = 'tool_calls' if stop == 'tool_use' else 'stop'
                            usage_payload = event.get('usage') or {}
                            if isinstance(usage_payload, dict):
                                prompt_tokens = int(
                                    usage_payload.get('input_tokens')
                                    or usage_payload.get('cache_creation_input_tokens')
                                    or 0
                                )
                                completion_tokens = int(usage_payload.get('output_tokens') or 0)
                                cached_tokens = int(usage_payload.get('cache_read_input_tokens') or 0)
                                usage = SimpleNamespace(
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=prompt_tokens + completion_tokens,
                                    prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
                                )
                    elif payload_type == 'assistant' and not raw_text_parts:
                        message = payload.get('message') or {}
                        if not isinstance(message, dict):
                            message = {}
                        fallback_assistant_text = _extract_text_from_content_blocks(message.get('content'))
                    elif payload_type == 'result':
                        result_payload = payload
                        session_id = str(payload.get('session_id') or '').strip()
                        if session_id:
                            self._last_session_id = session_id
                        total_cost = payload.get('total_cost_usd')
                        self._last_total_cost_usd = (
                            float(total_cost)
                            if isinstance(total_cost, (int, float))
                            else None
                        )
                        stop_reason = payload.get('stop_reason')
                        self._last_stop_reason = (
                            str(stop_reason).strip()
                            if isinstance(stop_reason, str)
                            else None
                        )

                def _drain_after_early_finish() -> None:
                    deadline = time.monotonic() + max(5.0, min(timeout_seconds, 60.0))
                    try:
                        with self._worker_lock:
                            while True:
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    raise TimeoutError('Claude CLI fast-drain timed out')

                                ready, _, _ = select.select(
                                    [stdout_reader.selectable, stderr_reader.selectable],
                                    [],
                                    [],
                                    remaining,
                                )
                                if not ready:
                                    raise TimeoutError('Claude CLI fast-drain timed out')

                                if stderr_reader.selectable in ready:
                                    for err_line in stderr_reader.read_ready_lines():
                                        if err_line:
                                            stderr_lines.append(err_line.rstrip('\n'))

                                if stdout_reader.selectable in ready:
                                    for line in stdout_reader.read_ready_lines():
                                        stripped = line.strip()
                                        if not stripped:
                                            continue
                                        try:
                                            payload = json.loads(stripped)
                                        except Exception:
                                            _debug_log(f"persistent:fast_drain_json_error preview={stripped[:200]!r}")
                                            continue

                                        payload_type = str(payload.get('type') or '').strip().lower()
                                        if payload_type == 'system':
                                            session_id = str(payload.get('session_id') or '').strip()
                                            if session_id:
                                                self._last_session_id = session_id
                                        elif payload_type == 'result':
                                            session_id = str(payload.get('session_id') or '').strip()
                                            if session_id:
                                                self._last_session_id = session_id
                                            total_cost = payload.get('total_cost_usd')
                                            self._last_total_cost_usd = (
                                                float(total_cost)
                                                if isinstance(total_cost, (int, float))
                                                else None
                                            )
                                            stop_reason = payload.get('stop_reason')
                                            self._last_stop_reason = (
                                                str(stop_reason).strip()
                                                if isinstance(stop_reason, str)
                                                else None
                                            )
                                            _debug_log(
                                                'persistent:fast_drain_done '
                                                f"pid={getattr(proc, 'pid', 0)} "
                                                f"session_id={self._last_session_id or ''}"
                                            )
                                            return

                                if proc.poll() is not None:
                                    break
                    except Exception as exc:
                        _debug_log(f"persistent:fast_drain_error error={type(exc).__name__}:{exc}")
                        self._close_persistent_worker(reason='fast_drain_error')

                try:
                    while result_payload is None:
                        elapsed = time.monotonic() - start
                        remaining = timeout_seconds - elapsed
                        if remaining <= 0:
                            raise TimeoutError(f'Claude CLI timed out after {timeout_seconds:.0f}s')

                        ready, _, _ = select.select(
                            [stdout_reader.selectable, stderr_reader.selectable],
                            [],
                            [],
                            remaining,
                        )
                        if not ready:
                            raise TimeoutError(f'Claude CLI timed out after {timeout_seconds:.0f}s')

                        if stderr_reader.selectable in ready:
                            for err_line in stderr_reader.read_ready_lines():
                                if err_line:
                                    stderr_lines.append(err_line.rstrip('\n'))

                        if stdout_reader.selectable in ready:
                            stdout_lines = stdout_reader.read_ready_lines()
                            if stdout_lines:
                                for line in stdout_lines:
                                    for chunk in _handle_stdout_line(line):
                                        yield chunk
                            elif proc.poll() is not None:
                                break

                        if (
                            finish_reason == 'tool_calls'
                            and streamed_tool_calls
                            and not inside_tool_block
                            and not early_finish_emitted
                        ):
                            visible_fragments, tool_calls_now = _emit_visible('', final=True)
                            for visible in visible_fragments:
                                yield _make_stream_chunk(model=model, content=visible)
                            for tool_chunk in _yield_stream_tool_call_chunks(tool_calls_now):
                                yield tool_chunk
                            early_finish_emitted = True
                            yield _make_stream_chunk(
                                model=model,
                                finish_reason=finish_reason,
                                usage=usage,
                            )

                        if proc.poll() is not None and result_payload is None:
                            break
                except GeneratorExit:
                    if early_finish_emitted:
                        threading.Thread(target=_drain_after_early_finish, daemon=True).start()
                    raise

                stderr = '\n'.join(part for part in stderr_lines if part).strip()
                if result_payload is None and not fallback_assistant_text and not raw_text_parts:
                    raise RuntimeError(
                        f"Claude CLI persistent worker exited before returning a result: {stderr or 'unknown error'}"
                    )

                if fallback_assistant_text and not raw_text_parts:
                    raw_text_parts.append(fallback_assistant_text)
                if not raw_text_parts and isinstance(result_payload, dict):
                    result_text = str(result_payload.get('result') or '').strip()
                    if result_text:
                        raw_text_parts.append(result_text)

                raw_text = ''.join(raw_text_parts).strip()
                visible_fragments, tool_calls_now = _emit_visible('', final=True)
                for visible in visible_fragments:
                    yield _make_stream_chunk(model=model, content=visible)
                for tool_chunk in _yield_stream_tool_call_chunks(tool_calls_now):
                    yield tool_chunk

                tool_calls, cleaned_text = _extract_tool_calls_from_text(raw_text)
                tool_calls = _filter_already_emitted_tool_calls(tool_calls, streamed_tool_calls)

                if cleaned_text and len(cleaned_text) > len(emitted_text) and cleaned_text.startswith(
                    emitted_text
                ):
                    tail = cleaned_text[len(emitted_text):]
                    if tail:
                        emitted_text += tail
                        yield _make_stream_chunk(model=model, content=tail)

                if not usage and isinstance(result_payload, dict):
                    usage_payload = result_payload.get('usage') or {}
                    if isinstance(usage_payload, dict):
                        prompt_tokens = int(
                            usage_payload.get('input_tokens')
                            or usage_payload.get('cache_creation_input_tokens')
                            or 0
                        )
                        completion_tokens = int(usage_payload.get('output_tokens') or 0)
                        cached_tokens = int(
                            usage_payload.get('cache_read_input_tokens') or 0
                        )
                        usage = SimpleNamespace(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
                        )

                if tool_calls and not early_finish_emitted:
                    finish_reason = 'tool_calls'
                    for index, tool_call in enumerate(tool_calls, start=len(streamed_tool_calls)):
                        yield _make_stream_chunk(
                            model=model,
                            tool_call_delta={
                                'index': index,
                                'id': getattr(tool_call, 'id', None),
                                'name': getattr(getattr(tool_call, 'function', None), 'name', ''),
                                'arguments': getattr(
                                    getattr(tool_call, 'function', None),
                                    'arguments',
                                    '',
                                ),
                            },
                        )

                if not early_finish_emitted:
                    yield _make_stream_chunk(
                        model=model,
                        finish_reason=finish_reason,
                        usage=usage,
                    )

        return _generator()


    def _ensure_bootstrap_hydrated(
        self,
        *,
        model: str | None,
        full_system_prompt: str,
        runtime_system_prompt: str | None = None,
        effort: str | None,
        timeout_seconds: float,
    ) -> None:
        if not self._strip_runtime:
            return
        hydration_source = str(full_system_prompt or "").strip()
        if not hydration_source:
            return

        prompt_hash = hashlib.sha1(hydration_source.encode("utf-8")).hexdigest()
        if isinstance(runtime_system_prompt, str) and runtime_system_prompt.strip():
            bootstrap_system_prompt = runtime_system_prompt.strip()
        else:
            bootstrap_system_prompt = (
                _lean_claude_cli_system_prompt(full_system_prompt)
                if self._persistent_enabled
                else ""
            )

        if (
            self._last_session_id
            and self._hydrated_session_id == self._last_session_id
            and self._hydrated_prompt_hash == prompt_hash
        ):
            return

        chunks = _chunk_hydration_text(hydration_source, max_chars=22000)
        hydration_effort = "low" if effort else None
        _debug_log(
            "hydrate:start "
            f"session_id={self._last_session_id} chunks={len(chunks)} chars={len(hydration_source)} "
            f"effort={hydration_effort or ''}"
        )
        for idx, chunk in enumerate(chunks, start=1):
            hidden_prompt = (
                f"Internal Hermes context block {idx}/{len(chunks)} for this session. "
                "Store and follow it for later turns in this session. "
                "Do not summarize it. Do not call any tools. Reply with exactly OK.\n\n"
                f"{chunk}"
            )
            result = self._run_prompt_oneshot(
                hidden_prompt,
                system_prompt=bootstrap_system_prompt,
                model=model,
                effort=hydration_effort,
                timeout_seconds=min(timeout_seconds, 120.0),
            )
            _debug_log(
                "hydrate:chunk_done "
                f"idx={idx}/{len(chunks)} session_id={self._last_session_id or ''} "
                f"result={str(result.get('result') or '')[:24]!r}"
            )

        if not self._last_session_id:
            _debug_log("hydrate:missing_session_id_after_chunks")
            return

        self._hydrated_session_id = self._last_session_id
        self._hydrated_prompt_hash = prompt_hash
        _debug_log(
            "hydrate:complete "
            f"session_id={self._last_session_id or ''} hash={prompt_hash[:10]}"
        )

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
        full_system_prompt = system_prompt
        if self._strip_runtime:
            original_system_prompt = system_prompt
            system_prompt = _lean_claude_cli_system_prompt(system_prompt)
            _debug_log(
                "create:lean_system_prompt "
                f"before={len(original_system_prompt)} after={len(system_prompt)}"
            )
        tool_guidance = _build_tool_guidance(
            tools=tools,
            tool_choice=tool_choice,
            compact_tools=self._strip_runtime,
        )
        prompt_text_cache: str | None = None

        def _get_prompt_text() -> str:
            nonlocal prompt_text_cache
            if prompt_text_cache is None:
                prompt_text_cache = _format_messages_as_prompt(
                    prompt_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    compact_tools=self._strip_runtime,
                )
            return prompt_text_cache

        effort = _extract_effort(extra_kwargs)
        structured_stream_input: str | None = None
        structured_stream_system_prompt = system_prompt

        if self._last_session_id and _resume_enabled():
            structured_stream_input = _build_resume_delta_payload(prompt_messages)
            if not structured_stream_input:
                # Some resumed turns arrive as a single fresh user message.
                # Keep those on the compact stream-json lane instead of
                # falling back to a full prompt replay.
                structured_stream_input = _build_initial_structured_payload(prompt_messages)
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

        if self._strip_runtime and full_system_prompt:
            self._ensure_bootstrap_hydrated(
                model=model,
                full_system_prompt=full_system_prompt,
                runtime_system_prompt=structured_stream_system_prompt or system_prompt,
                effort=effort,
                timeout_seconds=effective_timeout,
            )

        if stream:
            return self._stream_completion_live(
                model=model or "claude-cli",
                prompt_text=None if structured_stream_input else _get_prompt_text(),
                prompt_text_factory=_get_prompt_text,
                system_prompt=system_prompt,
                structured_input=structured_stream_input,
                structured_system_prompt=structured_stream_system_prompt,
                effort=effort,
                timeout_seconds=effective_timeout,
            )

        prompt_text = _get_prompt_text()
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

    def _stream_completion_oneshot(
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
                        env=self._process_env,
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

                try:
                    stderr_tail = proc.stderr.read()
                except Exception:
                    stderr_tail = ""
                if stderr_tail:
                    stderr_lines.extend(line for line in stderr_tail.splitlines() if line)

                stderr = "\n".join(part for part in stderr_lines if part).strip()
                if rc != 0 and not fallback_assistant_text and not raw_text_parts and not isinstance(result_payload, dict):
                    raise RuntimeError(
                        f"Claude CLI returned exit code {rc}: {stderr or 'unknown error'}"
                    )
                if rc != 0:
                    _debug_log(
                        "stream_prompt:nonzero_with_result "
                        f"rc={rc} stderr_len={len(stderr)} "
                        f"has_result={isinstance(result_payload, dict)} "
                        f"has_fallback={bool(fallback_assistant_text)} "
                        f"raw_parts={len(raw_text_parts)}"
                    )

                if fallback_assistant_text and not raw_text_parts:
                    raw_text_parts.append(fallback_assistant_text)
                if not raw_text_parts and isinstance(result_payload, dict):
                    result_text = str(result_payload.get("result") or "").strip()
                    if result_text:
                        raw_text_parts.append(result_text)

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

    def _run_prompt_oneshot(
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
                    env=self._process_env,
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
