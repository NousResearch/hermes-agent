"""posthog — Hermes plugin for PostHog AI observability.

Captures Hermes LLM calls as PostHog ``$ai_generation`` events and tool calls
as ``$ai_span`` events. Activation is handled by the Hermes plugin system — the
plugin only loads when listed in ``plugins.enabled``. At runtime it also
requires the optional ``posthog`` SDK and a PostHog project token; if either is
missing, hooks are inert.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from posthog import Posthog
except Exception:  # pragma: no cover - optional dependency
    Posthog = None


@dataclass
class GenerationState:
    span_id: str
    started_at: float
    input_messages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    api_mode: str = ""
    base_url: str = ""
    platform: str = ""


@dataclass
class ToolState:
    span_id: str
    parent_id: str
    started_at: float
    tool_name: str
    args: Any = None


@dataclass
class TraceState:
    trace_id: str
    session_id: str = ""
    task_id: str = ""
    started_at: float = field(default_factory=time.time)
    generations: Dict[str, GenerationState] = field(default_factory=dict)
    tools: Dict[str, ToolState] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list[ToolState]] = field(default_factory=dict)
    current_generation_id: str = ""
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
_TRACE_STATE: Dict[str, TraceState] = {}
_POSTHOG_CLIENT = None
_INIT_FAILED = object()
_READ_FILE_LINE_RE = re.compile(r"^\s*(\d+)\|(.*)$")
_READ_FILE_HEAD_LINES = 25
_READ_FILE_TAIL_LINES = 15
_DEFAULT_MAX_CHARS = 12000
_MAX_TRACE_STATES = 512
_TRACE_STATE_TTL_SECONDS = 60 * 60
_VALID_TOKEN_PREFIXES = ("phc_",)
_PLACEHOLDER_VALUES = {
    "placeholder",
    "test-key",
    "your-posthog-token",
    "your-posthog-key",
    "your-key",
    "change-me",
    "replace_me",
    "xxx",
    "dummy-key-here",
    "<your-key>",
    "<ph_project_token>",
}
_ALLOWED_ID_RE = re.compile(r"[^A-Za-z0-9\-_~.@()!':|]")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(*names: str) -> bool:
    for name in names:
        value = _env(name).lower()
        if value:
            return value in {"1", "true", "yes", "on"}
    return False


def _debug_enabled() -> bool:
    return _env_bool("HERMES_POSTHOG_DEBUG")


def _debug(message: str) -> None:
    if _debug_enabled():
        logger.info("PostHog observability: %s", message)


def _redact_key_preview(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 20:
        return repr(value)
    return repr(value[:6] + "...")


def _validate_project_token(value: str) -> Optional[str]:
    if not value:
        return None
    if value.lower() in _PLACEHOLDER_VALUES:
        return f"HERMES_POSTHOG_PROJECT_TOKEN={_redact_key_preview(value)} looks like a placeholder"
    if value.startswith(_VALID_TOKEN_PREFIXES):
        return None
    # PostHog project tokens are normally phc_...; fail closed for common junk,
    # but allow uncommon/self-hosted token formats that are long enough to be real.
    if len(value) < 16 or any(marker in value.lower() for marker in ("your", "placeholder", "dummy")):
        return (
            f"HERMES_POSTHOG_PROJECT_TOKEN={_redact_key_preview(value)} "
            "does not look like a PostHog project token (expected phc_...)"
        )
    return None


def _get_posthog():
    """Return a cached PostHog client, or ``None`` if unavailable."""
    global _POSTHOG_CLIENT
    if _POSTHOG_CLIENT is _INIT_FAILED:
        return None
    if _POSTHOG_CLIENT is not None:
        return _POSTHOG_CLIENT
    if Posthog is None:
        _POSTHOG_CLIENT = _INIT_FAILED
        return None

    token = _env("HERMES_POSTHOG_PROJECT_TOKEN") or _env("POSTHOG_PROJECT_API_KEY")
    if not token:
        _POSTHOG_CLIENT = _INIT_FAILED
        return None

    token_issue = _validate_project_token(token)
    if token_issue:
        logger.warning(
            "PostHog plugin: project token looks invalid, events will NOT be emitted (%s). "
            "Set a real project token (phc_...) in HERMES_POSTHOG_PROJECT_TOKEN or unset it.",
            token_issue,
        )
        _POSTHOG_CLIENT = _INIT_FAILED
        return None

    host = _env("HERMES_POSTHOG_HOST") or _env("POSTHOG_HOST") or "https://us.i.posthog.com"
    kwargs: Dict[str, Any] = {"host": host}
    # Keep sync mode conservative: the SDK option name has existed across recent
    # versions and makes short-lived Hermes runs more reliable when enabled.
    if _env_bool("HERMES_POSTHOG_SYNC_MODE"):
        kwargs["sync_mode"] = True
    if _env_bool("HERMES_POSTHOG_PRIVACY_MODE"):
        kwargs["privacy_mode"] = True

    try:
        _POSTHOG_CLIENT = Posthog(token, **kwargs)
    except Exception as exc:  # pragma: no cover - fail-open
        logger.warning("Could not initialize PostHog client: %s", exc)
        _POSTHOG_CLIENT = _INIT_FAILED
        return None
    return _POSTHOG_CLIENT


def _trace_key(task_id: str, session_id: str) -> str:
    if task_id:
        return task_id
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _safe_id(value: str) -> str:
    value = str(value or "")
    if not value:
        return uuid.uuid4().hex
    return _ALLOWED_ID_RE.sub("_", value)[:200]


def _new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex
    return f"{prefix}{value}" if prefix else value


def _distinct_id(*, session_id: str = "", task_id: str = "") -> str:
    configured = _env("HERMES_POSTHOG_DISTINCT_ID")
    if configured:
        return configured
    if session_id:
        return f"session:{_safe_id(session_id)}"
    if task_id:
        return f"task:{_safe_id(task_id)}"
    return "hermes-agent"


def _should_sample() -> bool:
    raw = _env("HERMES_POSTHOG_SAMPLE_RATE", "1.0") or "1.0"
    try:
        rate = float(raw)
    except ValueError:
        logger.warning("Invalid HERMES_POSTHOG_SAMPLE_RATE=%r", raw)
        return True
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _maybe_parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in "{[" or stripped[-1] not in "}]":
        if len(stripped) < 2 or stripped[0] not in "{[":
            return value
    try:
        parsed, idx = json.JSONDecoder().raw_decode(stripped)
    except Exception:
        return value
    if not isinstance(parsed, (dict, list)):
        return value
    trailing = stripped[idx:].strip()
    if not trailing:
        return parsed
    if isinstance(parsed, dict):
        merged = dict(parsed)
        merged["_trailing_text"] = trailing
        return merged
    return {"data": parsed, "_trailing_text": trailing}


def _looks_like_read_file_payload(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("content"), str)
        and "total_lines" in value
        and "file_size" in value
        and "is_binary" in value
        and "is_image" in value
        and not value.get("error")
    )


def _parse_read_file_lines(content: str) -> list[dict[str, Any]]:
    if not isinstance(content, str) or not content:
        return []
    lines = []
    for raw_line in content.splitlines():
        match = _READ_FILE_LINE_RE.match(raw_line)
        if not match:
            return []
        lines.append({"line": int(match.group(1)), "text": match.group(2)})
    return lines


def _build_read_file_preview(lines: list[dict[str, Any]]) -> dict[str, Any]:
    if len(lines) <= (_READ_FILE_HEAD_LINES + _READ_FILE_TAIL_LINES):
        return {"lines": lines}
    return {
        "head": lines[:_READ_FILE_HEAD_LINES],
        "tail": lines[-_READ_FILE_TAIL_LINES:],
        "omitted_line_count": len(lines) - _READ_FILE_HEAD_LINES - _READ_FILE_TAIL_LINES,
    }


def _normalize_read_file_payload(value: dict[str, Any], *, args: Any = None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(args, dict):
        for key in ("path", "offset", "limit"):
            if key in args:
                normalized[key] = args[key]
    lines = _parse_read_file_lines(value.get("content", ""))
    if lines:
        normalized["returned_lines"] = {"start": lines[0]["line"], "end": lines[-1]["line"], "count": len(lines)}
        normalized["content_preview"] = _build_read_file_preview(lines)
    elif value.get("content"):
        normalized["content_preview"] = {"text": value.get("content", "")}
    for key in ("total_lines", "file_size", "truncated", "is_binary", "is_image", "hint", "mime_type", "dimensions", "error"):
        if key in value:
            normalized[key] = value[key]
    if isinstance(value.get("base64_content"), str) and value["base64_content"]:
        normalized["base64_content"] = {"omitted": True, "length": len(value["base64_content"])}
    return normalized


def _normalize_payload(value: Any, *, tool_name: str = "", args: Any = None) -> Any:
    if _looks_like_read_file_payload(value):
        return _normalize_read_file_payload(value, args=args if tool_name == "read_file" else None)
    return value


def _max_chars() -> int:
    raw = _env("HERMES_POSTHOG_MAX_CHARS", str(_DEFAULT_MAX_CHARS)) or str(_DEFAULT_MAX_CHARS)
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid HERMES_POSTHOG_MAX_CHARS=%r; using %s", raw, _DEFAULT_MAX_CHARS)
        return _DEFAULT_MAX_CHARS
    if value <= 0:
        logger.warning("Invalid HERMES_POSTHOG_MAX_CHARS=%r; using %s", raw, _DEFAULT_MAX_CHARS)
        return _DEFAULT_MAX_CHARS
    return value


def _safe_value(value: Any, *, max_chars: Optional[int] = None, depth: int = 0, parse_json_strings: bool = False) -> Any:
    max_chars = max_chars if max_chars is not None else _max_chars()
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, str):
        if parse_json_strings:
            parsed = _maybe_parse_json_string(value)
            if parsed is not value:
                return _safe_value(parsed, max_chars=max_chars, depth=depth, parse_json_strings=True)
        return _truncate_text(value, max_chars)
    if isinstance(value, dict):
        normalized = _normalize_payload(value)
        if normalized is not value:
            return _safe_value(normalized, max_chars=max_chars, depth=depth, parse_json_strings=parse_json_strings)
        return {str(k): _safe_value(v, max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(v, max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings) for v in list(value)[:50]]
    if hasattr(value, "__dict__"):
        return _safe_value(vars(value), max_chars=max_chars, depth=depth + 1, parse_json_strings=parse_json_strings)
    return _truncate_text(repr(value), max_chars)


def _coerce_request_messages(*, request_messages: Any = None, messages: Any = None, conversation_history: Any = None, user_message: Any = None) -> list[dict[str, Any]]:
    for candidate in (request_messages, messages, conversation_history):
        if isinstance(candidate, list):
            return candidate
    if user_message is None:
        return []
    return [{"role": "user", "content": user_message}]


def _serialize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    serialized = []
    for message in messages[-12:]:
        if not isinstance(message, dict):
            continue
        item = {"role": message.get("role"), "content": _safe_value(message.get("content"), parse_json_strings=(message.get("role") == "tool"))}
        if message.get("tool_call_id"):
            item["tool_call_id"] = message.get("tool_call_id")
        if message.get("name"):
            item["name"] = _safe_value(message.get("name"))
        if message.get("tool_calls"):
            item["tool_calls"] = _safe_value(message.get("tool_calls"), parse_json_strings=True)
        serialized.append(item)
    return serialized


def _serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    serialized = []
    for tool_call in tool_calls:
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", None) if fn else None
        arguments = getattr(fn, "arguments", None) if fn else None
        serialized.append({
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", None) or "function",
            "name": name,
            "arguments": _safe_value(arguments),
            "function": {"name": name, "arguments": _safe_value(arguments)},
        })
    return serialized


def _serialize_assistant_output(*, assistant_message: Any = None, assistant_response: Any = None, assistant_content_chars: int = 0, assistant_tool_call_count: int = 0) -> list[dict[str, Any]]:
    if assistant_message is not None:
        return [{
            "role": "assistant",
            "content": _safe_value(getattr(assistant_message, "content", None)),
            "reasoning": _safe_value(getattr(assistant_message, "reasoning", None)),
            "tool_calls": _serialize_tool_calls(getattr(assistant_message, "tool_calls", None)),
        }]
    if assistant_response is not None:
        return [{"role": "assistant", "content": _safe_value(assistant_response)}]
    return [{
        "role": "assistant",
        "content": f"[{assistant_content_chars} chars]" if assistant_content_chars else None,
        "tool_calls": [{"id": f"tc_{i}"} for i in range(assistant_tool_call_count)] if assistant_tool_call_count else [],
    }]


def _tool_names_from_output(output_choices: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for choice in output_choices:
        calls = choice.get("tool_calls") if isinstance(choice, dict) else None
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict) and call.get("name"):
                    names.append(str(call["name"]))
    return names


def _usage_tokens(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    details: dict[str, int] = {}
    if input_tokens:
        details["$ai_input_tokens"] = int(input_tokens)
    if output_tokens:
        details["$ai_output_tokens"] = int(output_tokens)
    for src, dst in (
        ("cache_read_tokens", "$ai_cache_read_input_tokens"),
        ("cache_write_tokens", "$ai_cache_creation_input_tokens"),
        ("reasoning_tokens", "$ai_reasoning_tokens"),
    ):
        if usage.get(src):
            details[dst] = int(usage[src])
    return details


def _capture(event: str, *, distinct_id: str, properties: dict[str, Any]) -> None:
    client = _get_posthog()
    if client is None or not _should_sample():
        return
    try:
        client.capture(distinct_id=distinct_id, event=event, properties=properties)
        if hasattr(client, "flush"):
            client.flush()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"capture failed: {exc}")


def _base_properties(state: TraceState) -> dict[str, Any]:
    props: dict[str, Any] = {"$ai_trace_id": state.trace_id}
    if state.session_id:
        props["$ai_session_id"] = state.session_id
    env = _env("HERMES_POSTHOG_ENV")
    release = _env("HERMES_POSTHOG_RELEASE")
    if env:
        props["hermes.environment"] = env
    if release:
        props["hermes.release"] = release
    if state.task_id:
        props["hermes.task_id"] = state.task_id
    return props


def _prune_trace_states(now: Optional[float] = None) -> None:
    """Bound in-memory trace bookkeeping for long-running gateway processes."""
    if not _TRACE_STATE:
        return
    now = now or time.time()
    stale_keys = [
        key for key, state in _TRACE_STATE.items()
        if now - state.last_updated_at > _TRACE_STATE_TTL_SECONDS
    ]
    for key in stale_keys:
        _TRACE_STATE.pop(key, None)
    if len(_TRACE_STATE) <= _MAX_TRACE_STATES:
        return
    overflow = len(_TRACE_STATE) - _MAX_TRACE_STATES
    oldest = sorted(_TRACE_STATE.items(), key=lambda item: item[1].last_updated_at)[:overflow]
    for key, _state in oldest:
        _TRACE_STATE.pop(key, None)


def _ensure_trace_state(task_key: str, *, task_id: str = "", session_id: str = "") -> TraceState:
    now = time.time()
    _prune_trace_states(now)
    state = _TRACE_STATE.get(task_key)
    if state is None:
        seed = task_id or session_id or _new_id()
        state = TraceState(trace_id=_safe_id(seed), session_id=session_id, task_id=task_id)
        _TRACE_STATE[task_key] = state
    state.last_updated_at = now
    return state


def on_pre_llm_call(*, task_id: str = "", session_id: str = "", messages: Any = None, **kwargs: Any) -> None:
    # Legacy Hermes hook shape: only request-like when messages is a list.
    if not isinstance(messages, list):
        return
    on_pre_llm_request(task_id=task_id, session_id=session_id, messages=messages, **kwargs)


def on_pre_llm_request(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    **_: Any,
) -> None:
    if _get_posthog() is None:
        return
    input_messages = _serialize_messages(_coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    ))
    task_key = _trace_key(task_id, session_id)
    req_key = str(api_call_count or 0)
    with _STATE_LOCK:
        state = _ensure_trace_state(task_key, task_id=task_id, session_id=session_id)
        span_id = _new_id("gen_")
        state.current_generation_id = span_id
        state.generations[req_key] = GenerationState(
            span_id=span_id,
            started_at=time.time(),
            input_messages=input_messages,
            model=model,
            provider=provider,
            api_mode=api_mode,
            base_url=base_url,
            platform=platform,
        )


def on_post_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    model: str = "",
    api_call_count: int = 0,
    assistant_message: Any = None,
    response: Any = None,
    api_duration: float = 0.0,
    finish_reason: str = "",
    usage: Any = None,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    assistant_response: Any = None,
    **_: Any,
) -> None:
    if _get_posthog() is None:
        return
    task_key = _trace_key(task_id, session_id)
    req_key = str(api_call_count or 0)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(req_key, None) if state else None
    if state is None or generation is None:
        return

    output_choices = _serialize_assistant_output(
        assistant_message=assistant_message,
        assistant_response=assistant_response,
        assistant_content_chars=assistant_content_chars,
        assistant_tool_call_count=assistant_tool_call_count,
    )
    tool_names = _tool_names_from_output(output_choices)
    raw_usage = usage or getattr(response, "usage", None)
    props = _base_properties(state)
    props.update({
        "$ai_span_id": generation.span_id,
        "$ai_span_name": f"LLM call {api_call_count}",
        "$ai_model": model or generation.model,
        "$ai_provider": provider or generation.provider,
        "$ai_latency": round(float(api_duration), 3) if api_duration else round(time.time() - generation.started_at, 3),
        "$ai_tools_called": tool_names,
        "$ai_tool_call_count": len(tool_names) or assistant_tool_call_count,
        "hermes.api_mode": api_mode or generation.api_mode,
        "hermes.platform": generation.platform,
        "hermes.base_url": base_url or generation.base_url,
    })
    props.update(_usage_tokens(raw_usage))
    if finish_reason:
        props["hermes.finish_reason"] = finish_reason
    if not _env_bool("HERMES_POSTHOG_PRIVACY_MODE"):
        props["$ai_input"] = generation.input_messages
        props["$ai_output_choices"] = output_choices

    _capture("$ai_generation", distinct_id=_distinct_id(session_id=session_id, task_id=task_id), properties=props)


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, task_id: str = "", session_id: str = "", tool_call_id: str = "", **_: Any) -> None:
    if _get_posthog() is None:
        return
    task_key = _trace_key(task_id, session_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _ensure_trace_state(task_key, task_id=task_id, session_id=session_id)
        tool = ToolState(
            span_id=_new_id("tool_"),
            parent_id=state.current_generation_id or state.trace_id,
            started_at=time.time(),
            tool_name=tool_name,
            args=_safe_value(args, parse_json_strings=True),
        )
        if tool_call_id:
            state.tools[tool_call_id] = tool
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(tool)


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None, task_id: str = "", session_id: str = "", tool_call_id: str = "", **_: Any) -> None:
    if _get_posthog() is None:
        return
    task_key = _trace_key(task_id, session_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        tool = None
        if tool_call_id:
            tool = state.tools.pop(tool_call_id, None)
        if tool is None:
            queue = state.pending_tools_by_name.get(tool_name)
            if queue:
                tool = queue.pop(0)
                if not queue:
                    state.pending_tools_by_name.pop(tool_name, None)
    if tool is None:
        return

    result_value = _maybe_parse_json_string(result) if isinstance(result, str) else result
    result_value = _normalize_payload(result_value, tool_name=tool_name, args=args)
    props = _base_properties(state)
    props.update({
        "$ai_span_id": tool.span_id,
        "$ai_parent_id": tool.parent_id,
        "$ai_span_name": f"Tool: {tool_name}",
        "$ai_input_state": tool.args,
        "$ai_output_state": _safe_value(result_value, parse_json_strings=True),
        "$ai_latency": round(time.time() - tool.started_at, 3),
        "hermes.tool_name": tool_name,
        "hermes.tool_call_id": tool_call_id,
    })
    _capture("$ai_span", distinct_id=_distinct_id(session_id=session_id, task_id=task_id), properties=props)


def register(ctx) -> None:
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
