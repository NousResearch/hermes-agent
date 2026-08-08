"""Pure helpers for Completion Observation Trace Envelope v1.9.

This module intentionally has no runtime integration, persistence, transport,
database, or provider dependencies.  It creates and updates an in-memory
observability envelope only; callers decide whether and where to keep it.
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA = "hermes.runtime.completion_observation_trace"
VERSION = "1.9"

DEFAULT_PREVIEW_LIMIT = 2048
MAX_CONTAINER_ITEMS = 20
MAX_NESTING_DEPTH = 5

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|password|passwd|pwd|secret|token)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?[^\s,'\"]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,'\"]+"),
)
_BASE64ISH_RE = re.compile(r"^[A-Za-z0-9+/=\r\n]+$")
_MULTIMODAL_TYPES = {
    "image",
    "image_url",
    "input_image",
    "audio",
    "input_audio",
    "video",
    "file",
    "media",
}
_BLOB_KEYS = {
    "base64",
    "b64",
    "blob",
    "bytes",
    "data",
    "image",
    "image_base64",
    "audio",
    "video",
}
_UNTRUSTED_TOOL_PREFIXES = ("web_", "browser_", "mcp_")
_UNTRUSTED_TOOL_NAMES = {"web_search", "web_extract", "browser", "mcp"}


@dataclass(frozen=True)
class _Preview:
    text: str | None
    redacted: bool = False
    multimodal: bool = False
    content_part_types: tuple[str, ...] = ()
    multimodal_refs: tuple[str, ...] = ()
    large_refs: tuple[str, ...] = ()


def new_completion_trace(
    agent: Any | None = None,
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    api_request_id: str | None = None,
    platform: Mapping[str, Any] | None = None,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Completion Observation Trace Envelope v1.9.

    ``agent`` is optional and duck-typed so this module stays pure. Explicit
    keyword arguments take precedence over attributes discovered on ``agent``.
    """

    platform_info = {
        "source": _first_not_none(platform, "source", default=_get_attr(agent, "source")),
        "profile": _first_not_none(platform, "profile", default=_get_attr(agent, "profile")),
        "terminal_backend": _first_not_none(
            platform,
            "terminal_backend",
            default=_get_attr(agent, "terminal_backend"),
        ),
        "cwd": _first_not_none(platform, "cwd", default=_get_attr(agent, "cwd")),
        "host_kind": _first_not_none(platform, "host_kind", default="unknown"),
    }
    model_info = {
        "provider": _first_not_none(model, "provider", default=_get_attr(agent, "provider")),
        "model": _first_not_none(model, "model", default=_get_attr(agent, "model")),
        "api_mode": _first_not_none(model, "api_mode", default=_get_attr(agent, "api_mode")),
        "fallback_used": bool(_first_not_none(model, "fallback_used", default=False)),
        "fallback_chain": list(_first_not_none(model, "fallback_chain", default=[])),
    }

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "session_id": session_id if session_id is not None else _get_attr(agent, "session_id"),
        "turn_id": turn_id if turn_id is not None else _get_attr(agent, "turn_id"),
        "api_request_id": api_request_id,
        "task_id": task_id,
        "platform": platform_info,
        "model": model_info,
        "completion": {
            "status": "tool_calls_pending",
            "finish_reason": None,
            "turn_exit_reason": None,
            "assistant_message_id": None,
            "final_response_present": False,
        },
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated_cost": None,
        },
        "observations": [],
        "tool_trace": [],
        "safety": {
            "contains_untrusted_tool_data": False,
            "redacted": False,
            "multimodal_result_refs": [],
            "large_result_refs": [],
        },
    }


def record_observation(
    trace: dict[str, Any],
    *,
    kind: str,
    summary: str,
    message_index: int | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    status: str | None = None,
    error_type: str | None = None,
    duration_ms: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a normalized observation with a stable monotonically increasing sequence."""

    observations = trace.setdefault("observations", [])
    safe_metadata = _sanitize_for_preview(metadata or {}, limit=DEFAULT_PREVIEW_LIMIT)
    observation = {
        "kind": kind,
        "ts": time.time(),
        "sequence": len(observations) + 1,
        "summary": _redact_text(_truncate(str(summary), DEFAULT_PREVIEW_LIMIT)),
        "message_index": message_index,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "status": status,
        "error_type": error_type,
        "duration_ms": duration_ms,
        "metadata": safe_metadata,
    }
    observations.append(observation)
    if _contains_redaction_marker(safe_metadata) or _contains_redaction_marker(observation["summary"]):
        trace.setdefault("safety", {})["redacted"] = True
    return observation


def record_tool_result(
    trace: dict[str, Any],
    *,
    tool_name: str,
    tool_call_id: str,
    args: Mapping[str, Any] | None = None,
    result: Any = None,
    status: str,
    duration_ms: int | None = None,
    middleware_trace: Sequence[Mapping[str, Any]] | None = None,
    parallel_group: int | None = None,
    preview_limit: int = DEFAULT_PREVIEW_LIMIT,
) -> dict[str, Any]:
    """Record a tool result using bounded, redacted, summary-only previews."""

    args_preview = _make_preview(args or {}, limit=preview_limit)
    result_preview = _make_preview(result, limit=preview_limit, prefer_multimodal_summary=True)
    middleware_preview = _sanitize_for_preview(middleware_trace or [], limit=preview_limit)

    entry = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "args_preview": args_preview.text,
        "result_preview": result_preview.text,
        "status": status,
        "duration_ms": duration_ms,
        "middleware_trace": middleware_preview,
        "parallel_group": parallel_group,
    }
    if result_preview.multimodal:
        entry["metadata"] = {
            "multimodal": True,
            "content_part_types": list(result_preview.content_part_types),
        }

    trace.setdefault("tool_trace", []).append(entry)
    safety = trace.setdefault("safety", {})
    if _is_untrusted_tool(tool_name):
        safety["contains_untrusted_tool_data"] = True
    if args_preview.redacted or result_preview.redacted or _contains_redaction_marker(middleware_preview):
        safety["redacted"] = True
    _extend_unique(safety.setdefault("multimodal_result_refs", []), result_preview.multimodal_refs)
    _extend_unique(safety.setdefault("large_result_refs", []), args_preview.large_refs + result_preview.large_refs)

    record_observation(
        trace,
        kind="tool_result",
        summary=f"{tool_name} completed with status={status}",
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        status=status,
        duration_ms=duration_ms,
        metadata={
            "args_preview_truncated": _is_truncated(args_preview.text),
            "result_preview_truncated": _is_truncated(result_preview.text),
            "multimodal": result_preview.multimodal,
        },
    )
    return entry


def finalize_completion_trace(
    trace: dict[str, Any],
    *,
    status: str,
    finish_reason: str | None = None,
    turn_exit_reason: str | None = None,
    usage: Mapping[str, Any] | None = None,
    assistant_message_id: str | None = None,
    final_response_present: bool | None = None,
) -> dict[str, Any]:
    """Finalize completion and usage fields in-place and return ``trace``."""

    completion = trace.setdefault("completion", {})
    completion["status"] = status
    completion["finish_reason"] = finish_reason
    completion["turn_exit_reason"] = turn_exit_reason
    if assistant_message_id is not None:
        completion["assistant_message_id"] = assistant_message_id
    if final_response_present is not None:
        completion["final_response_present"] = final_response_present

    if usage:
        normalized_usage = trace.setdefault("usage", {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost"):
            if key in usage:
                normalized_usage[key] = usage[key]
    return trace


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_not_none(mapping: Mapping[str, Any] | None, key: str, *, default: Any = None) -> Any:
    if mapping is not None and mapping.get(key) is not None:
        return mapping[key]
    return default


def _make_preview(value: Any, *, limit: int, prefer_multimodal_summary: bool = False) -> _Preview:
    multimodal = _detect_multimodal(value)
    if prefer_multimodal_summary and multimodal.multimodal:
        summary = "[multimodal tool result: summary-only; blobs omitted"
        if multimodal.content_part_types:
            summary += f"; parts={','.join(multimodal.content_part_types)}"
        summary += "]"
        return _Preview(
            text=summary,
            redacted=multimodal.redacted,
            multimodal=True,
            content_part_types=multimodal.content_part_types,
            multimodal_refs=multimodal.multimodal_refs,
            large_refs=multimodal.large_refs,
        )

    sanitized = _sanitize_for_preview(value, limit=limit)
    text = _json_preview(sanitized, limit=limit)
    redacted = _contains_redaction_marker(sanitized) or _contains_redaction_marker(text)
    large_refs = tuple(_collect_large_markers(sanitized))
    return _Preview(text=text, redacted=redacted, large_refs=large_refs)


def _sanitize_for_preview(value: Any, *, limit: int, depth: int = 0, path: str = "$") -> Any:
    if depth > MAX_NESTING_DEPTH:
        return "[preview omitted: max nesting depth]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        return f"[binary data omitted: {len(value)} bytes at {path}]"
    if isinstance(value, str):
        return _sanitize_string(value, limit=limit, path=path)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= MAX_CONTAINER_ITEMS:
                sanitized["[items_omitted]"] = len(value) - MAX_CONTAINER_ITEMS
                break
            key_str = str(key)
            child_path = f"{path}.{key_str}"
            if _SECRET_KEY_RE.search(key_str):
                sanitized[key_str] = "[REDACTED]"
            elif key_str.lower() in _BLOB_KEYS and _looks_like_blob(item):
                sanitized[key_str] = f"[large/blob data omitted at {child_path}]"
            else:
                sanitized[key_str] = _sanitize_for_preview(item, limit=limit, depth=depth + 1, path=child_path)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = [
            _sanitize_for_preview(item, limit=limit, depth=depth + 1, path=f"{path}[{idx}]")
            for idx, item in enumerate(list(value)[:MAX_CONTAINER_ITEMS])
        ]
        if len(value) > MAX_CONTAINER_ITEMS:
            items.append(f"[items omitted: {len(value) - MAX_CONTAINER_ITEMS}]")
        return items
    return _sanitize_string(repr(value), limit=limit, path=path)


def _sanitize_string(value: str, *, limit: int, path: str) -> str:
    if _looks_like_base64_blob(value):
        return f"[large/base64 data omitted: {len(value)} chars at {path}]"
    redacted = _redact_text(value)
    return _truncate(redacted, limit)


def _json_preview(value: Any, *, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        text = repr(value)
    return _truncate(text, limit)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit] + f"...[truncated {omitted} chars]"


def _is_truncated(value: str | None) -> bool:
    return bool(value and "...[truncated " in value)


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]", redacted)
    return redacted


def _contains_redaction_marker(value: Any) -> bool:
    if isinstance(value, str):
        return "[REDACTED]" in value or "data omitted" in value
    if isinstance(value, Mapping):
        return any(_contains_redaction_marker(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_contains_redaction_marker(v) for v in value)
    return False


def _looks_like_blob(value: Any) -> bool:
    if isinstance(value, bytes | bytearray | memoryview):
        return True
    return isinstance(value, str) and _looks_like_base64_blob(value)


def _looks_like_base64_blob(value: str) -> bool:
    compact = "".join(value.split())
    if len(compact) < 256 or len(compact) % 4 != 0 or not _BASE64ISH_RE.match(compact):
        return False
    try:
        base64.b64decode(compact[:4096], validate=True)
    except Exception:
        return False
    return True


def _detect_multimodal(value: Any) -> _Preview:
    refs: list[str] = []
    large_refs: list[str] = []
    part_types: list[str] = []
    redacted = False

    def visit(item: Any, path: str, depth: int) -> None:
        nonlocal redacted
        if depth > MAX_NESTING_DEPTH:
            return
        if isinstance(item, Mapping):
            item_type = str(item.get("type", "")).lower()
            if item_type in _MULTIMODAL_TYPES:
                part_types.append(item_type)
                ref = item.get("url") or item.get("path") or item.get("file_id") or item.get("ref")
                if ref:
                    refs.append(str(ref))
            for key, child in item.items():
                key_str = str(key)
                if key_str.lower() in _BLOB_KEYS and _looks_like_blob(child):
                    part_types.append(key_str.lower())
                    large_refs.append(f"{path}.{key_str}")
                    redacted = True
                else:
                    visit(child, f"{path}.{key_str}", depth + 1)
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            for idx, child in enumerate(item[:MAX_CONTAINER_ITEMS]):
                visit(child, f"{path}[{idx}]", depth + 1)
        elif isinstance(item, bytes | bytearray | memoryview):
            large_refs.append(path)
            redacted = True

    visit(value, "$", 0)
    unique_types = tuple(dict.fromkeys(part_types))
    return _Preview(
        text=None,
        redacted=redacted,
        multimodal=bool(unique_types),
        content_part_types=unique_types,
        multimodal_refs=tuple(dict.fromkeys(refs)),
        large_refs=tuple(dict.fromkeys(large_refs)),
    )


def _collect_large_markers(value: Any) -> list[str]:
    markers: list[str] = []
    if isinstance(value, str) and ("data omitted" in value or "binary data omitted" in value):
        markers.append(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            markers.extend(_collect_large_markers(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            markers.extend(_collect_large_markers(child))
    return markers


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    seen = set(target)
    for value in values:
        if value not in seen:
            target.append(value)
            seen.add(value)


def _is_untrusted_tool(tool_name: str) -> bool:
    lowered = tool_name.lower()
    return lowered in _UNTRUSTED_TOOL_NAMES or lowered.startswith(_UNTRUSTED_TOOL_PREFIXES)


__all__ = [
    "SCHEMA",
    "VERSION",
    "new_completion_trace",
    "record_observation",
    "record_tool_result",
    "finalize_completion_trace",
]
