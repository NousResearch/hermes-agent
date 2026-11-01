"""Explicit model/endpoint-scoped compatibility for content-only tool calls (#5867)."""

from __future__ import annotations

import json
from typing import Any

from agent.transports.types import ToolCall

_CONTENT_TOOL_CALL_MAX_BYTES = 64 * 1024
_CONTENT_TOOL_CALL_MAX_DEPTH = 20
_CONTENT_TOOL_CALL_MAX_NODES = 1_000
_CONTENT_TOOL_CALL_MAX_CALLS = 16


def _content_tool_call_fallback_enabled(agent: Any) -> bool:
    """Check the explicit model/endpoint scope captured during agent init."""
    configured_scope = getattr(agent, "_content_tool_call_fallback_scope", None)
    current_scope = (
        getattr(agent, "model", None),
        getattr(agent, "base_url", None),
        getattr(agent, "api_mode", None),
    )
    return configured_scope is not None and configured_scope == current_scope


def _reject_non_finite_json(value: str) -> Any:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_value_within_limits(value: Any) -> bool:
    """Bound traversal before canonicalizing untrusted model output."""
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _CONTENT_TOOL_CALL_MAX_NODES or depth > _CONTENT_TOOL_CALL_MAX_DEPTH:
            return False
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return True


def _extract_content_tool_calls(agent: Any, content: Any) -> list[ToolCall] | None:
    """Promote JSON-only tool calls emitted in assistant content.

    Some Ollama-served models omit the structured ``tool_calls`` field and
    return ``{"name": ..., "arguments": ...}`` as content instead.  Only a
    complete object/list whose every name resolves through the normal repair
    path is promoted; otherwise the content remains ordinary assistant text.
    """
    if (
        getattr(agent, "api_mode", None) != "chat_completions"
        or not _content_tool_call_fallback_enabled(agent)
        or not getattr(agent, "tools", None)
        or not isinstance(content, str)
    ):
        return None
    if not content or len(content) > _CONTENT_TOOL_CALL_MAX_BYTES:
        return None
    try:
        content_bytes = content.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(content_bytes) > _CONTENT_TOOL_CALL_MAX_BYTES:
        return None
    text = content.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json,
        )
    except (TypeError, ValueError, RecursionError):
        return None

    candidates = [parsed] if isinstance(parsed, dict) else parsed
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) > _CONTENT_TOOL_CALL_MAX_CALLS
    ):
        return None

    calls: list[ToolCall] = []
    valid_names = getattr(agent, "valid_tool_names", set()) or set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != {"name", "arguments"}:
            return None
        raw_name = candidate.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            return None
        name = (
            raw_name if raw_name in valid_names else agent._repair_tool_call(raw_name)
        )
        if not name or name not in valid_names:
            return None

        raw_arguments = candidate.get("arguments")
        if not isinstance(raw_arguments, dict) or not _json_value_within_limits(
            raw_arguments
        ):
            return None
        try:
            arguments = json.dumps(
                raw_arguments,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            arguments.encode("utf-8")
        except (TypeError, ValueError, RecursionError, UnicodeEncodeError):
            return None
        calls.append(
            ToolCall(
                id=agent._deterministic_call_id(name, arguments, index),
                name=name,
                arguments=arguments,
            )
        )
    return calls
