"""Occurrence-aware metadata pairing for context compression.

Provider tool-call IDs are correlation aliases, not globally unique execution
identity. This helper preserves occurrence multiplicity when a provider reuses
an ID in a later call, without changing or repairing the transcript itself.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from agent.message_sanitization import coalesce_tool_call_id


def _normalize_ref(value: Any) -> str:
    raw = str(value or "").strip()
    return raw.split("|", 1)[0].strip() if raw else ""


def _raw_ref(tool_call: Any, field: str) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get(field) or "").strip()
    return str(getattr(tool_call, field, None) or "").strip()


def _tool_call_aliases(tool_call: Any) -> tuple[str, ...]:
    """Return usable aliases for one logical assistant tool-call occurrence."""
    refs: list[str] = []
    for raw in (
        coalesce_tool_call_id(tool_call),
        _raw_ref(tool_call, "call_id"),
        _raw_ref(tool_call, "id"),
    ):
        normalized = _normalize_ref(raw)
        if normalized and normalized not in refs:
            refs.append(normalized)
        raw = str(raw or "").strip()
        if raw and "|" not in raw and raw not in refs:
            refs.append(raw)
    return tuple(refs)


def _tool_metadata(tool_call: Any) -> tuple[str, str]:
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        return str(fn.get("name") or "unknown"), str(fn.get("arguments") or "")
    fn = getattr(tool_call, "function", None)
    if fn is None:
        return "unknown", ""
    return (
        str(getattr(fn, "name", None) or "unknown"),
        str(getattr(fn, "arguments", None) or ""),
    )


def tool_result_metadata_by_index(
    messages: Iterable[dict[str, Any]],
) -> dict[int, tuple[str, str]]:
    """Map each unambiguous tool-result position to its call metadata.

    Results consume one outstanding logical occurrence. A raw provider ID may
    therefore be reused after its earlier occurrence has completed. If one
    result alias can address multiple simultaneous live occurrences, no mapping
    is returned for that result rather than guessing provenance.
    """
    pending: dict[str, list[int]] = defaultdict(list)
    metadata: dict[int, tuple[str, str]] = {}
    consumed: set[int] = set()
    resolved: dict[int, tuple[str, str]] = {}
    occurrence_id = 0

    for message_index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue

        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                occurrence_id += 1
                metadata[occurrence_id] = _tool_metadata(tool_call)
                for ref in _tool_call_aliases(tool_call):
                    pending[ref].append(occurrence_id)
            continue

        if msg.get("role") != "tool":
            continue

        raw_result_ref = str(msg.get("tool_call_id") or "").strip()
        result_refs = {_normalize_ref(raw_result_ref)}
        if raw_result_ref and "|" not in raw_result_ref:
            result_refs.add(raw_result_ref)
        result_refs.discard("")

        candidates = {
            candidate
            for ref in result_refs
            for candidate in pending.get(ref, ())
            if candidate not in consumed
        }
        if len(candidates) != 1:
            continue

        matched = candidates.pop()
        consumed.add(matched)
        resolved[message_index] = metadata[matched]

    return resolved

