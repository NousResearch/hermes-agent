"""Occurrence-aware metadata pairing for context compression.

Provider tool-call IDs are correlation aliases, not globally unique execution
identity. This helper preserves occurrence multiplicity when a provider reuses
an ID in a later call, without changing or repairing the transcript itself.

Responses composite references are conventionally stored as
``call_id|response_item_id`` (for example, ``call_abc|fc_123``), with the
stable provider call ID first. Matching intentionally preserves the existing
behavior here: composite result references use their leading segment, while a
bare ``call_id`` or ``id`` remains a usable alias. Therefore a swapped
``fc_123|call_abc`` reference can still resolve when ``fc_123`` is itself a
known alias; this helper does not rewrite or canonicalize that producer shape.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable

from agent.message_sanitization import coalesce_tool_call_id


logger = logging.getLogger(__name__)
_DEBUG_REF_MAX_CHARS = 96


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
            if logger.isEnabledFor(logging.DEBUG):
                bounded_refs = tuple(
                    sorted(ref[:_DEBUG_REF_MAX_CHARS] for ref in result_refs)
                )
                logger.debug(
                    "Tool-result metadata unresolved at message index %d: "
                    "refs=%r candidates=%d",
                    message_index,
                    bounded_refs,
                    len(candidates),
                )
            continue

        matched = candidates.pop()
        consumed.add(matched)
        resolved[message_index] = metadata[matched]

    return resolved
