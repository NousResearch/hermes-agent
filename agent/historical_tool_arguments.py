"""Provenance markers for historical tool arguments omitted by compression."""

from __future__ import annotations

import json
from typing import Any

_HISTORY_OMITTED_KEY = "_hermes_history_omitted"
_HISTORY_OMITTED_REASON = "context_compression"
_MAX_MARKER_SCAN_NODES = 10_000
NON_REPLAYABLE_HISTORY_MESSAGE = (
    "Historical tool arguments omitted during context compression cannot be replayed. "
    "Read current state and construct fresh arguments before calling this tool."
)
TOOL_ARGUMENTS_TOO_COMPLEX_MESSAGE = (
    "Tool arguments are too deeply nested to inspect safely; tool was not executed."
)


class ToolArgumentTraversalLimit(RuntimeError):
    """Raised when marker inspection exceeds its bounded container budget."""


def omit_historical_tool_arguments(raw_arguments: Any) -> str:
    """Return bounded valid JSON that cannot be mistaken for executable arguments."""
    original_chars = len(raw_arguments) if isinstance(raw_arguments, str) else 0
    return json.dumps(
        {
            _HISTORY_OMITTED_KEY: {
                "non_replayable": True,
                "reason": _HISTORY_OMITTED_REASON,
                "original_chars": original_chars,
            }
        },
        separators=(",", ":"),
    )


def contains_non_replayable_history_args(value: Any) -> bool:
    """Detect the exact reserved marker with bounded, lazy traversal."""
    frames = [iter((value,))]
    seen: set[int] = set()
    visited_containers = 0
    while frames:
        try:
            current = next(frames[-1])
        except StopIteration:
            frames.pop()
            continue

        if isinstance(current, dict):
            marker = current.get(_HISTORY_OMITTED_KEY)
            if (
                isinstance(marker, dict)
                and len(marker) == 3
                and marker.get("non_replayable") is True
                and marker.get("reason") == _HISTORY_OMITTED_REASON
                and type(marker.get("original_chars")) is int
                and marker["original_chars"] >= 0
            ):
                return True
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            visited_containers += 1
            if visited_containers > _MAX_MARKER_SCAN_NODES:
                raise ToolArgumentTraversalLimit
            frames.append(iter(current.values()))
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            visited_containers += 1
            if visited_containers > _MAX_MARKER_SCAN_NODES:
                raise ToolArgumentTraversalLimit
            frames.append(iter(current))
    return False
