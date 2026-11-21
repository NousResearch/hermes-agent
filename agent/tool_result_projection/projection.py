"""Represent exact repeated terminal lines without discarding any unique evidence.

This is a compaction projection, never an instruction classifier. Unknown tools,
ambiguous JSON, images and non-repetitive text stay untouched. The existing prune
transaction archives the original and publishes the projection under its lease.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import groupby, pairwise
from typing import Any


@dataclass(frozen=True)
class TerminalOutput:
    output: str
    fields: dict[str, Any]


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("ambiguous duplicate JSON key")
        fields[key] = value
    return fields


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _exact_float(value: str) -> float:
    number = float(value)
    try:
        if not math.isfinite(number) or Decimal(str(number)) != Decimal(value):
            raise ValueError("JSON number cannot be preserved exactly")
    except InvalidOperation as exc:
        raise ValueError("unsupported JSON number") from exc
    return number


def _terminal_output(content: str) -> TerminalOutput | None:
    try:
        fields = json.loads(content, object_pairs_hook=_unique_fields,
                            parse_constant=_reject_constant, parse_float=_exact_float)
    except (ValueError, RecursionError):
        return None
    if not isinstance(fields, dict) or not isinstance(fields.get("output"), str):
        return None
    if type(fields.get("exit_code")) is not int:
        return None
    return TerminalOutput(fields.pop("output"), fields)


def _project(content: str) -> str | None:
    parsed = _terminal_output(content)
    if parsed is None:
        return None
    lines = parsed.output.splitlines(keepends=True)
    # Most diagnostic output is unique. Do not allocate one dictionary per line
    # when no line-run projection is possible in the first place.
    if not any(first == second for first, second in pairwise(lines)):
        return None
    runs = [{"line": line, "repeat": sum(1 for _ in group)} for line, group in groupby(lines)]
    projected = json.dumps({
        "projection": "hermes-terminal-line-runs-v1",
        "result": parsed.fields,
        "output_runs": runs,
    }, ensure_ascii=True, separators=(",", ":"))
    # The archive reference costs space too. Let the existing reclaim gate decide
    # whether this becomes a cache-breaking commit, but never propose expansion.
    return projected if len(content) - len(projected) >= 200 else None


def project_old_terminal_results(
    messages: list[dict[str, Any]], boundary: int, min_chars: int,
) -> int:
    changed = 0
    calls: dict[str, str | None] = {}
    for index in range(max(0, boundary)):
        message = messages[index]
        if message.get("role") == "assistant":
            _remember_calls(message.get("tool_calls"), calls)
        content = message.get("content")
        tool = calls.get(message.get("tool_call_id", ""))
        if message.get("role") != "tool" or tool != "terminal":
            continue
        if not isinstance(content, str) or len(content) <= min_chars:
            continue
        projected = _project(content)
        if projected is not None:
            messages[index] = {**message, "content": projected}
            changed += 1
    return changed


def _remember_calls(raw: Any, calls: dict[str, str | None]) -> None:
    """A resumed history may reuse IDs: bind to the preceding call, never a future one."""
    if not isinstance(raw, list):
        if raw is not None:
            calls.clear()
        return
    for call in raw:
        if not isinstance(call, dict):
            calls.clear()
            continue
        identifier = call.get("id")
        function = call.get("function")
        if isinstance(identifier, str):
            name = function.get("name") if isinstance(function, dict) else None
            calls[identifier] = name if isinstance(name, str) else None
