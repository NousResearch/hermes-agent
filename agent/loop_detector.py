"""Context-aware repeated tool-call loop detector."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_LOW_INFORMATION_MARKERS = {"", "no results", "not found", "none", "[]", "{}"}


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    args_key: str
    tokens: frozenset[str]
    low_information: bool


def _tokens(value: Any) -> frozenset[str]:
    if isinstance(value, dict):
        text = " ".join(str(v) for v in value.values())
    else:
        text = str(value)
    normalized = []
    for token in _TOKEN_RE.findall(text):
        token = token.lower()
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if len(token) > 1:
            normalized.append(token)
    return frozenset(normalized)


def _args_key(arguments: Any) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(arguments)


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _low_info(result_summary: str | None) -> bool:
    if result_summary is None:
        return False
    text = str(result_summary).strip().lower()
    return text in _LOW_INFORMATION_MARKERS or len(text) < 12


class ToolLoopDetector:
    """Detect repeated exact/similar calls and recommend a strategy switch."""

    def __init__(self, *, max_window: int = 10, repeat_threshold: int = 3, similarity_threshold: float = 0.85) -> None:
        self.max_window = max(1, int(max_window))
        self.repeat_threshold = max(2, int(repeat_threshold))
        self.similarity_threshold = float(similarity_threshold)
        self._records: deque[ToolCallRecord] = deque(maxlen=self.max_window)

    def record(self, tool_name: str, arguments: Any, *, result_summary: str | None = None) -> dict[str, Any] | None:
        record = ToolCallRecord(
            tool_name=str(tool_name),
            args_key=_args_key(arguments),
            tokens=_tokens(arguments),
            low_information=_low_info(result_summary),
        )
        self._records.append(record)
        recent = list(self._records)[-self.repeat_threshold:]
        if len(recent) < self.repeat_threshold:
            return None
        same_tool = all(r.tool_name == record.tool_name for r in recent)
        if not same_tool:
            return None
        if all(r.args_key == record.args_key for r in recent):
            return self._warning("repeated_tool_call", record.tool_name, recent)
        similarities = [_similarity(recent[i - 1].tokens, recent[i].tokens) for i in range(1, len(recent))]
        if all(s >= self.similarity_threshold for s in similarities) and any(r.low_information for r in recent):
            return self._warning("low_information_loop", record.tool_name, recent)
        return None

    def _warning(self, kind: str, tool_name: str, recent: list[ToolCallRecord]) -> dict[str, Any]:
        return {
            "kind": kind,
            "tool_name": tool_name,
            "repeat_count": len(recent),
            "recommendation": "Stop repeating this tool call; switch strategy, inspect a primary source, or ask for missing context.",
        }
