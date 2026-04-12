"""Detect degenerate tool-call loops via three strategies.

Detectors:
    generic_repeat   — same (tool, canonical_args) N consecutive times
    poll_no_progress — same tool, identical result hashes N consecutive times
    ping_pong        — alternating between exactly 2 (tool, args) states

Each call to record() returns a LoopVerdict with severity (none/warning/critical),
the detector that fired, the streak length, and optionally the intended tool name
extracted from the model's reasoning content.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LoopVerdict:
    severity: str  # "none", "warning", "critical"
    detector: Optional[str] = None
    streak: int = 0
    intended_tool: Optional[str] = None


def _canonical_key(tool_name: str, args: dict) -> str:
    """Stable hash key for (tool_name, args) regardless of key order."""
    normalized = json.dumps(args, sort_keys=True, default=str)
    return f"{tool_name}:{normalized}"


def _result_hash(result: str) -> str:
    """Short hash of a tool result for comparison."""
    return hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class _Entry:
    tool_name: str
    call_key: str
    result_hash: str


class ToolLoopDetector:
    def __init__(
        self,
        warning_threshold: int = 3,
        critical_threshold: int = 5,
        window_size: int = 30,
        valid_tool_names: set[str] | None = None,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self._valid_tool_names = valid_tool_names or set()
        self._history: deque[_Entry] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def record(
        self,
        tool_name: str,
        args: dict,
        result: str = "",
        reasoning: str | None = None,
    ) -> LoopVerdict:
        call_key = _canonical_key(tool_name, args)
        rhash = _result_hash(result)
        entry = _Entry(tool_name=tool_name, call_key=call_key, result_hash=rhash)
        self._history.append(entry)

        intended = self._extract_intended_tool(reasoning, tool_name) if reasoning else None

        verdict = self._check_generic_repeat(call_key)
        if verdict.severity != "none":
            return LoopVerdict(
                severity=verdict.severity,
                detector="generic_repeat",
                streak=verdict.streak,
                intended_tool=intended,
            )

        verdict = self._check_poll_no_progress(tool_name, rhash)
        if verdict.severity != "none":
            return LoopVerdict(
                severity=verdict.severity,
                detector="poll_no_progress",
                streak=verdict.streak,
                intended_tool=intended,
            )

        verdict = self._check_ping_pong()
        if verdict.severity != "none":
            return LoopVerdict(
                severity=verdict.severity,
                detector="ping_pong",
                streak=verdict.streak,
                intended_tool=intended,
            )

        return LoopVerdict(severity="none")

    def _check_generic_repeat(self, call_key: str) -> LoopVerdict:
        """Count consecutive identical (tool, args) from the tail of history."""
        streak = 0
        for entry in reversed(self._history):
            if entry.call_key == call_key:
                streak += 1
            else:
                break
        return self._severity(streak)

    def _check_poll_no_progress(self, tool_name: str, rhash: str) -> LoopVerdict:
        """Count consecutive same-tool calls with identical result hashes."""
        streak = 0
        for entry in reversed(self._history):
            if entry.tool_name == tool_name and entry.result_hash == rhash:
                streak += 1
            else:
                break
        return self._severity(streak)

    def _check_ping_pong(self) -> LoopVerdict:
        """Detect A-B-A-B-A-B alternation in the last 6+ entries."""
        h = list(self._history)
        if len(h) < 6:
            return LoopVerdict(severity="none")
        tail = h[-6:]
        keys = [e.call_key for e in tail]
        a, b = keys[-2], keys[-1]
        if a == b:
            return LoopVerdict(severity="none")
        expected = [a, b] * 3
        if keys == expected:
            pairs = len(tail) // 2
            return self._severity(pairs)
        return LoopVerdict(severity="none")

    def _severity(self, streak: int) -> LoopVerdict:
        if streak >= self.critical_threshold:
            return LoopVerdict(severity="critical", streak=streak)
        if streak >= self.warning_threshold:
            return LoopVerdict(severity="warning", streak=streak)
        return LoopVerdict(severity="none", streak=streak)

    def _extract_intended_tool(self, reasoning: str, actual_tool: str) -> str | None:
        """Search reasoning text for a tool name different from the one actually called.

        Looks for mentions of valid tool names in the reasoning content and
        returns the first one that differs from actual_tool.
        """
        if not reasoning or not self._valid_tool_names:
            return None
        for name in self._valid_tool_names:
            if name == actual_tool:
                continue
            if name in reasoning:
                return name
        return None
