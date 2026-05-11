"""Loop detection for the Hermes agent tool loop.

Detects when the agent is cycling through the same tool calls without
making progress. Two patterns are detected:

    Exact repeat — the same (tool_name, args_hash) called N times:
        read_file('/foo') × 3 → WARNING
        read_file('/foo') × 5 → inject "you're looping" signal

    Sequence cycle — the same ordered sequence repeated M times:
        [read_file, write_file] × 3 → WARNING

Usage (in _execute_tool_calls_sequential)::

    from loop_detector import LoopDetector
    # created once per run_conversation() call, passed into execute helpers
    detector = LoopDetector()

    # after each tool call:
    warning = detector.record(tool_name, function_args)
    if warning:
        hermes_log.event("loop_detected", task_id=task_id, **warning)
        # optionally inject a tool result message to break the loop

The detector is intentionally lightweight — no external deps, no DB I/O —
so it can never slow down or crash the agent loop.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any, Dict, Optional


# Thresholds
_EXACT_WARN_THRESHOLD = 3    # same tool+args N times → soft warning
_EXACT_HARD_THRESHOLD = 5    # same tool+args N times → hard signal (inject message)
_SEQ_WINDOW = 8              # rolling window length for sequence detection
_SEQ_REPEAT_THRESHOLD = 3    # same sequence repeated N times → warning


def _hash_args(args: Dict[str, Any]) -> str:
    """Stable hash of a tool args dict for identity comparison."""
    try:
        canonical = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(canonical.encode(), usedforsecurity=False).hexdigest()[:8]
    except Exception:
        return "err"


class LoopDetector:
    """Per-conversation loop detector.

    Create one instance at the start of run_conversation() and call
    record() after every tool execution.
    """

    def __init__(self) -> None:
        # Exact repeat tracking: maps (tool_name, args_hash) → count
        self._exact: Dict[tuple, int] = {}
        # Rolling window for sequence detection
        self._window: deque = deque(maxlen=_SEQ_WINDOW)

    def record(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Record a tool execution and return a warning dict if a loop is detected.

        Returns None if everything looks fine, or a dict with:
            {
                "loop_type": "exact_repeat" | "sequence_cycle",
                "tool_name": str,
                "count": int,
                "severity": "warn" | "hard",
                "inject_message": str | None,  # non-None → inject into tool results
            }
        """
        args_hash = _hash_args(args)
        key = (tool_name, args_hash)

        # --- Exact repeat check ---
        self._exact[key] = self._exact.get(key, 0) + 1
        count = self._exact[key]

        if count >= _EXACT_HARD_THRESHOLD:
            return {
                "loop_type": "exact_repeat",
                "tool_name": tool_name,
                "count": count,
                "severity": "hard",
                "inject_message": (
                    f"[Loop detected] {tool_name} has been called {count} times with "
                    f"identical arguments. You may be stuck in a loop. "
                    f"Please reconsider your approach — try a different strategy, "
                    f"check if the previous result already contains what you need, "
                    f"or summarise what you've done so far and stop."
                ),
            }

        if count >= _EXACT_WARN_THRESHOLD:
            return {
                "loop_type": "exact_repeat",
                "tool_name": tool_name,
                "count": count,
                "severity": "warn",
                "inject_message": None,  # log only, don't inject yet
            }

        # --- Sequence cycle check ---
        self._window.append(tool_name)
        window_list = list(self._window)

        # Look for sequences of length 2–4 that repeat in the window
        n = len(window_list)
        for seq_len in range(2, min(5, n // 2 + 1)):
            suffix = window_list[-seq_len:]
            repeats = 0
            pos = n - seq_len
            while pos >= seq_len:
                if window_list[pos - seq_len: pos] == suffix:
                    repeats += 1
                    pos -= seq_len
                else:
                    break
            if repeats + 1 >= _SEQ_REPEAT_THRESHOLD:
                seq_str = " → ".join(suffix)
                return {
                    "loop_type": "sequence_cycle",
                    "tool_name": tool_name,
                    "count": repeats + 1,
                    "severity": "warn",
                    "sequence": seq_str,
                    "inject_message": None,
                }

        return None

    def reset_exact(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Reset the exact-repeat counter for a specific tool+args pair.

        Useful when the agent has made progress on a task and you want
        to allow the same tool call again from a clean slate.
        """
        args_hash = _hash_args(args)
        key = (tool_name, args_hash)
        self._exact.pop(key, None)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all detected repeats (for logging at task end)."""
        repeated = {
            f"{name}({ah})": count
            for (name, ah), count in self._exact.items()
            if count >= _EXACT_WARN_THRESHOLD
        }
        return {"repeated_calls": repeated, "window": list(self._window)}
