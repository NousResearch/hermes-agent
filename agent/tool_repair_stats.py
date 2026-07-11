"""Tool-call repair observability.

Records structured events when the repair pipeline fires, enabling
per-model and per-pattern analysis without affecting the model layer.

This is an operator-only module — no new model tools, no prompt impact.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repair patterns — the finite set of known tool-call failure modes
# ---------------------------------------------------------------------------

class RepairPattern(str, Enum):
    """Known tool-call repair patterns.

    Derived from Hermes repair pipeline + Ahmad Awais (CommandCode) analysis
    of 1T+ tokens across DeepSeek, Qwen, GLM, and other open models.
    """
    # JSON syntax repairs (message_sanitization.py)
    EMPTY_ARGS = "empty_args"                    # Empty/whitespace → {}
    NONE_LITERAL = "none_literal"                # Python None → {}
    CONTROL_CHAR_ESCAPE = "control_char_escape"  # Unescaped control chars in JSON strings
    TRAILING_COMMA = "trailing_comma"            # Trailing commas in JSON
    UNCLOSED_BRACKET = "unclosed_bracket"        # Unclosed {}/[]
    TRAILING_CONTENT = "trailing_content"        # Complete JSON + extra prose
    DOUBLE_SERIALIZE = "double_serialize"        # JSON string wrapping JSON object
    UNREPAIRABLE = "unrepairable"                # Last resort → {}

    # Type coercion repairs (model_tools.py coerce_tool_args)
    STRING_TO_INT = "string_to_int"              # "42" → 42
    STRING_TO_FLOAT = "string_to_float"          # "3.14" → 3.14
    STRING_TO_BOOL = "string_to_bool"            # "true" → True
    STRING_TO_LIST = "string_to_list"            # JSON-encoded array string → list
    STRING_TO_OBJECT = "string_to_object"        # JSON-encoded object string → dict
    BARE_STRING_WRAP = "bare_string_wrap"        # "foo" → ["foo"] when array expected
    BARE_OBJECT_WRAP = "bare_object_wrap"        # {} → [{}] when array expected
    NULL_TO_NONE = "null_to_none"                # "null" → None for nullable schemas

    # Name repairs
    NAME_FUZZY_MATCH = "name_fuzzy_match"        # Fuzzy tool name matching
    MCP_PREFIX_DROP = "mcp_prefix_drop"          # Missing mcp_ prefix

    # Schema validation (PR #61550)
    REQUIRED_MISSING = "required_missing"        # Missing required parameters

    # Catch-all
    OTHER = "other"


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class RepairEvent:
    """One recorded repair event."""
    pattern: RepairPattern
    tool_name: str
    model_name: str
    timestamp: float
    success: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Thread-safe singleton with ring buffer
# ---------------------------------------------------------------------------

class ToolRepairStats:
    """Thread-safe repair event collector with bounded ring buffer.

    Usage::

        stats = get_stats()
        stats.record(RepairPattern.BARE_STRING_WRAP, "terminal", "deepseek/deepseek-v4-pro")
        print(stats.summary())
    """

    _MAX_EVENTS: int = 10_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[RepairEvent] = []
        self._model_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # -- recording ----------------------------------------------------------

    def record(
        self,
        pattern: Any,
        tool_name: str = "?",
        model_name: str = "unknown",
        success: bool = True,
        detail: str = "",
    ) -> None:
        """Record a single repair event.  Thread-safe, bounded, cheap."""
        try:
            # Normalize string patterns to their .value for consistent counting.
            pat_value = pattern.value if hasattr(pattern, "value") else str(pattern)
            evt = RepairEvent(
                pattern=pattern,
                tool_name=tool_name,
                model_name=model_name,
                timestamp=time.time(),
                success=success,
                detail=detail,
            )
            with self._lock:
                self._events.append(evt)
                if len(self._events) > self._MAX_EVENTS:
                    self._events = self._events[-self._MAX_EVENTS:]
                    # Rebuild model counts from retained events so totals
                    # stay consistent with total() after ring-buffer trim.
                    self._model_counts = defaultdict(lambda: defaultdict(int))
                    for e in self._events:
                        v = e.pattern if isinstance(e.pattern, str) else getattr(e.pattern, "value", str(e.pattern))
                        self._model_counts[e.model_name][v] += 1
                self._model_counts[model_name][pat_value] += 1
        except Exception:
            # Observability must NEVER break the repair pipeline.
            pass

    # -- querying -----------------------------------------------------------

    def total(self) -> int:
        with self._lock:
            return len(self._events)

    def by_model(self, model_name: str) -> Dict[str, int]:
        """Return pattern→count dict for a specific model."""
        with self._lock:
            return dict(self._model_counts.get(model_name, {}))

    def all_models(self) -> Dict[str, Dict[str, int]]:
        """Return {model: {pattern: count}} for all observed models."""
        with self._lock:
            return {m: dict(p) for m, p in self._model_counts.items()}

    def top_patterns(self, n: int = 10) -> List[tuple]:
        """Return the N most frequent (pattern, count) pairs across all models."""
        totals: Dict[str, int] = defaultdict(int)
        with self._lock:
            for model_counts in self._model_counts.values():
                for pattern, count in model_counts.items():
                    totals[pattern] += count
        return sorted(totals.items(), key=lambda x: -x[1])[:n]

    def recent(self, n: int = 20) -> List[RepairEvent]:
        """Return the N most recent events."""
        with self._lock:
            return list(self._events[-n:])

    # -- summary for CLI display --------------------------------------------

    def summary(self) -> str:
        """Human-readable summary table for CLI output."""
        models = self.all_models()
        if not models:
            return "No tool-call repairs recorded this session."

        lines = [
            "Tool-Call Repair Stats",
            "=" * 60,
            "",
        ]

        # Per-model breakdown
        for model_name in sorted(models):
            patterns = models[model_name]
            total = sum(patterns.values())
            lines.append(f"  {model_name}  ({total} repairs)")
            for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
                lines.append(f"    {pattern:30s} {count:>5d}")
            lines.append("")

        # Top patterns overall
        top = self.top_patterns(5)
        if top:
            lines.append("  Top patterns (all models):")
            for pattern, count in top:
                lines.append(f"    {pattern:30s} {count:>5d}")
            lines.append("")

        grand_total = self.total()
        lines.append(f"  Total events: {grand_total}")
        return "\n".join(lines)

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()
            self._model_counts.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_stats: Optional[ToolRepairStats] = None
_stats_lock = threading.Lock()


def get_stats() -> ToolRepairStats:
    """Return the module-level singleton.  Thread-safe, lazy-init."""
    global _stats
    if _stats is None:
        with _stats_lock:
            if _stats is None:
                _stats = ToolRepairStats()
    return _stats


# ---------------------------------------------------------------------------
# Convenience function — the single call-site for all hook points
# ---------------------------------------------------------------------------

def record_repair(
    pattern: RepairPattern,
    tool_name: str = "?",
    model_name: str = "unknown",
    success: bool = True,
    detail: str = "",
) -> None:
    """Record a repair event.  Silently no-ops on any failure."""
    try:
        get_stats().record(pattern, tool_name, model_name, success, detail)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Current model context — set by dispatch_tool, read by hooks
# ---------------------------------------------------------------------------

_current_model: str = "unknown"
_model_lock = threading.Lock()


def set_current_model(model_name: str) -> None:
    """Called by dispatch_tool to make the current model available to hooks."""
    global _current_model
    with _model_lock:
        _current_model = model_name


def get_current_model() -> str:
    """Read by hook points to tag events with the model that caused them."""
    with _model_lock:
        return _current_model
