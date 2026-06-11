"""Blackbox plugin — shared data contract.

This module defines the TurnRecord dataclass and the canonical field set that
every other blackbox module (store, cost, card, commands, hooks) builds to.
Workers MUST import from here rather than redefining the shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TurnRecord:
    """One conversational turn's telemetry (or one subagent run)."""
    turn_id: str                                   # "turn_" + ULID
    parent_turn_id: Optional[str] = None           # set for subagents; None at top level
    is_subagent: bool = False
    ts_start: float = 0.0
    ts_end: float = 0.0
    profile: str = ""                              # agent identity, e.g. "aegis"
    provider: str = ""
    model: str = ""
    platform: str = ""
    chat_id: str = ""
    chat_name: str = ""
    api_calls: int = 0
    tools: List[str] = field(default_factory=list)  # ["exec","exec","read"]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    context_used: int = 0
    context_length: int = 0
    # Last-call cache split — decomposes the context WINDOW (occupancy), as
    # opposed to the cache_*_tokens above which are whole-turn billing sums.
    # These three sum to the final call's prompt_tokens == context_used.
    # Nullable in the store (old rows predate the columns → None).
    last_cache_read_tokens: Optional[int] = None
    last_cache_write_tokens: Optional[int] = None
    last_uncached_tokens: Optional[int] = None
    # Request composition of the FINAL call (char/4 fixed vs non-fixed buckets).
    # Distinct from the cache split above: this decomposes the request PAYLOAD
    # by source (system / tool schemas / history / tool results / tool args),
    # not by cache state. Nullable: old rows / blackbox-pre-composition → None.
    comp_sys_tokens: Optional[int] = None
    comp_tool_schema_tokens: Optional[int] = None
    comp_history_tokens: Optional[int] = None
    comp_history_message_count: Optional[int] = None
    comp_tool_result_tokens: Optional[int] = None
    comp_tool_arg_tokens: Optional[int] = None
    comp_tool_result_count: Optional[int] = None
    # System-prompt sub-split (identity/rules vs skill-index catalog) and the
    # per-message wire-framing estimate. comp_skills_tokens is a subset of
    # comp_sys_tokens; comp_framing_tokens is part of the non-fixed subtotal.
    comp_skills_tokens: Optional[int] = None
    comp_skills_count: Optional[int] = None
    comp_framing_tokens: Optional[int] = None
    # Per-call composition history JSON (list of breakdown dicts) for forensics.
    comp_calls_json: Optional[str] = None
    cost_usd: Optional[float] = None
    cost_status: str = "unknown"                   # estimated|included|unknown|partial
    interrupted: bool = False
    alerted: bool = False
    user_text: str = ""
    final_text: str = ""
    # Not persisted on the main row — written to the side table:
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # {name,args_preview,result_preview}

    @property
    def latency_s(self) -> float:
        return max(0.0, self.ts_end - self.ts_start)

    def to_row(self) -> Dict[str, Any]:
        """Dict for the `turns` table (excludes tool_calls side-table data)."""
        d = asdict(self)
        d.pop("tool_calls", None)
        return d


def tools_summary(tools: List[str]) -> str:
    """Render ["exec","exec","exec","read"] -> "exec×3, read" preserving order."""
    if not tools:
        return "none"
    out: List[str] = []
    counts: Dict[str, int] = {}
    order: List[str] = []
    for t in tools:
        if t not in counts:
            order.append(t)
        counts[t] = counts.get(t, 0) + 1
    for t in order:
        out.append(f"{t}×{counts[t]}" if counts[t] > 1 else t)
    return ", ".join(out)
