"""Compaction policy matrix.

Each policy is a name -> spec mapping. A spec has:
  ctor:  extra kwargs for ContextCompressor(...)
  attrs: attribute overrides applied after construction (lets us pin
         tail_token_budget and other derived values without touching the
         class)
The runner constructs one compressor per policy and calls
compress(force=True) with the transcript's estimated tokens.
"""
from __future__ import annotations

from typing import Any, Dict

# Window we evaluate against (fable-5 class model).
EVAL_MODEL = "anthropic/claude-fable-5"
EVAL_WINDOW = 1_000_000

POLICIES: Dict[str, Dict[str, Any]] = {
    # Shipping behavior, untouched.
    "current": {
        "ctor": {},
        "attrs": {},
    },
    # Proposed: tail = max(10K, 0.025% ... interpreted as 2.5% of window)
    # capped hard at 25K on a 1M model. protect_last_n stays for message-count
    # floor semantics.
    "tail25k": {
        "ctor": {},
        "attrs": {"tail_token_budget": 25_000},
    },
    # Hard floor variant: minimum viable tail.
    "tail10k": {
        "ctor": {},
        "attrs": {"tail_token_budget": 10_000},
    },
    # Codex posture: nearly no tail; summary carries everything.
    "codex_style": {
        "ctor": {"protect_last_n": 3},
        "attrs": {"tail_token_budget": 2_000},
    },
    # Compaction-v2 lean mode: clamped 2.5% tail + tail tool demotion +
    # verbatim user messages in summary + session_search recovery pointers.
    "lean": {
        "ctor": {"tail_mode": "lean"},
        "attrs": {"_session_id": "eval-session"},
    },
    # fast-jev-compaction (evals/compaction/jev_arm.py): no summary at all —
    # Jev scores every tool call/result and stale ones are dropped or
    # truncated; user/assistant text stays verbatim. Plugin defaults.
    "jev": {
        "engine": "jev",
        "jev": {},
    },
    # Same, with the pinned tail widened from the plugin's 6 rows to roughly
    # lean's 25K-token tail so the two arms protect comparable recent context.
    "jev_tail40": {
        "engine": "jev",
        "jev": {"preserve_recent_messages": 40},
    },
    # Threshold lowered to ~the median keep_result Jev assigns on Hermes
    # transcripts (0.15): tests whether its ranking carries signal below the
    # plugin's 0.5 calibration point, where it drops every candidate.
    "jev_t15": {
        "engine": "jev",
        "jev": {"keep_threshold": 0.15},
    },
    # Matched-budget pair (eval-only extension): keep 60K tokens of tool
    # call+result pairs ranked by Jev's keep_result vs. ranked by recency.
    # Same retained size, so the recall gap is Jev's judgment alone.
    "jev_top60k": {
        "engine": "jev",
        "jev": {"select": "jev", "result_budget_tokens": 60_000},
    },
    "recent_top60k": {
        "engine": "jev",
        "jev": {"select": "recency", "result_budget_tokens": 60_000},
    },
}


def apply_policy(compressor, spec: Dict[str, Any]):
    for key, value in (spec.get("attrs") or {}).items():
        setattr(compressor, key, value)
    return compressor
