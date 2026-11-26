"""Output-budget stale-timeout scaling for multi-hop non-streaming API calls.

A provider that spreads a large ``extra_body.max_output_tokens`` budget over several
server-capped unary hops uses one non-streaming stale timer across every hop. Scaling that
implicit timer by the upper-bound hop count keeps healthy long-decode answers from being killed
early; streaming re-arms on every chunk and needs no scaling.
"""

from __future__ import annotations

from typing import Any

# Smallest per-request cap (unary; streamed hops get 65,536): hop counts derived from it are upper bounds.
OUTPUT_BUDGET_HOP_TOKENS = 32_768


def output_budget_hops(max_output_tokens: Any) -> int:
    """Upper-bound hop count for ``max_output_tokens``; 1 when it fits in a single request."""
    budget = max_output_tokens if isinstance(max_output_tokens, int) and not isinstance(max_output_tokens, bool) else 0
    return max(1, -(-budget // OUTPUT_BUDGET_HOP_TOKENS))


def output_budget_stale_multiplier(api_payload: Any) -> int:
    """Stale-timeout multiplier for a non-streaming request carrying ``extra_body.max_output_tokens``."""
    extra = api_payload.get("extra_body") if isinstance(api_payload, dict) else None
    return output_budget_hops(extra.get("max_output_tokens")) if isinstance(extra, dict) else 1
