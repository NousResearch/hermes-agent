"""Machine-readable completion metadata for one-shot CLI consumers."""

from __future__ import annotations

import json
import math
import sys


_COUNTERS = {
    "inputTokens": "session_input_tokens",
    "outputTokens": "session_output_tokens",
    "cachedInputTokens": "session_cache_read_tokens",
    "cacheWriteTokens": "session_cache_write_tokens",
    "reasoningTokens": "session_reasoning_tokens",
}


def capture_usage(agent) -> dict:
    """Capture cumulative counters at a run boundary."""
    snapshot = {key: max(0, int(getattr(agent, attr, 0) or 0))
                for key, attr in _COUNTERS.items()}
    snapshot["estimatedCostUsd"] = float(
        getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0)
    return snapshot


def usage_delta(before: dict, after: dict) -> dict:
    """Return this invocation's counters, excluding resumed-session history."""
    usage = {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
             for key in _COUNTERS}
    estimated = float(after.get("estimatedCostUsd", 0.0)) - float(
        before.get("estimatedCostUsd", 0.0))
    # A provider/catalog defect must never become a billed-looking cost.
    cost = estimated if math.isfinite(estimated) and estimated >= 0 else None
    return {"usage": usage, "costUsd": cost,
            "costStatus": "estimated" if cost is not None else "unpriced"}


def emit_machine_result(session_id: str, result: dict) -> None:
    """Emit one NDJSON-compatible event on stderr, leaving answer stdout clean."""
    event = {"type": "hermes.result", "sessionId": session_id, **result}
    print("hermes_result:" + json.dumps(event, separators=(",", ":")),
          file=sys.stderr, flush=True)
