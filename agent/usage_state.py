"""Initial session accounting state and usage anchors."""
from collections import deque
from typing import Any

USAGE_STATE: dict[str, Any] = {
    "_user_turn_count": 0,
    "_is_user_initiated_turn": False,  # Copilot x-initiator: first call of a user turn = "user"
    # Usage anchors (agent/usage_anchor.py): last response's exact usage + transcript
    # snapshot; invalidated on compaction/session switch so stale anchors never suppress compression.
    "_usage_anchor": None,
    "_turn_base_usage_anchor": None,
    "_request_pressure_anchored": False,  # whether the last pressure figure came from the anchor
    # Cumulative token usage for the session
    "session_prompt_tokens": 0,
    "session_completion_tokens": 0,
    "session_total_tokens": 0,
    "session_api_calls": 0,
    "session_input_tokens": 0,
    "session_output_tokens": 0,
    "session_cache_read_tokens": 0,
    "session_cache_write_tokens": 0,
    "session_reasoning_tokens": 0,
    "session_estimated_cost_usd": 0.0,
    "session_cost_status": "unknown",
    "session_cost_source": "none",
    # Status-bar latency/velocity history (last 10 calls), shared by loop + codex_runtime.
    "_api_latency_history": lambda: deque(maxlen=10),
    "_api_output_history": lambda: deque(maxlen=10),
}

