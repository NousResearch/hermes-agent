"""Per-session usage/accounting state and the tool-result side of agent construction.

Split from ``agent/agent_init.py``: the subdirectory hint tracker and the profile's
exact-value redaction registration both decide what a tool result may carry to the model, and the
facade is over its line cap.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from agent.redact import register_env_redaction_values
from agent.subdirectory_hints import SubdirectoryHintTracker


def _init_usage_state(agent):
    from agent.agent_init import _set_defaults
    from agent.runtime_cwd import scope_terminal_cwd
    # The profile's own .env secrets become exact-value redaction entries before anything reads a
    # file for this agent (context files at prompt build, tool results later). Memory only.
    register_env_redaction_values()
    # Prefer the session's explicitly adopted workspace (a Desktop session created under the
    # spawn-time home pin records none; a picked/adopted one does — agent.session_cwd is set
    # at build time and on every workspace move). TERMINAL_CWD is the launch fallback.
    working_dir = getattr(agent, "session_cwd", None) or scope_terminal_cwd() or None
    agent._subdirectory_hints = SubdirectoryHintTracker(
        working_dir=working_dir, enabled=not agent.skip_context_files)
    _set_defaults(agent, _USAGE_STATE)


# Per-session usage accounting.
_USAGE_STATE: dict[str, Any] = {
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
