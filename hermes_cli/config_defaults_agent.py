"""Agent review-prompt and liveness defaults."""

AGENT_REVIEW_DEFAULTS = {
    # Inline null and blank file paths keep shipped prompts; inline "" disables automatic reviews.
    "review_prompts": {"memory": None, "memory_file": "", "skill": None, "skill_file": "",
                       "combined": None, "combined_file": ""},
    # Turn liveness watchdog: a turn with no observable progress for `timeout_s` seconds is
    # logged, force-interrupted so the UI can retry, and its lease stops renewing so stale-turn
    # cleanup can reclaim the session even if the interrupt can't unwind a wedged frame.
    # timeout_s <= 0 disables; poll_s = sampling interval. Invalid values (NaN, Inf,
    # non-positive poll) warn and fall back to defaults. See agent/turn_liveness.py.
    "turn_liveness": {"timeout_s": 600.0, "poll_s": 15.0},
}
