"""Tool-loop guardrail defaults (pure-data leaf, no imports).

Sits beside ``config_defaults.py`` so that file's line count does not grow: it is already
over the 2,000-line health target, so new behaviour has to live in a sibling.
"""


def tool_loop_guardrails() -> dict:
    """Fresh ``tool_loop_guardrails`` block for ``DEFAULT_CONFIG``.

    A factory rather than a module constant so every caller gets its own dict to mutate.
    """
    return {
        "warnings_enabled": True,
        "hard_stop_enabled": False,
        # Unattended gateway/cron platforms hard-stop by default (nobody can /stop a model that
        # ignores warnings); interactive cli/tui/desktop/acp stay warning-only.
        "non_interactive_hard_stop_enabled": True,
        "warn_after": {"exact_failure": 2, "same_tool_failure": 3, "idempotent_no_progress": 2,
                       "read_only_loop": 3},
        "hard_stop_after": {
            "exact_failure": 5, "same_tool_failure": 8, "idempotent_no_progress": 5,
            "read_only_loop": 6,
        },
        # Read-only-loop guard: a run of read-only calls (file reads, inspection commands)
        # with no mutating call in between. Off by default so normal work is unaffected;
        # when on it warns after 3 and refuses the read after 6, telling the model to act.
        "read_only_loop_enabled": False,
        # Per-turn hard ceilings for runaway-prone tools; counters reset every turn, always on
        # regardless of the thresholds above. Dozens of searches/subagents in ONE turn is already
        # pathological, hence low defaults. 0 = unlimited.
        "loop_caps": {
            "max_web_searches": 50,   # web_search calls per turn
            "max_subagents": 50,      # subagents spawned per turn
        },
    }
