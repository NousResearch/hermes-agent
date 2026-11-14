"""Pure-data factories for the autonomy-loop configuration defaults (`goals`, `loops`).

Kept out of ``config_defaults.py`` (a facade over the default config dict) so the loop defaults and
their rationale live together and the facade stays under its line budget.
"""

from __future__ import annotations


def goals_defaults() -> dict:
    """Defaults for the persistent cross-turn /goal loop.

    After each turn an aux-model judge checks whether the goal is satisfied; if not, a continuation
    prompt re-enters the session until the goal is done, the budget is exhausted, or it is paused.
    Judge failures fail OPEN — the turn budget is the backstop.
    """
    return {
        # Max continuation turns before auto-pause (/goal resume) — guards against judge false
        # negatives and unbounded spend.
        "max_turns": 20,
        # Standing instructions appended to EVERY continuation prompt (and to the first-turn kick),
        # whatever its variant. This is the place for a durable working charter — e.g. "use your
        # best judgement, record decisions where the next session will read them, and push a
        # blocked item down the road instead of stopping". Empty = the built-in wording only.
        "directives": "",
        # Full replacement for the continuation prompt itself. Must contain {goal}; blank = built-in.
        "continuation_prompt": "",
        # Variants used when /subgoal criteria or a completion contract are on the goal. Same rules:
        # {goal} plus {subgoals_block} / {contract_block} are required, blank = built-in.
        "continuation_prompt_with_subgoals": "",
        "continuation_prompt_with_contract": "",
    }


def loops_defaults() -> dict:
    """Defaults for /loop: re-run a prompt or slash command on a cadence in-session.

    A fixed interval fires on the user's clock; self-paced (no interval) starts at the floor and
    backs off exponentially while replies stop changing.
    """
    return {
        "min_interval_seconds": 30,  # smallest fixed interval; tighter cadences raised to it
        "max_ticks": 100,  # auto-pause after this many wakeups unless --times set; 0 = unlimited
        "self_paced_floor_seconds": 60,  # Self-paced cadence bounds (seconds).
        "self_paced_ceiling_seconds": 900,
    }
