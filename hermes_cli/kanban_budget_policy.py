"""Deterministic, inspectable v1 budgets for Kanban worker tasks.

This is deliberately a static policy: historical outcomes may be analysed by
operators, but they never mutate the enforcement selected for an existing run.
"""

from __future__ import annotations

from typing import Any


POLICY_VERSION = "kanban-archetype-v1"
_MIN_OVERRIDE_TURNS = 8
_MAX_OVERRIDE_TURNS = 120

_ARCHETYPES = (
    ("research", ("research", "investigate", "analyze", "analyse", "explore"),
     {"plan": 8, "implement": 12, "verify": 6, "handoff": 2}),
    ("implementation", ("implement", "build", "fix", "feature", "refactor", "code"),
     {"plan": 6, "implement": 24, "verify": 8, "handoff": 2}),
    ("verification", ("verify", "test", "validate", "review", "reproduce"),
     {"plan": 6, "implement": 10, "verify": 12, "handoff": 2}),
    ("orchestration", ("orchestrate", "coordinate", "dispatch", "migrate", "rollout"),
     {"plan": 8, "implement": 16, "verify": 8, "handoff": 2}),
)
_UNKNOWN_PHASE_TURNS = {"plan": 6, "implement": 10, "verify": 6, "handoff": 2}


def classify_task_archetype(title: object, body: object = None) -> str:
    """Return the first documented archetype signalled by task text.

    Title signals take precedence over body signals, and the tuple ordering is
    stable, making the result independent of dictionary iteration or models.
    """
    for text in (str(title or "").casefold(), str(body or "").casefold()):
        for archetype, signals, _phases in _ARCHETYPES:
            if any(signal in text for signal in signals):
                return archetype
    return "unknown"


def _scaled_phases(phases: dict[str, int], total: int) -> dict[str, int]:
    """Scale policy phases while always reserving verification and handoff."""
    baseline = sum(phases.values())
    fixed = {"handoff": max(2, round(total * phases["handoff"] / baseline)),
             "verify": max(2, round(total * phases["verify"] / baseline))}
    remaining = total - fixed["handoff"] - fixed["verify"]
    plan_weight = phases["plan"]
    implement_weight = phases["implement"]
    plan = max(1, round(remaining * plan_weight / (plan_weight + implement_weight)))
    return {"plan": plan, "implement": remaining - plan, **fixed}


def resolve_task_budget(task: Any) -> dict[str, Any]:
    """Resolve a serializable budget without changing an already stored task.

    ``goal_max_turns`` is the existing per-card explicit operator cap.  Its
    bounded value overrides the recommendation for both goal-loop and regular
    workers; invalid values leave the conservative policy in charge.
    """
    archetype = classify_task_archetype(getattr(task, "title", None), getattr(task, "body", None))
    phases = next((p for name, _signals, p in _ARCHETYPES if name == archetype), _UNKNOWN_PHASE_TURNS)
    fallback_reason = "no_archetype_signal" if archetype == "unknown" else None
    override = getattr(task, "goal_max_turns", None)
    override_turns = None
    if isinstance(override, int) and not isinstance(override, bool):
        if _MIN_OVERRIDE_TURNS <= override <= _MAX_OVERRIDE_TURNS:
            override_turns = override
        elif override is not None:
            fallback_reason = "invalid_override"
    total = override_turns or sum(phases.values())
    resolved_phases = _scaled_phases(phases, total) if override_turns else dict(phases)
    return {
        "policy_version": POLICY_VERSION,
        "archetype": archetype,
        "max_turns": total,
        "phase_turns": resolved_phases,
        "override_turns": override_turns,
        "fallback_reason": fallback_reason,
    }
