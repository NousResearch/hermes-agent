"""The self-learning cadence a turn ended on, for ``post_llm_call`` (``review_signals``).

Hermes runs its background memory/skill review on two counters: user turns since the last
memory review (``memory.nudge_interval``) and tool-calling iterations since the last skill review
(``skills.creation_nudge_interval``). A plugin that does its own after-turn work (a review-style
:meth:`PluginContext.fork_session`, say) can follow the same cadence instead of counting tool calls
itself. ``post_llm_call`` fires before the review decision, so the due flags here are the exact
values that decision uses: :func:`skill_review_due` is evaluated once per turn by the finalizer
and passed both to the hook and to the spawn; ``memory_review_due`` was decided when the turn
started (and that counter was already reset to 0 when it fired).
"""

from __future__ import annotations

from typing import Any, NoReturn


def _read_only(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise TypeError("review_signals is read-only")


class ReviewSignals(dict):
    """A read-only ``dict``: one instance reaches every ``post_llm_call`` callback, so no plugin may
    change what the next one reads. Plain ``dict`` on the wire (plugin host) and through copy/pickle."""

    __setitem__ = __delitem__ = __ior__ = _read_only  # type: ignore[assignment]
    clear = pop = popitem = setdefault = update = _read_only  # type: ignore[assignment]

    def __reduce__(self):
        return (ReviewSignals, (dict(self),))


def skill_review_due(agent: Any) -> bool:
    """The skill-review trigger: enough tool-calling iterations since the last skill review (or
    ``skill_manage`` call) and the agent can manage skills at all."""
    return (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    )


def background_review_enabled(agent: Any) -> bool:
    """Whether a due automatic review would actually be forked for this agent: not suppressed for
    it (cron), not a delegated child, and ``auxiliary.background_review.enabled`` (fail-open)."""
    if getattr(agent, "skip_background_review", False) or getattr(agent, "_delegate_depth", 0) > 0:
        return False
    from agent.background_review import load_background_review_settings

    return load_background_review_settings()[0]


def build_review_signals(agent: Any, *, memory_review_due: bool, skill_review_due: bool) -> ReviewSignals:
    """``post_llm_call``'s ``review_signals`` for the turn ``agent`` is finalizing; counters are the
    values at hook time (a due skill review resets its counter only after the hook)."""
    return ReviewSignals(
        tool_iterations_since_skill_review=int(agent._iters_since_skill),
        user_turns_since_memory_review=int(agent._turns_since_memory),
        skill_review_due=bool(skill_review_due),
        memory_review_due=bool(memory_review_due),
        background_review_enabled=background_review_enabled(agent),
        skill_nudge_interval=int(agent._skill_nudge_interval),
        memory_nudge_interval=int(agent._memory_nudge_interval),
    )
