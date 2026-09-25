"""Verification-loop helpers for the ``pre_verify`` round-end gate.

After code edits the loop fires ``pre_verify`` (directives resolved by
:func:`hermes_cli.plugins.get_pre_verify_continue_message`). The shipped coding
guidance rides on the evidence-based verification-stop nudge rather than a second
default stop gate, so default token cost stays tied to the "missing verification
evidence" decision while ``pre_verify`` remains free for user/plugin policy.
"""

from __future__ import annotations

from typing import Any, Optional

from utils import is_truthy_value

DEFAULT_MAX_VERIFY_NUDGES = 3

# Attribute carrying the turn's pending enforced verdict (see
# :func:`apply_pre_verify_verdict`). Per-turn; cleared by ``turn_context``.
PRE_VERIFY_VERDICT_ATTR = "_pre_verify_final_verdict"

# Appended to the verification-stop nudge when code lacks fresh evidence. Mirrors
# the user-facing "clean your work" workflow without adding its own model turn.
CODING_VERIFY_GUIDANCE = (
    "[Coding] Before you run tests/linters or call this done: if this is "
    "creative UI/visual work, hold off on tests and linters until the user says "
    "they like the result or you're about to commit. And before every commit, "
    "clean your work: keep it KISS/DRY, match the surrounding code style, and be "
    "elitist, shorthand, clever, concise, efficient, and elegant."
)


def max_verify_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive ``pre_verify`` continue directives per turn (>= 0)."""
    try:
        return max(0, int(_agent_cfg(config).get("max_verify_nudges")))
    except (TypeError, ValueError):
        return DEFAULT_MAX_VERIFY_NUDGES


def pre_verify_on_no_edit_turns(config: Optional[dict[str, Any]] = None) -> bool:
    """Whether ``pre_verify`` may also fire on turns that edited no files.

    Default **false**: the gate keeps its shipped shape (edited-code turns
    only), so no registered hook sees a new call and no turn gains a new
    continuation path unless the user opts in with
    ``agent.pre_verify_on_no_edit_turns: true`` in ``config.yaml``.

    When enabled, the same call site fires with ``changed_paths=[]`` — same
    payload shape, same ``agent.max_verify_nudges`` bound, same ``has_hook``
    guard. It exists for policy hooks whose subject is not the diff (board /
    backlog / review-state reconciliation), which previously had no supported
    way to continue a no-edit turn.
    """
    return is_truthy_value(
        _agent_cfg(config).get("pre_verify_on_no_edit_turns", False), default=False
    )


def record_pre_verify_verdict(agent: Any, verdict: Optional[str]) -> None:
    """Record (or clear) the verdict a ``pre_verify`` hook demands be delivered.

    Extension of the existing ``pre_verify`` contract, not a new hook: a hook
    that returns ``final_verdict`` (or ``{"action": "final", ...}``) states the
    text the turn may not end without. Every ``pre_verify`` evaluation replaces
    the pending value, so the latest evaluation governs and a verdict cannot
    outlive the condition that produced it. No model call, no timer.
    """
    setattr(agent, PRE_VERIFY_VERDICT_ATTR, str(verdict or "").strip())


def apply_pre_verify_verdict(
    agent: Any, final_response: Any, *, interrupted: bool = False
) -> Any:
    """Enforce the pending ``pre_verify`` verdict on the DELIVERED answer.

    Called by :func:`agent.turn_finalizer.finalize_turn` **after** the
    ``transform_llm_output`` hooks, so the outcome does not depend on plugin
    ordering: a transform that rewrites (or silently replaces) the answer
    cannot drop the verdict, and a model that ignores the last continuation's
    instruction cannot end quiet. It also covers the budget-exhaustion exits,
    which no continuation message can reach.

    Exactly-once and idempotent: the pending value is consumed here, and a
    response that already carries it verbatim is left untouched (so a plugin
    that also emits the text through a transform does not duplicate it).
    An interrupt (user stop) wins — nothing is appended to a stopped turn.
    Default-off: with no hook returning a verdict there is no pending value and
    this is a no-op, byte-for-byte.
    """
    raw = getattr(agent, PRE_VERIFY_VERDICT_ATTR, "")
    # Strictly a string: an agent double / mock attribute that auto-creates a
    # non-string must never be able to inject text into a delivered answer.
    verdict = raw.strip() if isinstance(raw, str) else ""
    try:
        setattr(agent, PRE_VERIFY_VERDICT_ATTR, "")
    except Exception:
        pass
    if not verdict or interrupted:
        return final_response
    if not isinstance(final_response, str) or not final_response.strip():
        return verdict
    if verdict in final_response:
        return final_response
    # Prepended: the fail-explicit statement must be the first thing read, and
    # must survive a long or hostile draft that buries or contradicts it.
    return verdict + "\n\n" + final_response


def coding_verify_guidance(config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Return the optional guidance appended to verification-stop nudges."""
    if not is_truthy_value(_agent_cfg(config).get("verify_guidance", True), default=True):
        return None
    return CODING_VERIFY_GUIDANCE


def _agent_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


__all__ = [
    "CODING_VERIFY_GUIDANCE",
    "DEFAULT_MAX_VERIFY_NUDGES",
    "PRE_VERIFY_VERDICT_ATTR",
    "apply_pre_verify_verdict",
    "coding_verify_guidance",
    "max_verify_nudges",
    "record_pre_verify_verdict",
    "pre_verify_on_no_edit_turns",
]
