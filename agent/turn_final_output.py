"""Terminal disposition for the ``llm_final_output_commit`` final-output gate.

The gate decides whether a turn's final assistant candidate may become durable
conversation state, eligible for next-turn replay, and delivered to the user.
Exactly one function owns that decision — :func:`gate_final_output` — and it is
called from the two points that can mint a final candidate:

* ``agent/turn_final_response.py::finish_text_response`` — normal text
  completion, before ``append_message`` + the durable flush (the first commit);
* ``agent/turn_finalizer.py::finalize_turn`` — the budget-fallback summary and
  stream recovery, which create their candidate *after* the loop has ended and
  therefore never reach ``finish_text_response``.

The verdict is recorded on the agent for the turn, and every later owner reads
it instead of evaluating the chain again:

* ``allow`` is recorded together with the committed candidate, so an identical
  candidate is never gated twice in one turn (one-evaluation), while a *new*
  candidate minted later (stream recovery over a blank completion) is still
  gated;
* ``drop`` and ``refused`` are terminal for the whole turn: no later call
  re-evaluates them, so no finalizer path can reverse a DROP into a persisted,
  replayable, deliverable assistant row, and no second evaluation can flip a
  refusal into an allow.

``refused`` is recorded *before* the fail-closed
:class:`~hermes_cli.middleware.LLMStreamMiddlewareRefusal` is re-raised, so the
loop's outer error owner and the finalizer both see the same terminal state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

VERDICT_ALLOW = "allow"
VERDICT_DROP = "drop"
VERDICT_REFUSED = "refused"
#: Verdicts that end the turn's output path: honored by every later owner.
TERMINAL_VERDICTS = frozenset({VERDICT_DROP, VERDICT_REFUSED})

#: ``turn_exit_reason`` carried out of the loop alongside the recorded verdict,
#: so the disposition survives even if the per-turn state is lost.
EXIT_REASON_DROPPED = "final_output_dropped"
EXIT_REASON_REFUSED = "final_output_refused"

#: User-facing copy for a fail-closed refusal (mirrors the API-path refusal in
#: ``agent/turn_api_error.py``): what happened, in plain words, no mechanism.
FINAL_OUTPUT_REFUSAL_COPY = (
    "Turn blocked: fail-closed final-output middleware refused the response."
)


def _turn_key(agent: Any, turn_id: Optional[str] = None) -> str:
    if turn_id is not None:
        return str(turn_id or "")
    return str(getattr(agent, "_current_turn_id", "") or "")


def reset_final_output_disposition(agent: Any) -> None:
    """Clear the recorded verdict at turn start.

    Agents are reused across turns (the gateway caches them), so a stale verdict
    must never leak into the next turn. The turn-id check in
    :func:`final_output_disposition` is the second line of defence for callers
    that never reset.
    """
    agent._final_output_disposition = None


def final_output_disposition(agent: Any, turn_id: Optional[str] = None) -> Optional[str]:
    """Recorded verdict for this turn, or ``None`` when no gate has run yet."""
    state = getattr(agent, "_final_output_disposition", None)
    if not isinstance(state, dict):
        return None
    if state.get("turn_id") != _turn_key(agent, turn_id):
        return None  # verdict belongs to an earlier turn
    verdict = state.get("verdict")
    return verdict if verdict in (VERDICT_ALLOW, VERDICT_DROP, VERDICT_REFUSED) else None


def is_final_output_terminal(agent: Any, turn_id: Optional[str] = None) -> bool:
    """True when this turn's output path already ended in DROP or refusal."""
    return final_output_disposition(agent, turn_id) in TERMINAL_VERDICTS


def disposition_from_exit_reason(exit_reason: Any) -> Optional[str]:
    """Recover the disposition from the loop's exit reason (belt and braces)."""
    reason = str(exit_reason or "")
    if reason == EXIT_REASON_DROPPED:
        return VERDICT_DROP
    if reason == EXIT_REASON_REFUSED:
        return VERDICT_REFUSED
    return None


def _record(agent: Any, *, turn_id: str, verdict: str, content: Any) -> None:
    agent._final_output_disposition = {
        "turn_id": turn_id,
        "verdict": verdict,
        "content": content,
    }


def gate_final_output(
    agent: Any,
    *,
    candidate: Dict[str, Any],
    context: Dict[str, Any],
    turn_id: Optional[str] = None,
) -> str:
    """Evaluate ``llm_final_output_commit`` for ``candidate`` and record the verdict.

    Returns ``"allow"``, ``"drop"`` or ``"refused"``. A fail-closed callback
    failure records ``"refused"`` and then re-raises
    :class:`~hermes_cli.middleware.LLMStreamMiddlewareRefusal` so the caller's
    owner (the outer loop error handler, or the finalizer's settle path) decides
    how the attempt ends.

    Evaluation rules:

    * a terminal verdict is returned without touching the chain again — a DROP
      or refusal can never be reversed by a later owner;
    * ``allow`` is re-used for an identical candidate in the same turn, so the
      normal commit point plus the finalizer evaluate a committed candidate once;
    * a *different* candidate in the same turn is evaluated on its own merits.
    """
    from hermes_cli.middleware import (
        LLMStreamMiddlewareRefusal,
        run_llm_final_output_commit_middleware,
    )

    turn_id = _turn_key(agent, turn_id)
    content = candidate.get("content") if isinstance(candidate, dict) else None

    state = getattr(agent, "_final_output_disposition", None)
    if isinstance(state, dict) and state.get("turn_id") == turn_id:
        verdict = state.get("verdict")
        if verdict in TERMINAL_VERDICTS:
            return verdict
        if verdict == VERDICT_ALLOW and state.get("content") == content:
            return VERDICT_ALLOW

    try:
        verdict = run_llm_final_output_commit_middleware(candidate, context)
    except LLMStreamMiddlewareRefusal:
        _record(agent, turn_id=turn_id, verdict=VERDICT_REFUSED, content=content)
        raise
    verdict = VERDICT_DROP if str(verdict).strip().lower() != VERDICT_ALLOW else VERDICT_ALLOW
    _record(agent, turn_id=turn_id, verdict=verdict, content=content)
    return verdict
