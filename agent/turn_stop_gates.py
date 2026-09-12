"""Text-response stop gates for the conversation turn loop.

When the model stops with a text answer, three gates may instead append the answer as an
interim row plus a synthetic user-role nudge and continue the turn: verify-on-stop (#65919),
the ``pre_verify`` plugin hook after code edits, and the kanban worker terminal-tool guard.
Each keeps the candidate answer as a budget-exhaustion fallback
(``pending_verification_response``) and clears ``final_response`` so the finalizer can tell
this gate from error exits (#61631). Nothing here imports ``agent.conversation_loop`` at
module level (cycle).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")

_MAX_PRE_DELIVERY_REPAIR_ATTEMPTS = 4
_USE_FINAL_RESPONSE_AS_FALLBACK = object()
_PRE_DELIVERY_REASON_MAX_CHARS = 2_000


def _build_pre_delivery_repair_directive(*, attempt: int, reason: str) -> str:
    """Build the bounded instruction shared by every pre-delivery repair path."""
    repair_contract = {
        "gate": "pre_delivery",
        "action": "repair",
        "attempt": attempt,
        "mode": (
            "standard" if attempt <= _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS else "escalated"
        ),
        "reason": (reason or "policy_block")[:_PRE_DELIVERY_REASON_MAX_CHARS],
        "required": (
            "Retén la respuesta anterior. Recupera la evidencia o certificación "
            "faltante con las herramientas disponibles, regenera una respuesta "
            "específica y completa, y vuelve a someterla a pre_delivery."
        ),
    }
    return "[PRE_DELIVERY_REPAIR]\n" + json.dumps(
        repair_contract, ensure_ascii=False, sort_keys=True
    )


@dataclass
class StopGateVerdict:
    """``continue_turn`` True → a nudge was appended; re-enter the turn loop with
    ``final_response=None`` and the pending-verification fields updated."""

    continue_turn: bool
    final_response: Any
    pending_verification_response: Any
    pending_verification_response_previewed: Any


def _verify_on_stop_nudge(agent) -> Optional[str]:
    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge, verify_on_stop_enabled
        )

        if verify_on_stop_enabled():
            return build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
    except Exception:
        logger.debug("verification stop-loop check failed", exc_info=True)
    return None


def _pre_verify_nudge(agent, final_response, attempt: int) -> Optional[str]:
    """After code edits a registered ``pre_verify`` hook may keep the agent going one
    more turn; no default continuation cost."""
    _edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        if _edited and has_hook("pre_verify") and attempt < max_verify_nudges():
            # Posture is fixed for the session — resolve once + cache.
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context
                coding = bool(is_coding_context(platform=getattr(agent, "platform", "") or ""))
                agent._resolved_is_coding = coding
            return get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "", coding=coding, attempt=attempt,
                final_response=final_response, changed_paths=_edited,
            )
    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
    """Workers must end with kanban_complete / kanban_block; a narrated stop is recorded
    as protocol_violation, so nudge once or twice first."""
    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        return build_kanban_stop_nudge(
            messages=messages, attempts=getattr(agent, "_kanban_stop_nudges", 0)
        )
    except Exception:
        logger.debug("kanban stop-loop check failed", exc_info=True)
        return None


def _append_interim_answer(agent, final_msg, messages, conversation_history, flush_fail_msg: str) -> None:
    """Real content: persist and emit as interim so the user sees the attempted answer;
    only the nudge is flagged synthetic (#65919)."""
    agent._emit_interim_assistant_message(final_msg)
    append_message(messages, final_msg)
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.debug(flush_fail_msg, exc_info=True)


def apply_stop_gates(
    agent: Any, final_msg: Dict[str, Any], *, final_response: Any, messages: List[Dict[str, Any]],
    conversation_history: Any, pending_verification_response: Any,
    pending_verification_response_previewed: Any, effective_task_id: str = "",
    turn_id: str = "", original_user_message: Any = None,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard, in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""

    def _continue(
        nudge: str, flag: str, *, fallback_response: Any = _USE_FINAL_RESPONSE_AS_FALLBACK,
        fallback_previewed: Optional[bool] = None,
    ) -> StopGateVerdict:
        """Append the synthetic nudge row and hand the turn back to the loop."""
        append_message(messages, {"role": "user", "content": nudge, flag: True})
        agent._session_messages = messages
        # Keep the answer only as a budget-exhaustion fallback; clear ``final_response`` so
        # the finalizer can tell this gate from error exits. Mark previewed only if the
        # candidate is reused (#61631).
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=(
                final_response
                if fallback_response is _USE_FINAL_RESPONSE_AS_FALLBACK
                else fallback_response
            ),
            pending_verification_response_previewed=(
                agent._interim_content_was_streamed(final_response or "")
                if fallback_previewed is None else fallback_previewed
            ),
        )

    # A delivery gate runs while the candidate is still private. A block does not become a
    # terminal generic refusal: feed a structured repair request back into the same agent
    # turn so it can gather evidence/use tools/regenerate. Neither the blocked candidate nor this
    # nudge is emitted or persisted; both are stripped once the next candidate arrives.
    try:
        from hermes_cli.plugins import has_hook

        _has_pre_delivery = has_hook("pre_delivery")
    except Exception:
        logger.warning("pre_delivery hook lookup failed", exc_info=True)
        # Lookup uncertainty is not permission to bypass a potentially installed
        # delivery boundary. Dispatch below normalizes the failure fail-closed.
        _has_pre_delivery = True
    if _has_pre_delivery:
        from agent.turn_finalizer import _prepare_output_for_delivery

        decision = _prepare_output_for_delivery(
            agent, final_response, logger,
            platform=getattr(agent, "platform", "") or "",
            effective_task_id=effective_task_id,
            turn_id=turn_id,
            original_user_message=(
                original_user_message if original_user_message is not None else ""
            ),
            messages=messages,
        )
        if decision.action != "block":
            final_response = decision.response_text
            final_msg["content"] = final_response
            agent._pre_delivery_output_cache = {
                "transformed": decision.transformed,
                "pre_transform": decision.pre_transform_response,
                "response_text": final_response,
            }
            release = getattr(agent, "_release_pre_delivery_text", None)
            if callable(release):
                release(final_response)
        else:
            attempt = int(getattr(agent, "_pre_delivery_repair_attempts", 0) or 0)
            # A policy block is private recovery state, never a user-facing answer.  Crossing
            # the standard repair threshold escalates the directive but does not publish a
            # canned refusal or retain one as an exhaustion fallback.
            agent._pre_delivery_repair_attempts = attempt + 1
            from agent.clinical_trace import emit
            emit(
                agent, "repair", outcome="block",
                repair_count=agent._pre_delivery_repair_attempts,
            )
            final_msg["finish_reason"] = "pre_delivery_repair_required"
            final_msg["_pre_delivery_repair_synthetic"] = True
            append_message(messages, final_msg)
            from agent.turn_finalizer import _pre_delivery_reason_codes
            reason = _pre_delivery_reason_codes(decision.reason)[0]
            nudge = _build_pre_delivery_repair_directive(
                attempt=agent._pre_delivery_repair_attempts,
                reason=reason,
            )
            verdict = _continue(
                nudge, "_pre_delivery_repair_synthetic",
                fallback_response=None, fallback_previewed=False,
            )
            logger.info(
                "pre_delivery repair nudge issued (attempt %d, mode=%s)",
                agent._pre_delivery_repair_attempts,
                (
                    "standard"
                    if agent._pre_delivery_repair_attempts <= _MAX_PRE_DELIVERY_REPAIR_ATTEMPTS
                    else "escalated"
                ),
            )
            return verdict

    _verify_nudge = _verify_on_stop_nudge(agent)
    if _verify_nudge:
        agent._verification_stop_nudges = getattr(agent, "_verification_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "verify-on-stop interim flush failed"
        )
        verdict = _continue(_verify_nudge, "_verification_stop_synthetic")
        # Internal nudge: stay silent on the terminal, debug-log only.
        logger.debug("verification stop-loop nudge issued (attempt %d)", agent._verification_stop_nudges)
        return verdict

    _attempt = getattr(agent, "_pre_verify_nudges", 0)
    _verify_nudge2 = _pre_verify_nudge(agent, final_response, _attempt)
    if _verify_nudge2:
        agent._pre_verify_nudges = _attempt + 1
        final_msg["finish_reason"] = "verify_hook_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_verify interim flush failed"
        )
        verdict = _continue(_verify_nudge2, "_pre_verify_synthetic")
        logger.debug("pre_verify nudge issued (attempt %d)", agent._pre_verify_nudges)
        return verdict

    _kanban_nudge = _kanban_stop_nudge(agent, messages)
    if _kanban_nudge:
        agent._kanban_stop_nudges = getattr(agent, "_kanban_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "kanban_terminal_required"
        final_msg["_kanban_stop_synthetic"] = True
        append_message(messages, final_msg)
        verdict = _continue(_kanban_nudge, "_kanban_stop_synthetic")
        logger.info(
            "kanban stop-loop nudge issued (attempt %d) task=%s",
            agent._kanban_stop_nudges,
            os.environ.get("HERMES_KANBAN_TASK", ""),
        )
        agent._emit_status(
            "⚠️ Kanban worker tried to exit without "
            "kanban_complete/kanban_block — nudging to finish"
        )
        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
