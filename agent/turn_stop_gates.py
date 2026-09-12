"""Text-response stop gates for the conversation turn loop.

When the model stops with a text answer, four gates may instead append the answer as an
interim row plus a synthetic user-role nudge and continue the turn: verify-on-stop (#65919),
the ``pre_verify`` plugin hook after code edits, the kanban worker terminal-tool guard, and
the generic ``pre_turn_finalize`` plugin hook. Each keeps the candidate answer as a
budget-exhaustion fallback
(``pending_verification_response``) and clears ``final_response`` so the finalizer can tell
this gate from error exits (#61631). Nothing here imports ``agent.conversation_loop`` at
module level (cycle).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


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


#: Hard bound of ``pre_turn_finalize`` continuations per turn (V1 has no config key, so a
#: plugin that always returns ``continue`` can never trap the loop).
MAX_PRE_TURN_FINALIZE_NUDGES = 1


def _pre_turn_finalize_nudge(
    agent, final_response, *, finish_reason: str, api_call_count: int, turn_id: str,
    attempt: int,
) -> Optional[str]:
    """Last plugin checkpoint before a plain-text final is accepted. Fires after every
    built-in stop gate, so it can never bypass mandatory stop policy. Fail-open: any
    lookup/dispatch error finalizes normally."""
    if attempt >= MAX_PRE_TURN_FINALIZE_NUDGES:
        return None
    try:
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_turn_finalize_continue_message

        if not has_hook("pre_turn_finalize"):
            return None
        return get_pre_turn_finalize_continue_message(
            session_id=getattr(agent, "session_id", None) or "",
            turn_id=turn_id or getattr(agent, "_current_turn_id", "") or "",
            platform=getattr(agent, "platform", "") or "",
            model=getattr(agent, "model", "") or "",
            provider=getattr(agent, "provider", "") or "",
            final_response=final_response if isinstance(final_response, str) else "",
            finish_reason=finish_reason or "stop",
            api_call_count=api_call_count,
            attempt=attempt,
        )
    except Exception:
        logger.debug("pre_turn_finalize hook check failed", exc_info=True)
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
    pending_verification_response_previewed: Any,
    finish_reason: Any = "stop", api_call_count: int = 0, turn_id: Any = None,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard → pre_turn_finalize hook, in
    that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there). The generic ``pre_turn_finalize`` plugin checkpoint runs last so it can never
    bypass the mandatory built-in stop policy."""

    def _continue(nudge: str, flag: str) -> StopGateVerdict:
        """Append the synthetic nudge row and hand the turn back to the loop."""
        append_message(messages, {"role": "user", "content": nudge, flag: True})
        agent._session_messages = messages
        # Keep the answer only as a budget-exhaustion fallback; clear ``final_response`` so
        # the finalizer can tell this gate from error exits. Mark previewed only if the
        # candidate is reused (#61631).
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=final_response,
            pending_verification_response_previewed=agent._interim_content_was_streamed(
                final_response or ""
            ),
        )

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

    _finalize_attempt = getattr(agent, "_pre_turn_finalize_nudges", 0)
    _finalize_nudge = _pre_turn_finalize_nudge(
        agent, final_response, finish_reason=str(finish_reason or "stop"),
        api_call_count=api_call_count, turn_id=str(turn_id or ""),
        attempt=_finalize_attempt,
    )
    if _finalize_nudge:
        agent._pre_turn_finalize_nudges = _finalize_attempt + 1
        final_msg["finish_reason"] = "pre_turn_finalize_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history,
            "pre_turn_finalize interim flush failed",
        )
        verdict = _continue(_finalize_nudge, "_pre_turn_finalize_synthetic")
        logger.debug("pre_turn_finalize nudge issued (attempt %d)", agent._pre_turn_finalize_nudges)
        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
