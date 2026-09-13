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


def todo_continuation_nudge(agent, messages, full_response: str) -> Optional[str]:
    """Reconcile adopted work, not a stale plan or a new mandate.

    Todo statuses do not encode authority or wait ownership. Only an adopted
    in-progress task is a candidate; declared human gates and owned wakeups yield.
    """
    import json
    import re

    if (getattr(agent, "_interrupt_requested", False)
            or getattr(agent, "_pending_steer", None)
            or getattr(agent, "_tool_guardrail_halt_decision", None) is not None):
        return None
    store = getattr(agent, "_todo_store", None)
    if store is None or not any(t.get("status") == "in_progress" for t in store.read()):
        return None
    # This is a conservative opt-out, not an authorization classifier. A
    # declared dependency or question is enough to leave the answer alone.
    if re.search(r"\b(?:approval|permission|consent|paused|stopped|blocked|"
                 r"awaiting|waiting|pending|credentials)\b|\bneed (?:your|you to)\b|[?]",
                 full_response, re.IGNORECASE):
        return None

    # A successful write after the latest user row adopts the plan. Reading
    # it for a status question, history, or a redirect does not. Synthetic user
    # rows also terminate the scan conservatively.
    results = {}
    adopted = False
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = message.get("content")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "todo_list":
                continue
            try:
                args = json.loads(function.get("arguments", "{}"))
                result = json.loads(results.get(call.get("id")) or "{}")
                adopted = adopted or (bool(args.get("todos"))
                                      and result.get("revision") == store.snapshot()["revision"]
                                      and result.get("todos") == store.read())
            except (ValueError, TypeError, AttributeError):
                continue
    if not adopted:
        return None

    # Parent turns must end to receive async completion callbacks. Do not
    # let another session's processes suppress this session's reconciliation.
    from hermes_cli.goals import count_active_delegations, gather_background_processes
    if count_active_delegations(getattr(agent, "session_id", None)):
        return None
    owner = getattr(agent, "_current_task_id", None)
    if owner and any(p.get("notify_on_complete") or p.get("watch_patterns")
                     for p in gather_background_processes(owner_task_id=owner)):
        return None
    return (
        "Your current task list still has in-progress work. Before ending, reconcile it "
        "with the latest user request and the evidence already collected. Continue only "
        "a runnable, already-authorized next step; isolate a blocked lane rather than "
        "repeating failed attempts. Do not treat the task list as permission to send, "
        "spend, publish, merge, or bypass approval. Respect stop/redirect instructions. "
        "If work is complete, update the list truthfully. If only a human gate or an "
        "owned background wakeup remains, state that boundary and end the turn without "
        "busy-polling or starting duplicate work. This is a bounded reconciliation, "
        "not a new mandate or an extension of the run budget."
    )


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
    pending_verification_response_previewed: Any,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard, in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""

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
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
