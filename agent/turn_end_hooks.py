"""Generic pre-delivery text-turn policy gate."""
import copy
import logging
from dataclasses import dataclass

from agent.message_metadata import append_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalGateVerdict:
    action: str = "allow"
    message: str = ""
    code: str = ""


_INTERRUPTED = FinalGateVerdict("interrupt", "", "interrupted_during_completion_review")


def _failure(agent, message, code):
    verdict = FinalGateVerdict("fail", message, code)
    agent._turn_end_failure = verdict
    agent._turn_end_pending = False
    return verdict


def without_rejected_messages(messages):
    if not any(isinstance(row, dict) and row.get("_turn_end_synthetic") for row in messages):
        return messages
    return [row for row in messages if not (isinstance(row, dict) and row.get("_turn_end_synthetic"))]


def _max_continuations():
    from hermes_cli.config import load_config
    value = (load_config().get("agent") or {}).get("max_final_continuations", 2)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("agent.max_final_continuations must be a non-negative integer")
    return value


def _directive(result):
    if result is None:
        return FinalGateVerdict()
    if not isinstance(result, dict):
        raise ValueError("completion policy must return a directive or None")
    action = result.get("action")
    if action == "allow":
        return FinalGateVerdict()
    message = result.get("message")
    if action in {"continue", "block"} and isinstance(message, str) and message.strip():
        return FinalGateVerdict("continue", message.strip())
    if action == "fail" and isinstance(message, str) and message.strip():
        code = result.get("code", "final_response_rejected")
        if not isinstance(code, str) or not code.strip() or len(code) > 128:
            raise ValueError("completion policy failure code must be a bounded string")
        return FinalGateVerdict("fail", message.strip(), code.strip())
    raise ValueError("invalid completion policy directive")


def _current_turn_messages(messages):
    """Exclude prior turns while retaining this hook's synthetic repair round."""
    from agent.conversation_compression import _is_real_user_message, _message_contains_busy_steer
    for index in range(len(messages or ()) - 1, -1, -1):
        item = messages[index]
        if _is_real_user_message(item) and not _message_contains_busy_steer(item):
            return copy.deepcopy(messages[index:])
    return []


def defers_text_delivery():
    try:
        from hermes_cli.lifecycle import has_hook
        return has_hook("before_turn_end")
    except Exception:
        logger.warning("before_turn_end lookup failed; withholding candidate text", exc_info=True)
        return True


def prepared_response(agent, text):
    record = getattr(agent, "_turn_end_prepared", None)
    if record and record[0] == getattr(agent, "_current_turn_id", "") and record[1][0] == text:
        return record[1]
    return None


def prepare_response(agent, text):
    """Opted-in gates must see output transforms, not a draft later replaced by a plugin."""
    if not defers_text_delivery():
        return text
    if prepared_response(agent, text):
        return text
    from agent.turn_finalizer import _append_file_mutation_footer, apply_llm_output_transform
    turn_id = getattr(agent, "_current_turn_id", "") or ""
    text = _append_file_mutation_footer(agent, text, logger)
    # The stock transform records one outcome per turn; a reworked candidate is new output.
    agent._llm_output_transform = None
    value = apply_llm_output_transform(agent, text, turn_id=turn_id, logger=logger)
    agent._turn_end_prepared = (getattr(agent, "_current_turn_id", ""), value)
    return value[0]


def before_turn_end(agent, final_response, final_msg, messages, *, user_message, can_continue):
    if getattr(agent, "_interrupt_requested", False):
        return _INTERRUPTED
    try:
        from hermes_cli.lifecycle import invoke_hook
        if not defers_text_delivery():
            return FinalGateVerdict()
        turn_id = getattr(agent, "_current_turn_id", "")
        agent._turn_end_checked = (turn_id, final_response)
        attempt = getattr(agent, "_turn_end_continuations", 0)
        budget = _max_continuations()
        available = can_continue and attempt < budget
        results = invoke_hook("before_turn_end", final_response=final_response,
            session_id=getattr(agent, "session_id", "") or "",
            task_id=getattr(agent, "_current_task_id", "") or "", turn_id=turn_id,
            platform=getattr(agent, "platform", "cli"),
            model=getattr(agent, "model", ""), provider=getattr(agent, "provider", ""),
            effort=(getattr(agent, "_turn_end_request_route", None) or {}).get("effort"),
            requested_effort=(getattr(agent, "reasoning_config", None) or {}).get("effort"),
            attempt=attempt, already_blocked=attempt > 0, can_continue=available,
            user_message=user_message, messages=_current_turn_messages(messages),
            source_identity=getattr(agent, "_current_source_identity", None))
        if getattr(agent, "_interrupt_requested", False):
            return _INTERRUPTED
        directives = [_directive(result) for result in results]
        for directive in directives:
            if directive.action == "fail":
                return _failure(agent, directive.message, directive.code)
        feedback = [directive.message for directive in directives if directive.action == "continue"]
        if feedback:
            if not available:
                code = "final_continuation_limit" if attempt >= budget else "final_iteration_limit"
                return _failure(agent, "The answer could not pass completion review within the allowed iterations.", code)
            agent._turn_end_continuations = attempt + 1
            agent._turn_end_pending = True
            # Both rows are private repair context. Neither can become a fallback answer.
            final_msg["_turn_end_synthetic"] = True
            append_message(messages, final_msg)
            append_message(messages, {"role": "user", "content": "\n\n".join(feedback), "_turn_end_synthetic": True})
            agent._session_messages = messages
            return FinalGateVerdict("continue")
        agent._turn_end_pending = False
        return FinalGateVerdict()
    except Exception:
        logger.warning("before_turn_end policy could not complete", exc_info=True)
        return _failure(agent, "The answer was withheld because completion review could not finish.", "final_policy_error")


def prepare_text_candidate(agent, final_response, final_msg, promoted):
    """Apply output transforms before review and write the reviewed text to the candidate row."""
    final_response = prepare_response(agent, final_response)
    if defers_text_delivery():
        # Same row contract as the stock transform: a promoted reasoning-only reply keeps ``content``
        # empty and carries the text in the ``api_content`` sidecar.
        final_msg["api_content" if promoted else "content"] = final_response
    return final_response


def review_text_candidate(agent, final_response, final_msg, messages, user_message, api_call_count):
    """Review a text-turn candidate before the stop gates. ``None`` lets delivery proceed; otherwise the
    loop ``action`` plus the turn-loop fields to rebind (the candidate is never the turn's answer)."""
    gate = before_turn_end(
        agent, final_response, final_msg, messages, user_message=user_message,
        can_continue=api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0,
    )
    if gate.action == "allow":
        return None
    fields = {"_pending_verification_response": None, "_pending_verification_response_previewed": False,
              "failed": gate.action == "fail", "interrupted": gate.action == "interrupt"}
    if gate.action == "continue":
        return {"action": "continue", "final_response": None, **fields}
    return {"action": "break", "final_response": gate.message, "_turn_exit_reason": gate.code, **fields}


def settle_review_failure(agent, final_response, turn_exit_reason, failed, interrupted):
    """A repair round that never produced a reviewed answer fails the turn, and a recorded policy failure
    replaces the response. Returns ``(final_response, turn_exit_reason, failed)``."""
    if getattr(agent, "_turn_end_pending", False) and not interrupted:
        _failure(agent, "The answer could not pass completion review before the turn ended.", "final_review_incomplete")
    failure = getattr(agent, "_turn_end_failure", None)
    if failure is None or interrupted:
        return final_response, turn_exit_reason, failed
    return failure.message, failure.code, True


def review_terminal_response(agent, final_response, turn_exit_reason, failed, interrupted, messages, *,
                             turn_id, user_message):
    """Review a recovery answer (iteration-limit summary or partial answer) before any durable write, then
    drop held rows. Recovery answers have no repair budget. Returns ``(final_response, turn_exit_reason,
    failed, interrupted)``. A ``pre_llm_call`` veto has no model answer, so the policy never sees its
    block message or the vetoed prompt."""
    from agent.turn_context import PRE_LLM_BLOCK_EXIT_REASON
    if (final_response and not interrupted and turn_exit_reason != PRE_LLM_BLOCK_EXIT_REASON
            and getattr(agent, "_turn_end_failure", None) is None
            and defers_text_delivery() and not getattr(agent, "_tool_guardrail_halt_decision", None)
            and getattr(agent, "_turn_end_checked", None) != (turn_id, final_response)):
        final_response = prepare_response(agent, final_response)
        gate = before_turn_end(agent, final_response, {}, messages, user_message=user_message, can_continue=False)
        if gate.action == "fail":
            final_response, turn_exit_reason, failed = gate.message, gate.code, True
        elif gate.action == "interrupt":
            final_response, turn_exit_reason, interrupted = "", gate.code, True
    messages[:] = without_rejected_messages(messages)
    return final_response, turn_exit_reason, failed, interrupted


def stamp_review_failure(agent, result, interrupted):
    """Report a policy failure as the turn's non-retryable failure reason."""
    failure = getattr(agent, "_turn_end_failure", None)
    if failure is not None and not interrupted:
        from agent.turn_failure_copy import stamp_failure
        result["error"] = failure.message
        stamp_failure(result, failure.code, False)


def finalize_early_result(agent, result, user_message):
    """Review partial model answers from exits that bypass normal turn finalization."""
    if not isinstance(result, dict) or not defers_text_delivery():
        return result
    messages = result.get("messages")
    if not isinstance(messages, list):
        return result
    held = any(isinstance(row, dict) and row.get("_turn_end_synthetic") for row in messages)
    text = result.get("final_response")
    reviewed = getattr(agent, "_turn_end_checked", None) == (getattr(agent, "_current_turn_id", ""), text)
    failure = getattr(agent, "_turn_end_failure", None)
    if result.get("interrupted"):
        if held:
            result["final_response"] = ""
    elif (failure is None and text and not reviewed and result.get("api_calls", 0) > 0
          and not getattr(agent, "_tool_guardrail_halt_decision", None)):
        text = prepare_response(agent, text)
        verdict = before_turn_end(agent, text, {}, messages, user_message=user_message, can_continue=False)
        if verdict.action == "interrupt":
            result.update(final_response="", interrupted=True, completed=False)
        elif verdict.action == "fail":
            failure = verdict
        else:
            result["final_response"] = text
    if failure is not None and not result.get("interrupted"):
        from agent.turn_failure_copy import stamp_failure
        result.update(final_response=failure.message, error=failure.message, failed=True, completed=False,
                      partial=False, turn_exit_reason=failure.code)
        stamp_failure(result, failure.code, False)
    if held:
        messages[:] = without_rejected_messages(messages)
        result["last_reasoning"] = ""
        # An early partial may already have attempted persistence. Its held rows never reached SQLite.
        # Write only the reviewed answer (or controlled failure) at the same transcript boundary.
        if result.get("final_response") and not result.get("interrupted"):
            append_message(messages, {"role": "assistant", "content": result["final_response"]})
        agent._session_messages = messages
        agent._persist_session(messages, None)
    return result
