"""Deliver diagnostics and lifecycle evidence for early failed turn returns."""

import logging

logger = logging.getLogger(__name__)


def finalize_failed_turn(agent, result, *, task_id, turn_id, failure_reason, finish_reason):
    """Preserve failure while allowing plugins to format its diagnostic text.

    This path never runs the successful-output transform. A plugin can replace
    diagnostic text but cannot change the failed result or the source reason.
    """
    from hermes_cli.lifecycle import invoke_hook
    from agent.message_sanitization import _sanitize_surrogates

    result = dict(result)
    result.update(completed=False, failed=True, failure_reason=failure_reason)
    result["turn_exit_reason"] = failure_reason
    result["finish_reason"] = finish_reason
    context = dict(
        session_id=agent.session_id or "", task_id=task_id, turn_id=turn_id,
        completed=False, failed=True, interrupted=False,
        turn_exit_reason=failure_reason, finish_reason=finish_reason,
        model=agent.model, platform=getattr(agent, "platform", None) or "",
    )
    text = result.get("final_response") or result.get("error") or "Turn failed"
    try:
        for replacement in invoke_hook(
            "transform_turn_failure", response_text=text,
            error=result.get("error"), **context,
        ):
            if isinstance(replacement, str) and replacement:
                text = replacement
                break
    except Exception:
        logger.warning("transform_turn_failure hook failed", exc_info=True)
    result["final_response"] = _sanitize_surrogates(text)
    try:
        invoke_hook("on_session_end", **context)
    except Exception:
        logger.warning("on_session_end hook failed", exc_info=True)
    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    return result
