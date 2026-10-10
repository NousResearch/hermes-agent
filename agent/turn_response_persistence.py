"""Final text-row persistence seam, fenced for selected session topics."""
from __future__ import annotations
import logging
from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


def commit_final_text_response(agent, messages, conversation_history, final_msg, final_response, promoted):
    """Transform and flush the ordinary path; hold selected-topic candidates for finalization."""
    from agent.turn_finalizer import apply_llm_output_transform

    selected = getattr(agent, "_topic_segmentation_enabled", False)
    if not getattr(agent, "_interrupt_requested", False) and not selected:
        final_response, transformed, _ = apply_llm_output_transform(
            agent, final_response, turn_id=getattr(agent, "_current_turn_id", "") or "", logger=logger,
        )
        if transformed:
            if promoted:
                final_msg["api_content"] = final_response
            else:
                final_msg["content"] = final_response
    append_message(messages, final_msg)
    # The selected topic owns its candidate until the finalizer commits selection.
    if not selected:
        try:
            agent._flush_messages_to_session_db(messages, conversation_history)
        except Exception:
            logger.warning(
                "final text-turn flush failed (session=%s) — reply is "
                "not yet durable; relying on finalize_turn retry",
                getattr(agent, "session_id", None) or "none",
                exc_info=True,
            )
    return final_response
