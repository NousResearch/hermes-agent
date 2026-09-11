"""Recovery-message policy for turns resumed after a gateway interruption."""

from typing import Optional

from agent.replay_cleanup import RESUME_RECOVERY_NOTE_PREFIX


def build_resume_recovery_note(
    reason: Optional[str], message: str = "", *, interactive: bool = True
) -> str:
    """Build the recovery note for an interrupted turn.

    An empty ``message`` denotes the startup auto-resume event. It always
    continues the saved work because acknowledging the restore would complete
    the synthetic turn and clear the durable marker while abandoning the task.
    ``interactive`` remains part of the call contract for adapter compatibility.
    """
    reason_phrase = (
        "a gateway restart"
        if reason == "restart_timeout"
        else "a gateway shutdown"
        if reason == "shutdown_timeout"
        else "a gateway interruption"
    )
    if message:
        resume_guidance = (
            "Address the user's NEW message below FIRST and focus on what the user is asking now. "
            "If it explicitly asks to resume or finish the interrupted task, continue from the next "
            "unfinished step using recorded results. Status or reporting requests must report from "
            "recorded history without resuming work. Otherwise do not revive unrelated unfinished work."
        )
        tail_guidance = (
            "Do NOT repeat successful tool calls merely to recreate completed work. "
            "A failed or incomplete result may require a retry after checking the cause and current state."
        )
    else:
        resume_guidance = (
            "Do NOT emit only a 'session restored' acknowledgement "
            "or ask the user to restate the task. Review the conversation history and "
            "CONTINUE the interrupted task to completion."
        )
        tail_guidance = (
            "Use recorded results to identify the next unfinished step; do NOT repeat successful calls "
            "merely to recreate completed work. Retry failed or incomplete work only after checking the "
            "cause and current state. If a tool call has no recorded result, its effect is UNKNOWN. Inspect "
            "current state before retrying. If the effect cannot be verified and retrying "
            "could duplicate an external or irreversible action, stop and ask one specific safety question."
        )
    return (
        f"{RESUME_RECOVERY_NOTE_PREFIX} "
        f"{reason_phrase}; the gateway is now back online. "
        f"Do NOT repeat a restart/shutdown command simply because its response was interrupted. "
        f"Use recorded tool results as evidence. If a tool's effect is UNKNOWN, inspect current state "
        f"with read-only checks before deciding whether any unfinished action is still needed. "
        f"Read-only verification of the running revision or task outcome is allowed when required "
        f"by the active request. {resume_guidance} {tail_guidance}]"
        + (f"\n\n{message}" if message else "")
    )


def prepare_resume_pending_message(
    reason: Optional[str], message: Optional[str], *, interactive: bool = True
) -> tuple[str, str]:
    """Return model recovery guidance and the user text to persist.

    A synthesized blank event persists the recovery note because a blank user
    row repeatedly triggers the pre-call sanitizer (#86580). Real user text is
    stored without the recovery scaffold while the model receives both.
    """
    recovery_message = build_resume_recovery_note(
        reason, message or "", interactive=interactive
    )
    persist_message = (
        message if isinstance(message, str) and message.strip() else recovery_message
    )
    return recovery_message, persist_message
