"""Opt-in reaction lifecycle for platform adapters."""

from typing import Any, Optional
from .event import MessageEvent, ProcessingOutcome


class ProcessingReactionHooksMixin:
    # ── Processing lifecycle hooks (Discord 👀/✅/❌ reactions). Adapters exposing
    # ``_add_reaction(chat_id, message_id, emoji)`` / ``_remove_reaction(chat_id, message_id)``
    # can just set the emoji attributes; left ``None`` the hook is a no-op.
    _ACK_EMOJI: Optional[str] = None
    _OK_EMOJI: Optional[str] = None
    _FAIL_EMOJI: Optional[str] = None

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """Hook called when background processing completes. Default: opt-in reaction ack — with
        ``_OK_EMOJI``/``_FAIL_EMOJI`` set and ``_add_reaction``/``_remove_reaction`` present, swap
        the in-progress reaction for the outcome one. Remove-then-add is deterministic whether the
        platform replaces or stacks a sender's reactions. CANCELLED leaves it unreacted."""
        if self._OK_EMOJI is None and self._FAIL_EMOJI is None:
            return
        add: Any = getattr(self, "_add_reaction", None)
        remove: Any = getattr(self, "_remove_reaction", None)
        enabled = getattr(self, "_reactions_enabled", None)
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if (
            not callable(add)
            or not callable(remove)
            or (callable(enabled) and not enabled())
            or not chat_id
            or not message_id
        ):
            return
        await remove(chat_id, message_id)
        emoji = {
            ProcessingOutcome.SUCCESS: self._OK_EMOJI,
            ProcessingOutcome.FAILURE: self._FAIL_EMOJI,
        }.get(outcome)
        if emoji:
            await add(chat_id, message_id, emoji)
