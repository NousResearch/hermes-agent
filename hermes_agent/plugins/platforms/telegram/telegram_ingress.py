"""Inbound Telegram shapes the core handlers miss.

Contact cards (#98598): a shared contact arrives as a ``contact`` payload with no ``text``, so no
text/media handler matched and PTB dropped the update; the agent never learned a contact was sent
and guessed at a permissions problem instead. Rendered as a text turn, like location pins.

Reset boundaries (#81370): a photo burst (``_pending_photo_batches``, ~0.8 s) or album
(``_media_group_events``, ~1.5 s) is held in the adapter before it becomes a turn. The base
adapter's ``/stop``, ``/new`` and ``/reset`` handling clears only its own text debounce and pending
slot, so a photo sent just before ``/stop`` flushed right after it and started a NEW turn the user
had just cancelled.
"""

from typing import Any

from hermes_agent.gateway.platforms.event import MessageEvent, MessageType


def contact_text(contact) -> str:
    """The agent-facing text block for a Telegram ``Contact``."""
    parts = ["[The user shared a contact card.]"]
    name = " ".join(p for p in (getattr(contact, "first_name", None), getattr(contact, "last_name", None)) if p)
    for label, value in (("Name", name), ("Phone", getattr(contact, "phone_number", None)),
                         ("Telegram user id", getattr(contact, "user_id", None))):
        if value:
            parts.append(f"{label}: {value}")
    vcard = getattr(contact, "vcard", None)
    if vcard:
        parts += ["vCard:", vcard]
    return "\n".join(parts)


class TelegramContactMixin:
    """``filters.CONTACT`` handler for ``TelegramAdapter``."""

    async def _handle_contact_message(self, update, context) -> None:
        msg = self._effective_update_message(update)
        contact = getattr(msg, "contact", None) if msg else None
        if not contact:
            return
        if not self._is_user_authorized_from_message(msg):
            self._log_blocked_user(msg)
            return
        proceed = self._should_process_message(msg)
        if not proceed and not self._should_observe_unmentioned_group_message(msg):
            return
        event = self._build_message_event(msg, MessageType.TEXT, update_id=update.update_id)
        event.text = contact_text(contact)
        if not proceed:  # observed chatter keeps the card, so a later mention can use it
            self._observe_unmentioned_group_message(msg, MessageType.TEXT, update_id=update.update_id, event=event)
            return
        await self.handle_message(self._apply_telegram_group_observe_attribution(event))

    def _discard_text_debounce(self, session_key: str) -> None:
        """Base calls this where it honours ``/stop``, ``/new`` or ``/reset`` for the session (and on a
        stale-lock heal); the photo/album buffers are this adapter's debounce, so they go too (#81370)."""
        super()._discard_text_debounce(session_key)
        discard_session_media_buffers(self, session_key)


def _drop_matching(adapter: Any, pending: dict, tasks: dict, session_key: str) -> int:
    """Drop every buffered event in ``pending`` whose session is ``session_key``; cancel its flush."""
    dropped = 0
    for key, buffered in list(pending.items()):
        if adapter._event_session_key(buffered) != session_key:
            continue
        pending.pop(key, None)
        task = tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()
        dropped += 1
    return dropped


def discard_session_media_buffers(adapter: Any, session_key: str) -> int:
    """Drop ``session_key``'s buffered photo bursts and albums so they cannot surface as a turn after a
    reset boundary. Returns the count dropped."""
    return (_drop_matching(adapter, adapter._pending_photo_batches, adapter._pending_photo_batch_tasks, session_key)
            + _drop_matching(adapter, adapter._media_group_events, adapter._media_group_tasks, session_key))
