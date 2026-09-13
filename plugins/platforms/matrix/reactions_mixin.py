"""Reaction send/receive and the reaction-driven processing lifecycle for MatrixAdapter.

Extracted from ``plugins/platforms/matrix/adapter.py`` as part of the god-file
decomposition campaign, following the same mechanical mixin lift that produced
``gateway/slash_commands_session.py`` and the ``SessionDB`` mixins in
``hermes_state_*.py``.

Matrix has no native buttons, so reactions are the interaction surface: they
are how the adapter signals that a turn started or finished, and how a user
answers an approval prompt or a model picker. That is the cluster this module
holds.

Mixin contract: a plain mixin consumed by ``MatrixAdapter``. It defines no
``__init__`` and no state of its own; methods reach the host's attributes and
its other methods (``self._client``, ``self.send``, ``self.redact_message``,
``self._claim_reaction_prompt``, ``self._handle_approval_reaction``,
``self._handle_picker_reaction``) through the MRO.  ``MatrixReactionsMixin``
precedes ``BasePlatformAdapter`` in the bases, which matters here:
``on_processing_start`` and ``on_processing_complete`` are overrides of
base-class methods, and the mixin must win exactly as the adapter's own
definitions did. It never imports the adapter module, so there is no cycle.

Behavior-neutral: every method is lifted verbatim from ``MatrixAdapter``.
``logger`` is bound by explicit name rather than ``__name__``, so records keep
the name ``"plugins.platforms.matrix.adapter"``; ``getLogger`` returns the
same singleton object the adapter module holds. ``EventType`` and ``RoomID``
are imported under the adapter's own ``ImportError`` guard; the moved code
uses only ``RoomID`` as a cast and ``EventType.REACTION`` as a string
constant, so the real mautrix types and the fallback stubs behave alike.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from gateway.platforms.event import MessageEvent, ProcessingOutcome

try:
    from mautrix.types import EventType, RoomID
except ImportError:
    # Import-safe stubs without mautrix, mirroring the adapter's own guard.
    RoomID = str  # type: ignore[misc,assignment]

    EventType = type("_EventTypeStub", (), {  # type: ignore[misc,assignment]
        "ROOM_MESSAGE": "m.room.message", "REACTION": "m.reaction",
        "ROOM_ENCRYPTED": "m.room.encrypted", "ROOM_NAME": "m.room.name"})

# Bind the adapter's logger by name so records lifted with these methods are
# emitted under exactly the name they were before.
logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixReactionsMixin:
    """See module docstring - reactions cluster lifted verbatim from MatrixAdapter."""


    async def _send_reaction(self, room_id: str, event_id: str, emoji: str) -> Optional[str]:
        """Send an emoji reaction; returns the reaction event_id, or None on failure."""
        if not self._client:
            return None
        content = {"m.relates_to": {"rel_type": "m.annotation", "event_id": event_id, "key": emoji}}
        try:
            resp_event_id = await self._client.send_message_event(RoomID(room_id), EventType.REACTION, content)
            logger.debug("Matrix: sent reaction %s to %s", emoji, event_id)
            return str(resp_event_id)
        except Exception as exc:
            logger.debug("Matrix: reaction send error: %s", exc)
            return None

    async def _redact_reaction(self, room_id: str, reaction_event_id: str, reason: str = "") -> bool:
        return await self.redact_message(room_id, reaction_event_id, reason)

    def _schedule_reaction_redaction(self, room_id: str, reaction_event_id: str, reason: str = "") -> None:
        """Redact a reaction after a short delay so message delivery settles."""

        async def _redact_later() -> None:
            try:
                if self._reaction_redaction_delay_seconds:
                    await asyncio.sleep(self._reaction_redaction_delay_seconds)
                if not await self._redact_reaction(room_id, reaction_event_id, reason):
                    logger.debug("Matrix: failed to redact reaction %s", reaction_event_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Matrix: delayed reaction redaction failed for %s: %s", reaction_event_id, exc)
        task = asyncio.create_task(_redact_later())
        self._reaction_redaction_tasks.add(task)
        task.add_done_callback(self._reaction_redaction_tasks.discard)

    async def on_processing_start(self, event: MessageEvent) -> None:
        msg_id, room_id = event.message_id, event.source.chat_id
        if self._reactions_enabled and msg_id and room_id:
            reaction_event_id = await self._send_reaction(room_id, msg_id, "\U0001f440")
            if reaction_event_id:
                self._pending_reactions[(room_id, msg_id)] = reaction_event_id

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        msg_id, room_id = event.message_id, event.source.chat_id
        if not self._reactions_enabled or not msg_id or not room_id or outcome == ProcessingOutcome.CANCELLED:
            return
        eyes_event_id = self._pending_reactions.pop((room_id, msg_id), None)
        if eyes_event_id:
            self._schedule_reaction_redaction(room_id, eyes_event_id, "processing complete")
        await self._send_reaction(room_id, msg_id, "\u2705" if outcome == ProcessingOutcome.SUCCESS else "\u274c")

    async def _on_reaction(self, event: Any) -> None:
        sender = str(getattr(event, "sender", ""))
        if self._is_self_sender(sender):
            return
        event_id = str(getattr(event, "event_id", ""))
        if self._is_duplicate_event(event_id):
            return
        room_id = str(getattr(event, "room_id", ""))
        content = getattr(event, "content", None)
        if not content:
            return
        relates_to = (content.get("m.relates_to", {}) if isinstance(content, dict)
                      else getattr(content, "relates_to", {}))
        reacts_to = key = ""
        if isinstance(relates_to, dict):
            reacts_to = relates_to.get("event_id", "")
            key = relates_to.get("key", "")
        elif hasattr(relates_to, "event_id"):
            reacts_to = str(getattr(relates_to, "event_id", ""))
            key = str(getattr(relates_to, "key", ""))
        logger.info("Matrix: reaction %s from %s on %s in %s", key, sender, reacts_to, room_id)
        for handler in (self._handle_approval_reaction, self._handle_model_picker_reaction,
                        self._handle_choice_picker_reaction):
            if await handler(room_id, reacts_to, key, sender):
                return

    def _matrix_prompt_expired(self, prompt: Any) -> bool:
        expires_at = getattr(prompt, "expires_at", None)
        return expires_at is not None and time.monotonic() > float(expires_at)

    async def _validate_matrix_prompt_reactor(
        self, room_id: str, target_event_id: str, sender: str, prompt: Any, prompt_label: str) -> bool:
        if not self._is_authorized_user(sender):
            logger.info(
                "Matrix: ignoring %s reaction from unauthorized user %s on %s", prompt_label, sender, target_event_id)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, "Only an authorized Matrix user can use these controls.")
            return False
        requester = getattr(prompt, "requester_user_id", None)
        # getattr: object.__new__-built test doubles may lack the attribute.
        if getattr(self, "_approval_require_sender", True) and requester and sender != requester:
            logger.info("Matrix: ignoring %s reaction from %s; requester is %s", prompt_label, sender, requester)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, "Only the user who requested this action can use these controls.")
            return False
        return True

    async def _send_invalid_reaction_feedback(self, room_id: str, target_event_id: str, text: str) -> None:
        try:
            await self.send(room_id, text, reply_to=target_event_id)
        except Exception as exc:
            logger.debug("Matrix: failed to send invalid reaction feedback: %s", exc)

    async def _expire_matrix_approval_prompt(self, room_id: str, target_event_id: str, prompt: Any) -> None:
        prompt.resolved = True
        self._approval_prompts_by_event.pop(target_event_id, None)
        self._approval_prompt_by_session.pop(prompt.session_key, None)
        await self._redact_bot_approval_reactions(room_id, prompt)
        await self._send_invalid_reaction_feedback(
            room_id, target_event_id,
            "This approval prompt has expired. Run the command again if you still want to approve it.")

    async def _expire_matrix_model_picker_prompt(self, room_id: str, target_event_id: str, prompt: Any) -> None:
        prompt.resolved = True
        self._model_picker_prompts_by_event.pop(target_event_id, None)
        await self._redact_bot_model_picker_reactions(room_id, prompt)
        await self._send_invalid_reaction_feedback(
            room_id, target_event_id, "This model picker has expired. Run `/model` again to choose a model.")

    async def _redact_bot_approval_reactions(self, room_id: str, prompt: Any) -> None:
        """Redact the bot's seeded approval reactions (delayed), leaving only the user's reaction."""
        for emoji, evt_id in prompt.bot_reaction_events.items():
            self._schedule_reaction_redaction(room_id, evt_id, "approval resolved")
            logger.debug("Matrix: scheduled bot reaction redaction %s (%s)", emoji, evt_id)

    async def _redact_bot_model_picker_reactions(self, room_id: str, prompt: Any) -> None:
        for emoji, evt_id in prompt.bot_reaction_events.items():
            try:
                await self.redact_message(room_id, evt_id, "model picker resolved")
                logger.debug("Matrix: redacted model picker reaction %s (%s)", emoji, evt_id)
            except Exception as exc:
                logger.debug("Matrix: failed to redact model picker reaction %s: %s", emoji, exc)
