"""Matrix picker state and shared reaction control feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from agent.i18n import t

if TYPE_CHECKING:
    from gateway.platforms.base import SendResult


@dataclass
class _MatrixPickerPrompt:
    """Pending reaction-based picker; ``choices`` maps emoji -> selection, ``on_selected`` is the callback."""
    chat_id: str
    message_id: str
    session_key: str
    choices: dict
    on_selected: Any
    requester_user_id: str | None = None
    expires_at: float | None = None
    resolved: bool = False
    metadata: dict = field(default_factory=dict)
    bot_reaction_events: dict[str, str] = field(default_factory=dict)

    @property
    def notice_metadata(self) -> dict[str, str] | None:
        """Send metadata that keeps a notice about this card in the card's thread."""
        thread_id = str(self.metadata.get("thread_id") or "")
        return {"thread_id": thread_id} if thread_id else None


class MatrixReactionControlMixin:
    """Process registered picker choices and shared reaction feedback."""

    if TYPE_CHECKING:
        _model_picker_prompts_by_event: dict
        _choice_picker_prompts_by_event: dict
        _send_reaction_prompt: Any
        _is_authorized_user: Any
        _matrix_prompt_expired: Any
        send: Any
        redact_message: Any

    async def _send_picker(
        self, chat_id: str, lines: list, choices: dict, session_key: str, on_selected, metadata, registry: dict,
        label: str) -> SendResult:
        """Send picker *lines*, register a _MatrixPickerPrompt under the event, seed its reactions."""
        return await self._send_reaction_prompt(
            chat_id, "\n".join(lines), metadata,
            lambda message_id, requester, expires_at: _MatrixPickerPrompt(
                chat_id=chat_id, message_id=message_id, session_key=session_key, choices=choices,
                on_selected=on_selected, requester_user_id=requester, expires_at=expires_at,
                metadata=dict(metadata or {})),
            registry, choices, label)

    async def _claim_reaction_prompt(
        self, registry: dict, room_id: str, reacts_to: str, key: str, sender: str, label: str, invalid_text: str,
        on_expired, choices: Optional[dict] = None) -> tuple[bool, Any, Any]:
        """Shared gate for reaction prompts: (handled, prompt, selection). handled=False => not our
        prompt; selection=None with handled=True => consumed without action (wrong room, expired,
        unauthorized reactor, or a key that is not a choice). ``choices`` defaults to ``prompt.choices``."""
        prompt = registry.get(reacts_to)
        if not prompt or prompt.resolved:
            return False, None, None
        if room_id != prompt.chat_id:
            return True, prompt, None
        if self._matrix_prompt_expired(prompt):
            await on_expired(room_id, reacts_to, prompt)
            return True, prompt, None
        if not await self._validate_matrix_prompt_reactor(room_id, reacts_to, sender, prompt, label):
            return True, prompt, None
        selection = (prompt.choices if choices is None else choices).get(key)
        if selection is None:
            await self._send_invalid_reaction_feedback(room_id, reacts_to, invalid_text, prompt.notice_metadata)
        return True, prompt, selection

    async def _handle_model_picker_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Apply a model-picker reaction. True if the reaction targeted a pending picker."""
        return await self._handle_picker_reaction(
            self._model_picker_prompts_by_event, room_id, reacts_to, key, sender, "model picker",
            t("platform.matrix.picker.invalid_model_reaction"), self._expire_matrix_model_picker_prompt,
            ("switch model", "platform.matrix.picker.verb_switch_model"), redact_bot_reactions=True)

    async def _handle_choice_picker_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Apply a choice-picker reaction. True if the reaction targeted a pending picker."""
        async def _expire(_room_id, target_event_id, _prompt):
            self._choice_picker_prompts_by_event.pop(target_event_id, None)
        return await self._handle_picker_reaction(
            self._choice_picker_prompts_by_event, room_id, reacts_to, key, sender, "choice picker",
            t("platform.matrix.picker.invalid_choice_reaction"), _expire,
            ("apply choice", "platform.matrix.picker.verb_apply_selection"))

    async def _handle_picker_reaction(
        self, registry: dict, room_id: str, reacts_to: str, key: str, sender: str, label: str, invalid_text: str,
        on_expired, verbs: tuple[str, str], *, redact_bot_reactions: bool = False) -> bool:
        """Claim the picker, fire ``on_selected(room_id, *selection)`` and post its confirmation (or the error).
        ``verbs`` = (log verb, catalog key of the user-facing verb)."""
        from plugins.platforms.matrix.adapter import logger

        handled, prompt, selection = await self._claim_reaction_prompt(
            registry, room_id, reacts_to, key, sender, label, invalid_text, on_expired)
        if selection is None:
            return handled
        prompt.resolved = True
        registry.pop(reacts_to, None)
        args = selection if isinstance(selection, tuple) else (selection,)
        try:
            confirmation = await prompt.on_selected(room_id, *args)
            if redact_bot_reactions:
                await self._redact_bot_model_picker_reactions(room_id, prompt)
            if confirmation:
                await self.send(room_id, confirmation, reply_to=reacts_to, metadata=prompt.notice_metadata)
        except Exception as exc:
            logger.error("Failed to %s from Matrix reaction: %s", verbs[0], exc)
            await self.send(room_id, t("platform.matrix.picker.failed", action=t(verbs[1]), error=str(exc)),
                            reply_to=reacts_to, metadata=prompt.notice_metadata)
        return True

    async def _validate_matrix_prompt_reactor(
        self, room_id: str, target_event_id: str, sender: str, prompt: Any, prompt_label: str) -> bool:
        from plugins.platforms.matrix.adapter import logger

        if not self._is_authorized_user(sender, room_id):
            logger.info(
                "Matrix: ignoring %s reaction from unauthorized user %s on %s", prompt_label, sender, target_event_id)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, t("platform.matrix.reaction.unauthorized"), prompt.notice_metadata)
            return False
        requester = getattr(prompt, "requester_user_id", None)
        # getattr: object.__new__-built test doubles may lack the attribute.
        if getattr(self, "_approval_require_sender", True) and requester and sender != requester:
            logger.info("Matrix: ignoring %s reaction from %s; requester is %s", prompt_label, sender, requester)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, t("platform.matrix.reaction.not_requester"), prompt.notice_metadata)
            return False
        return True

    async def _send_invalid_reaction_feedback(
            self, room_id: str, target_event_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        from plugins.platforms.matrix.adapter import logger

        try:
            result = await self.send(room_id, text, reply_to=target_event_id, metadata=metadata)
            return bool(result and result.success and result.message_id)
        except Exception as exc:
            logger.debug("Matrix: failed to send invalid reaction feedback: %s", exc)
            return False

    async def _expire_matrix_model_picker_prompt(self, room_id: str, target_event_id: str, prompt: Any) -> None:
        prompt.resolved = True
        self._model_picker_prompts_by_event.pop(target_event_id, None)
        await self._redact_bot_model_picker_reactions(room_id, prompt)
        await self._send_invalid_reaction_feedback(
            room_id, target_event_id, t("platform.shared.model_picker_expired"), prompt.notice_metadata)

    async def _redact_bot_model_picker_reactions(self, room_id: str, prompt: Any) -> None:
        from plugins.platforms.matrix.adapter import logger

        for emoji, evt_id in prompt.bot_reaction_events.items():
            try:
                await self.redact_message(room_id, evt_id, "model picker resolved")
                logger.debug("Matrix: redacted model picker reaction %s (%s)", emoji, evt_id)
            except Exception as exc:
                logger.debug("Matrix: failed to redact model picker reaction %s: %s", emoji, exc)
