"""Matrix prompts methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Dict, Optional
from gateway.platforms.base import SendResult


class MatrixPromptsMixin:
    async def _send_reaction_prompt(
        self, chat_id: str, text: str, metadata: Optional[dict], make_prompt, registry: dict, emojis,
        label: str) -> SendResult:
        """Send *text*, register ``make_prompt(message_id, requester, expires_at)`` under
        the resulting event, then seed the bot's reaction controls (recording their IDs)."""
        from . import adapter as _adapter

        result = await self.send(chat_id, text, metadata=metadata)
        if not result.success or not result.message_id:
            return result
        prompt = make_prompt(
            result.message_id, str((metadata or {}).get("requester_user_id") or "") or None,
            _adapter.time.monotonic() + max(self._approval_timeout_seconds, 0))
        registry[result.message_id] = prompt
        for emoji in emojis:
            try:
                reaction_event_id = await self._send_reaction(chat_id, result.message_id, emoji)
                if reaction_event_id:
                    prompt.bot_reaction_events[emoji] = str(reaction_event_id)
            except Exception as exc:
                _adapter.logger.debug("Matrix: failed to add %s reaction %s: %s", label, emoji, exc)
        return result

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[dict] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False) -> SendResult:
        from . import adapter as _adapter

        if not self._client:
            return _adapter.SendResult(success=False, error="Not connected")
        if smart_denied:
            scope_choices = "Smart DENY: owner override applies to this one operation only.\n"
        else:
            scope_choices = (
                ("Reply `!approve session` to approve this pattern for the session, " if allow_session else "")
                + ("`!approve always` to approve permanently, " if allow_permanent else ""))
        legend = ["✅ = approve once"]
        reactions = ["✅"]
        if allow_session:
            legend.append("🌀 = approve for this session")
            reactions.append("🌀")
            if allow_permanent:
                legend.append("♾️ = approve always")
                reactions.append("♾️")
        legend.append("❎ = deny")
        reactions.append("❌")
        text = (
            f"{self._format_exec_approval(command, description)}\n\n"
            f"{scope_choices}Reply `!approve` to execute once, or `!deny` to cancel.\n\n"
            "You can also click the reaction to approve:\n" + "\n".join(legend))

        def _make(message_id, requester, expires_at):
            old_event = self._approval_prompt_by_session.get(session_key)
            if old_event:
                self._approval_prompts_by_event.pop(old_event, None)
            self._approval_prompt_by_session[session_key] = message_id
            return _adapter._MatrixApprovalPrompt(
                session_key=session_key, chat_id=chat_id, message_id=message_id, requester_user_id=requester,
                expires_at=expires_at)
        return await self._send_reaction_prompt(
            chat_id, text, metadata, _make, self._approval_prompts_by_event, tuple(reactions), "approval")

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str, session_key: str,
        on_model_selected, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        from . import adapter as _adapter

        if not self._client:
            return _adapter.SendResult(success=False, error="Not connected")
        flat_choices = [
            (str(model_id), str(p.get("slug") or ""), str(p.get("name") or p.get("slug") or ""))
            for p in providers or [] for model_id in (p.get("models") or [])][:len(_adapter._MATRIX_MODEL_PICKER_REACTIONS)]
        if not flat_choices:
            return await self.send(
                chat_id, "No authenticated models are available for this session.", metadata=metadata)
        try:
            from hermes_cli.providers import get_label
            provider_label = get_label(current_provider)
        except Exception:
            provider_label = current_provider
        lines = [
            "⚙ **Model Configuration**", f"Current model: `{current_model or 'unknown'}`",
            f"Provider: {provider_label or 'unknown'}", "", "React to choose a model:"]
        choices: dict[str, tuple[str, str]] = {}
        for emoji, (model_id, provider_slug, provider_name) in zip(_adapter._MATRIX_MODEL_PICKER_REACTIONS, flat_choices):
            choices[emoji] = (model_id, provider_slug)
            lines.append(f"{emoji} `{model_id}` — {provider_name}")
        return await self._send_picker(
            chat_id, lines, choices, session_key, on_model_selected, metadata, self._model_picker_prompts_by_event,
            "model picker")

    async def _send_picker(
        self, chat_id: str, lines: list, choices: dict, session_key: str, on_selected, metadata, registry: dict,
        label: str) -> SendResult:
        """Send picker *lines*, register a _MatrixPickerPrompt under the event, seed its reactions."""
        from . import adapter as _adapter

        return await self._send_reaction_prompt(
            chat_id, "\n".join(lines), metadata,
            lambda message_id, requester, expires_at: _adapter._MatrixPickerPrompt(
                chat_id=chat_id, message_id=message_id, session_key=session_key, choices=choices,
                on_selected=on_selected, requester_user_id=requester, expires_at=expires_at),
            registry, choices, label)

    supports_choice_pages = True
    choice_pages_edit_in_place = False

    async def send_choice_picker(
        self,
        chat_id: str,
        title: str,
        choices: list,
        session_key: str,
        on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Matrix reaction-based choice picker (/reasoning, /fast).

        Generic single-level companion to ``send_model_picker``. Each choice
        dict: ``{"value": str, "label": str, "is_current": bool}``.
        """
        from . import adapter as _adapter
        from . import choice_picker as _choice_picker

        return await _choice_picker.send_choice_picker(
            self, chat_id, title, choices, session_key, on_choice_selected,
            metadata, _adapter._MATRIX_CHOICE_PICKER_REACTIONS,
        )

    async def _on_reaction(self, event: Any) -> None:
        from . import adapter as _adapter

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
        _adapter.logger.info("Matrix: reaction %s from %s on %s in %s", key, sender, reacts_to, room_id)
        for handler in (self._handle_approval_reaction, self._handle_model_picker_reaction):
            if await handler(room_id, reacts_to, key, sender):
                return
        await self._handle_choice_picker_reaction(room_id, reacts_to, key, sender, event_id=event_id)

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
            await self._send_invalid_reaction_feedback(room_id, reacts_to, invalid_text)
        return True, prompt, selection

    async def _handle_approval_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Resolve a pending exec-approval prompt from a reaction. True if it was the target."""
        from . import adapter as _adapter

        handled, prompt, choice = await self._claim_reaction_prompt(
            self._approval_prompts_by_event, room_id, reacts_to, key, sender, "approval",
            "That reaction is not valid for this approval prompt.", self._expire_matrix_approval_prompt,
            choices=self._approval_reaction_map)
        if choice is None:
            return handled
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(prompt.session_key, choice)
            if count:
                prompt.resolved = True
                self._approval_prompts_by_event.pop(reacts_to, None)
                self._approval_prompt_by_session.pop(prompt.session_key, None)
                _adapter.logger.info(
                    "Matrix reaction resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, prompt.session_key, choice, sender)
                await self._redact_bot_approval_reactions(room_id, prompt)
        except Exception as exc:
            _adapter.logger.error("Failed to resolve gateway approval from Matrix reaction: %s", exc)
        return True

    async def _handle_model_picker_reaction(self, room_id: str, reacts_to: str, key: str, sender: str) -> bool:
        """Apply a model-picker reaction. True if the reaction targeted a pending picker."""
        return await self._handle_picker_reaction(
            self._model_picker_prompts_by_event, room_id, reacts_to, key, sender, "model picker",
            "That reaction is not one of the available model choices.", self._expire_matrix_model_picker_prompt,
            ("switch model", "switch model"), redact_bot_reactions=True)

    async def _handle_choice_picker_reaction(
        self, room_id: str, reacts_to: str, key: str, sender: str, *, event_id: str = ""
    ) -> bool:
        """Apply a choice-picker reaction. True if the reaction targeted a pending picker."""
        from . import adapter as _adapter
        from .choice_picker import handle_choice_reaction

        if reacts_to not in self._choice_picker_prompts_by_event:
            return False
        await handle_choice_reaction(
            self, room_id, reacts_to, sender, key, event_id, _adapter._MATRIX_CHOICE_PICKER_REACTIONS
        )
        return True

    async def _handle_picker_reaction(
        self, registry: dict, room_id: str, reacts_to: str, key: str, sender: str, label: str, invalid_text: str,
        on_expired, verbs: tuple[str, str], *, redact_bot_reactions: bool = False) -> bool:
        """Claim the picker, fire ``on_selected(room_id, *selection)`` and post its confirmation (or the error).
        ``verbs`` = (log verb, user-facing verb)."""
        from . import adapter as _adapter

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
                await self.send(room_id, confirmation, reply_to=reacts_to)
        except Exception as exc:
            _adapter.logger.error("Failed to %s from Matrix reaction: %s", verbs[0], exc)
            await self.send(room_id, f"Failed to {verbs[1]}: {exc}", reply_to=reacts_to)
        return True

    def _matrix_prompt_expired(self, prompt: Any) -> bool:
        from . import adapter as _adapter

        expires_at = getattr(prompt, "expires_at", None)
        return expires_at is not None and _adapter.time.monotonic() > float(expires_at)

    def _is_authorized_user(self, user_id: str) -> bool:
        """GATEWAY_ALLOW_ALL_USERS, or membership in MATRIX_ALLOWED_USERS."""
        from . import adapter as _adapter

        return _adapter._env_truthy("GATEWAY_ALLOW_ALL_USERS") or bool(
            self._allowed_user_ids and user_id in self._allowed_user_ids)

    async def _validate_matrix_prompt_reactor(
        self, room_id: str, target_event_id: str, sender: str, prompt: Any, prompt_label: str) -> bool:
        from . import adapter as _adapter

        if not self._is_authorized_user(sender):
            _adapter.logger.info(
                "Matrix: ignoring %s reaction from unauthorized user %s on %s", prompt_label, sender, target_event_id)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, "Only an authorized Matrix user can use these controls.")
            return False
        requester = getattr(prompt, "requester_user_id", None)
        # getattr: object.__new__-built test doubles may lack the attribute.
        if getattr(self, "_approval_require_sender", True) and requester and sender != requester:
            _adapter.logger.info("Matrix: ignoring %s reaction from %s; requester is %s", prompt_label, sender, requester)
            await self._send_invalid_reaction_feedback(
                room_id, target_event_id, "Only the user who requested this action can use these controls.")
            return False
        return True

    async def _send_invalid_reaction_feedback(self, room_id: str, target_event_id: str, text: str) -> None:
        from . import adapter as _adapter

        try:
            await self.send(room_id, text, reply_to=target_event_id)
        except Exception as exc:
            _adapter.logger.debug("Matrix: failed to send invalid reaction feedback: %s", exc)

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
        from . import adapter as _adapter

        for emoji, evt_id in prompt.bot_reaction_events.items():
            self._schedule_reaction_redaction(room_id, evt_id, "approval resolved")
            _adapter.logger.debug("Matrix: scheduled bot reaction redaction %s (%s)", emoji, evt_id)

    async def _redact_bot_model_picker_reactions(self, room_id: str, prompt: Any) -> None:
        from . import adapter as _adapter

        for emoji, evt_id in prompt.bot_reaction_events.items():
            try:
                await self.redact_message(room_id, evt_id, "model picker resolved")
                _adapter.logger.debug("Matrix: redacted model picker reaction %s (%s)", emoji, evt_id)
            except Exception as exc:
                _adapter.logger.debug("Matrix: failed to redact model picker reaction %s: %s", emoji, exc)
