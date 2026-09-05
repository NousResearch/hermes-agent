"""Slack prompts methods; SDK and mutable dependencies remain on the facade."""

from typing import Any, Callable, Dict, List, Optional, Tuple
from gateway.platforms.base import SendResult
try:
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    AsyncApp = AsyncWebClient = Any


class SlackPromptsMixin:
    @staticmethod
    def _button(
        text: str, action_id: str, value: str, *, style: str = "", emoji: bool = False) -> dict:
        """Block Kit button element; ``style``/``emoji`` keys only present when set."""
        from . import adapter as _adapter

        text_obj: _adapter.Dict[str, _adapter.Any] = {"type": "plain_text", "text": text}
        if emoji:
            text_obj["emoji"] = True
        btn: _adapter.Dict[str, _adapter.Any] = {"type": "button", "text": text_obj}
        if style:
            btn["style"] = style
        btn["action_id"] = action_id
        btn["value"] = value
        return btn

    async def _post_interactive_blocks(
        self, chat_id: str, text: str, blocks: list, metadata: Optional[Dict[str, Any]], *,
        sanitize: bool = True, team_scoped: bool = True):
        """chat.postMessage with ``blocks`` (threaded via metadata); returns the raw response."""
        from . import adapter as _adapter

        kwargs: _adapter.Dict[str, _adapter.Any] = {
            "channel": chat_id, "text": text,
            "blocks": _adapter.sanitize_blocks(blocks) if sanitize else blocks}
        thread_ts = self._resolve_thread_ts(None, metadata)
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        team_id = self._metadata_team_id(metadata) if team_scoped else None
        return await self._get_client(chat_id, team_id=team_id).chat_postMessage(**kwargs)

    async def _send_interactive_prompt(
        self, chat_id: str, metadata: Optional[Dict[str, Any]],
        build: Callable[[], Tuple[str, list]], label: str, *,
        resolved: Optional[Dict[Any, bool]] = None, resolved_max: int = 0,
        team_scoped_key: bool = True, sanitize: bool = True) -> SendResult:
        """Shared body of the Block Kit prompt senders: DM-resolve, ``build()`` -> ``(fallback
        text, blocks)``, post, then mark the message unresolved in ``resolved`` (double-click
        guard). Any failure is logged as ``<label> failed`` and returned, never raised."""
        from . import adapter as _adapter

        if not self._app:
            return _adapter.SendResult(success=False, error="Not connected")
        chat_id = await self._dm_target(chat_id, metadata)
        try:
            text, blocks = build()
            result = await self._post_interactive_blocks(
                chat_id, text, blocks, metadata, sanitize=sanitize, team_scoped=team_scoped_key)
            msg_ts = result.get("ts", "")
            if msg_ts and resolved is not None:
                key = msg_ts
                if team_scoped_key:
                    key = self._workspace_message_marker(self._metadata_team_id(metadata), msg_ts)
                resolved[key] = False
                self._trim_oldest_dict_entries(resolved, resolved_max)
            return _adapter.SendResult(success=True, message_id=msg_ts, raw_response=result)
        except Exception as e:
            _adapter.logger.error("[Slack] %s failed: %s", label, e, exc_info=True)
            return _adapter.SendResult(success=False, error=str(e))

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None, allow_permanent: bool = True,
        allow_session: bool = True, smart_denied: bool = False) -> SendResult:
        """Send a Block Kit approval prompt with interactive buttons.
        The buttons call ``resolve_gateway_approval()`` to unblock the waiting agent thread — same
        mechanism as the text ``/approve`` flow."""

        def _build() -> Tuple[str, list]:
            # Slack caps a section's text at 3000 chars (overflow → invalid_blocks → no buttons);
            # execute_code approvals embed the whole script, so budget the preview.
            header = ":warning: *Command Approval Required*\n"
            if smart_denied:
                header += "*Smart DENY:* owner override applies to this one operation only.\n"
            reason = f"Reason: {description[:500]}"
            budget = 3000 - len(header) - len(reason) - len("``````\n") - len("...")
            cmd_preview = command[:budget] + "..." if len(command) > budget else command
            actions = [
                self._button("Allow Once", "hermes_approve_once", session_key, style="primary")]
            if not smart_denied and allow_session:
                actions.append(self._button("Allow Session", "hermes_approve_session", session_key))
                if allow_permanent:
                    actions.append(
                        self._button("Always Allow", "hermes_approve_always", session_key))
            actions.append(self._button("Deny", "hermes_deny", session_key, style="danger"))
            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"{header}```{cmd_preview}```\n{reason}"}},
                {"type": "actions", "elements": actions}]
            return f"⚠️ Command approval required: {cmd_preview[:100]}", blocks

        return await self._send_interactive_prompt(
            chat_id, metadata, _build, "send_exec_approval",
            resolved=self._approval_resolved, resolved_max=self._APPROVAL_RESOLVED_MAX)

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a Block Kit three-option slash-command confirmation prompt."""

        def _build() -> Tuple[str, list]:
            # Same 3000-char section cap as send_exec_approval: budget the body
            # against the rendered title.
            _title = (title or "Confirm")[:150]
            budget = 3000 - len(f"*{_title}*\n\n") - len("...")
            body = message[:budget] + "..." if len(message) > budget else message
            # session_key|confirm_id in the button value lets the callback resolve
            # without extra bookkeeping.
            value = f"{session_key}|{confirm_id}"
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*{_title}*\n\n{body}"}},
                {
                    "type": "actions",
                    "elements": [
                        self._button("Approve Once", "hermes_confirm_once", value, style="primary"),
                        self._button("Always Approve", "hermes_confirm_always", value),
                        self._button("Cancel", "hermes_confirm_cancel", value, style="danger")]}]
            return f"{title or 'Confirm'}: {body[:100]}", blocks

        return await self._send_interactive_prompt(chat_id, metadata, _build, "send_slash_confirm")

    def _build_model_picker_provider_blocks(
        self, providers: list, current_model: str, provider_label: str
    ) -> List[dict]:
        """Build the provider-select stage of the model picker.

        A section header (current model/provider) plus an actions block with a
        ``static_select`` of providers and a Cancel button. Provider option
        ``value`` carries the list index (same scheme as the model stage) so
        an over-long custom provider slug never trips Slack's 75-char option
        value cap — the handler resolves the real slug from picker state.
        """
        from . import adapter as _adapter

        options = []
        for idx, p in enumerate(providers[:100]):
            count = p.get("total_models", len(p.get("models", [])))
            options.append({
                "text": {
                    "type": "plain_text",
                    "text": f"{p['name']} ({count} models)"[:75],
                    "emoji": True,
                },
                "value": str(idx),
            })
        extra = (
            f"\n*{len(providers) - 100} more available — type `/model <name>` directly*"
            if len(providers) > 100
            else ""
        )
        section_text = (
            f"*⚙ Model Configuration*\n"
            f"Current model: `{current_model or 'unknown'}`\n"
            f"Provider: {provider_label}\n\n"
            f"Select a provider:{extra}"
        )
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": section_text[:3000]}},
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "static_select",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Choose a provider…",
                            "emoji": True,
                        },
                        "action_id": _adapter._MODEL_PICKER_PROVIDER_ACTION,
                        "options": options,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Cancel", "emoji": True},
                        "style": "danger",
                        "action_id": _adapter._MODEL_PICKER_CANCEL_ACTION,
                        "value": "cancel",
                    },
                ],
            },
        ]

    def _build_model_picker_model_blocks(
        self, providers: list, provider_slug: str
    ) -> List[dict]:
        """Build the model-select stage for a chosen provider.

        A section header (provider name) plus an actions block with a
        ``static_select`` of models and Back/Cancel buttons. Model option
        ``value`` carries the list index so over-long model IDs never trip
        Slack's 75-char value cap; the handler resolves the real model ID
        from the provider's model list in picker state.
        """
        from . import adapter as _adapter

        provider = next((p for p in providers if p["slug"] == provider_slug), None)
        pname = provider.get("name", provider_slug) if provider else provider_slug
        models = (provider or {}).get("models", [])[:100]
        options = []
        for idx, model_id in enumerate(models):
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            options.append({
                "text": {"type": "plain_text", "text": short[:75], "emoji": True},
                "value": str(idx),
            })
        total = (provider or {}).get("total_models", len(models))
        extra = (
            f"\n*{total - len(models)} more available — type `/model <name>` directly*"
            if total > len(models)
            else ""
        )
        section_text = f"*⚙ Model Configuration*\n\nProvider: *{pname}*\nSelect a model:{extra}"
        elements = [
            {
                "type": "static_select",
                "placeholder": {
                    "type": "plain_text",
                    "text": f"Choose a model from {pname}…"[:150],
                    "emoji": True,
                },
                "action_id": _adapter._MODEL_PICKER_MODEL_ACTION,
                "options": options,
            },
        ]
        if provider_slug:
            elements.append({
                "type": "button",
                "text": {"type": "plain_text", "text": "◀ Back", "emoji": True},
                "action_id": _adapter._MODEL_PICKER_BACK_ACTION,
                "value": provider_slug,
            })
        elements.append({
            "type": "button",
            "text": {"type": "plain_text", "text": "Cancel", "emoji": True},
            "style": "danger",
            "action_id": _adapter._MODEL_PICKER_CANCEL_ACTION,
            "value": "cancel",
        })
        return [
            {"type": "section", "text": {"type": "mrkdwn", "text": section_text[:3000]}},
            {"type": "actions", "elements": elements},
        ]

    async def send_model_picker(
        self,
        chat_id: str,
        providers: list,
        current_model: str,
        current_provider: str,
        session_key: str,
        on_model_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an interactive Block Kit model picker.

        Two-step drill-down: provider ``static_select`` → model
        ``static_select``, with Back/Cancel buttons. Resolves via
        ``_handle_model_picker_action``, which calls ``on_model_selected`` on
        a model choice.
        """
        from . import adapter as _adapter

        if not self._app:
            return SendResult(success=False, error="Not connected")

        chat_id = await self._ensure_dm_conversation(
            chat_id, team_id=self._metadata_team_id(metadata)
        )
        try:
            thread_ts = self._resolve_thread_ts(None, metadata)

            try:
                from hermes_cli.providers import get_label
                provider_label = get_label(current_provider)
            except Exception:
                provider_label = current_provider

            if not providers:
                return SendResult(success=False, error="No providers available")

            blocks = self._build_model_picker_provider_blocks(
                providers, current_model, provider_label
            )

            kwargs: Dict[str, Any] = {
                "channel": chat_id,
                "text": "⚙ Model Configuration — select a provider",
                "blocks": _adapter.sanitize_blocks(blocks),
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts

            result = await self._get_client(
                chat_id, team_id=self._metadata_team_id(metadata)
            ).chat_postMessage(**kwargs)
            msg_ts = result.get("ts", "")
            if not msg_ts:
                return SendResult(success=False, error="No message timestamp returned")

            team_id = self._metadata_team_id(metadata)
            self._model_picker_state[
                self._workspace_message_marker(team_id, msg_ts)
            ] = {
                "providers": providers,
                "session_key": session_key,
                "chat_id": chat_id,
                "team_id": team_id,
                "current_model": current_model,
                "current_provider": current_provider,
                "on_model_selected": on_model_selected,
                "stage": "provider",
                "selected_provider_slug": "",
            }
            self._trim_oldest_dict_entries(
                self._model_picker_state, self._MODEL_PICKER_STATE_MAX
            )

            return SendResult(success=True, message_id=msg_ts, raw_response=result)
        except Exception as e:
            _adapter.logger.error("[Slack] send_model_picker failed: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def _update_picker_message(
        self,
        channel_id: str,
        team_id: str,
        msg_ts: str,
        section_text: str,
    ) -> None:
        """Replace the picker message body with a plain section (no controls)."""
        from . import adapter as _adapter

        try:
            await self._get_client(channel_id, team_id=team_id or None).chat_update(
                channel=channel_id,
                ts=msg_ts,
                text=section_text[:3000],
                blocks=_adapter.sanitize_blocks([
                    {"type": "section", "text": {"type": "mrkdwn", "text": section_text[:3000]}},
                ]),
            )
        except Exception as e:
            _adapter.logger.warning("[Slack] Failed to update model picker message: %s", e)

    async def _handle_model_picker_action(self, ack, body, action) -> None:
        """Handle a model picker Block Kit interaction.

        Dispatches on the action_id: provider static_select advances to the
        model stage, model static_select runs ``on_model_selected``, Back
        returns to the provider stage, Cancel dismisses the picker.
        """
        from . import adapter as _adapter

        await ack()

        team_id = self._event_team_id({}, body)
        action_id = action.get("action_id", "")
        message = body.get("message", {})
        msg_ts = message.get("ts", "")
        channel_id = body.get("channel", {}).get("id", "")
        user_name = body.get("user", {}).get("name", "unknown")
        user_id = body.get("user", {}).get("id", "")

        if not self._is_interactive_user_authorized(
            user_id,
            channel_id=channel_id,
            user_name=user_name,
            team_id=team_id,
        ):
            _adapter.logger.warning(
                "[Slack] Unauthorized model picker click by %s (%s) - ignoring",
                user_name, user_id,
            )
            return

        # Look up the picker state. The send path may have stored it under a
        # bare ts (metadata-poor send, no team id) while this click event
        # carries a team id — that mismatch must not swallow a legitimate
        # interaction (mirrors _handle_approval_action's dual-key lookup).
        marker = self._workspace_message_marker(team_id, msg_ts)
        if msg_ts in self._model_picker_state:
            marker = msg_ts
        state = self._model_picker_state.get(marker)
        if not state:
            _adapter.logger.debug("[Slack] Model picker state not found for marker=%s", marker)
            # Gateway restarted or the entry aged out of the bounded dict —
            # there is no gateway-side registry to fall back on, so this
            # dict is the picker's only state. Kill the live-looking
            # control visibly instead of silently swallowing clicks
            # (mirrors the clarify handler's expiry notice).
            await self._update_picker_message(
                channel_id, team_id, msg_ts, _adapter._MODEL_PICKER_EXPIRED_NOTICE
            )
            return

        providers = state.get("providers", [])
        on_model_selected = state.get("on_model_selected")

        # Cancel → dismiss.
        if action_id == _adapter._MODEL_PICKER_CANCEL_ACTION:
            self._model_picker_state.pop(marker, None)
            await self._update_picker_message(
                channel_id, team_id, msg_ts, "❌ Model selection cancelled."
            )
            return

        # Provider selected → advance to model stage. The option value is a
        # list index into the stored providers slice (never the raw slug —
        # custom slugs can exceed Slack's 75-char option value cap).
        if action_id == _adapter._MODEL_PICKER_PROVIDER_ACTION:
            selected = action.get("selected_option", {})
            idx_token = selected.get("value", "")
            try:
                idx = int(idx_token)
                provider = providers[idx] if idx >= 0 else None
            except (ValueError, IndexError, TypeError):
                provider = None
            if provider is None:
                # Message and stored state are out of sync (stale payload,
                # re-seeded entry) — the picker can no longer resolve, so
                # kill it visibly like the expiry path.
                _adapter.logger.warning(
                    "[Slack] Invalid provider picker index token: %r", idx_token
                )
                self._model_picker_state.pop(marker, None)
                await self._update_picker_message(
                    channel_id, team_id, msg_ts, _adapter._MODEL_PICKER_EXPIRED_NOTICE
                )
                return
            provider_slug = provider.get("slug", "")
            if not provider.get("models"):
                await self._update_picker_message(
                    channel_id, team_id, msg_ts,
                    f"No models available for `{provider_slug}`.",
                )
                self._model_picker_state.pop(marker, None)
                return

            state["stage"] = "model"
            state["selected_provider_slug"] = provider_slug
            blocks = self._build_model_picker_model_blocks(providers, provider_slug)
            try:
                await self._get_client(channel_id, team_id=team_id or None).chat_update(
                    channel=channel_id,
                    ts=msg_ts,
                    text=f"⚙ Model Configuration — {provider.get('name', provider_slug)}",
                    blocks=_adapter.sanitize_blocks(blocks),
                )
            except Exception as e:
                _adapter.logger.warning(
                    "[Slack] Failed to update model picker (provider→model): %s", e
                )
            return

        # Back → return to provider stage.
        if action_id == _adapter._MODEL_PICKER_BACK_ACTION:
            state["stage"] = "provider"
            state["selected_provider_slug"] = ""
            try:
                from hermes_cli.providers import get_label
                provider_label = get_label(
                    state.get("current_provider", "")
                )
            except Exception:
                provider_label = state.get("current_provider", "")
            blocks = self._build_model_picker_provider_blocks(
                providers, state.get("current_model", ""), provider_label
            )
            try:
                await self._get_client(channel_id, team_id=team_id or None).chat_update(
                    channel=channel_id,
                    ts=msg_ts,
                    text="⚙ Model Configuration — select a provider",
                    blocks=_adapter.sanitize_blocks(blocks),
                )
            except Exception as e:
                _adapter.logger.warning("[Slack] Failed to update model picker (back): %s", e)
            return

        # Model selected → run the switch.
        if (
            action_id == _adapter._MODEL_PICKER_MODEL_ACTION
            and state.get("stage") == "model"
        ):
            selected = action.get("selected_option", {})
            idx_token = selected.get("value", "")
            provider_slug = state.get("selected_provider_slug", "")
            provider = next((p for p in providers if p["slug"] == provider_slug), None)
            models = (provider or {}).get("models", [])
            try:
                idx = int(idx_token)
                model_id = models[idx] if idx >= 0 else None
            except (ValueError, IndexError, TypeError):
                model_id = None
            if model_id is None:
                # Message and stored state are out of sync — kill the picker
                # visibly instead of leaving a dead control.
                _adapter.logger.warning("[Slack] Invalid model picker index token: %r", idx_token)
                self._model_picker_state.pop(marker, None)
                await self._update_picker_message(
                    channel_id, team_id, msg_ts, _adapter._MODEL_PICKER_EXPIRED_NOTICE
                )
                return

            if not on_model_selected:
                self._model_picker_state.pop(marker, None)
                await self._update_picker_message(
                    channel_id, team_id, msg_ts, _adapter._MODEL_PICKER_EXPIRED_NOTICE
                )
                return

            # Pop the state up-front (double-click guard, mirrors approval).
            self._model_picker_state.pop(marker, None)
            await self._update_picker_message(
                channel_id, team_id, msg_ts, f"⚙ Switching to `{model_id}`…"
            )

            switch_failed = False
            try:
                confirmation = await on_model_selected(
                    state["chat_id"], model_id, provider_slug
                )
                # The gateway reports a failed in-place swap as a localized
                # error-prefixed return string, not an exception (#50163).
                # Compare against the same i18n prefix so both failure
                # shapes get the failed header.
                try:
                    from agent.i18n import t as _t

                    _error_prefix = _t("gateway.model.error_prefix", error="").strip()
                except Exception:
                    _error_prefix = "Error:"
                if _error_prefix and str(confirmation).startswith(_error_prefix):
                    switch_failed = True
            except Exception as exc:
                _adapter.logger.error(
                    "[Slack] Model picker callback failed: %s", exc, exc_info=True
                )
                confirmation = f"❌ Model switch failed: {exc}"
                switch_failed = True

            header = "⚙ Model Switch Failed" if switch_failed else "⚙ Model Switched"
            await self._update_picker_message(
                channel_id, team_id, msg_ts, f"{header}\n\n{confirmation}"
            )
            return

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Clarify prompt as Block Kit buttons: one ``hermes_clarify_choice_<idx>`` per option
        (value ``clarify_id|idx``) plus "✏️ Other…" (``hermes_clarify_other``), which flips the
        entry into text-capture mode for the gateway's text-intercept. No choices → base impl."""
        if not choices:
            return await super().send_clarify(
                chat_id=chat_id, question=question, choices=choices, clarify_id=clarify_id,
                session_key=session_key, metadata=metadata)

        def _build() -> Tuple[str, list]:
            # Escape mrkdwn control chars so the question renders literally;
            # budget against the 3000-char section cap.
            q = (question or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            body = f"❓ {q}"
            budget = 3000 - len("...")
            if len(body) > budget:
                body = body[:budget] + "..."
            # Slack caps an actions block at 5 elements; clarify caps choices at 4 (+ Other) but
            # chunk anyway so larger lists degrade gracefully instead of 400ing.
            elements = []
            for idx, choice in enumerate(choices):
                label = str(choice).strip() or f"Option {idx + 1}"
                elements.append(
                    self._button(
                        label[:75], f"hermes_clarify_choice_{idx}",
                        f"{clarify_id}|{idx}", emoji=True))
            elements.append(
                self._button("✏️ Other…", "hermes_clarify_other", f"{clarify_id}|other", emoji=True)
            )
            blocks: list = [{"type": "section", "text": {"type": "mrkdwn", "text": body}}]
            for start in range(0, len(elements), 5):
                blocks.append({"type": "actions", "elements": elements[start : start + 5]})
            return body, blocks

        # Bare-ts key (not workspace-scoped) so the action handler's atomic-pop guard
        # can reject double-clicks (mirrors _approval_resolved).
        return await self._send_interactive_prompt(
            chat_id, metadata, _build, "send_clarify",
            resolved=self._clarify_resolved, resolved_max=self._CLARIFY_RESOLVED_MAX,
            team_scoped_key=False, sanitize=False)

    def _is_interactive_user_authorized(
        self, user_id: str, *, channel_id: str = "", user_name: Optional[str] = None,
        team_id: str = "") -> bool:
        """Return whether a Slack interactive caller may perform gated actions."""
        from . import adapter as _adapter

        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False
        chat_type = "dm" if str(channel_id or "").startswith("D") else "group"
        # Preferred: the injected profile-bound check (``set_authorization_check``); unlike the
        # ``__self__`` introspection below it works under multiplex (handler is a closure).
        # getattr: object.__new__ test doubles never ran BasePlatformAdapter.__init__.
        # Preferred path: the auth callback GatewayRunner injects at connect time
        # (``set_authorization_check``) runs the full, profile-bound ``_is_user_authorized`` chain. Unlike
        # the ``__self__`` introspection below it also resolves on a multiplexed adapter, whose message
        # handler is a profile closure with no ``__self__`` (#72657, same class as Telegram's #86296).
        if getattr(self, "_authorization_check", None) is not None:
            injected = self._is_sender_authorized(
                normalized_user_id, chat_type, str(channel_id or ""))
            if injected is not None:
                return injected
        auth_fn = self._runner_auth_fn()
        if callable(auth_fn):
            try:
                from gateway.session import SessionSource
                source = SessionSource(
                    platform=_adapter.Platform.SLACK, chat_id=str(channel_id or normalized_user_id),
                    chat_type=chat_type, user_id=normalized_user_id,
                    user_name=str(user_name).strip() if user_name else None,
                    scope_id=str(team_id) if team_id else None)
                return bool(auth_fn(source))
            except Exception:
                _adapter.logger.debug(
                    "[Slack] Falling back to env-only interactive auth for user %s",
                    normalized_user_id, exc_info=True)
        # Env-only fallback. Per-profile accessor: under multiplex a scoped miss
        # returns "" rather than leaking the DEFAULT profile's os.environ allowlist.
        from gateway.authz_mixin import _platform_gate_env as _env
        if _env("SLACK_ALLOW_ALL_USERS").lower() in {"true", "1", "yes"}:
            return True
        allowed_ids = {
            uid.strip()
            for var in ("SLACK_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS")
            for uid in _env(var).split(",")
            if uid.strip()}
        if allowed_ids:
            return "*" in allowed_ids or normalized_user_id in allowed_ids
        return _env("GATEWAY_ALLOW_ALL_USERS").lower() in {"true", "1", "yes"}

    @staticmethod
    def _interaction_fields(body: dict, action: dict) -> Tuple[str, str, dict, str, str, str, str]:
        """Unpack a Block Kit interaction payload into
        ``(action_id, value, message, msg_ts, channel_id, user_name, user_id)``."""
        message = body.get("message", {})
        return (
            action.get("action_id", ""), action.get("value", ""), message, message.get("ts", ""),
            body.get("channel", {}).get("id", ""), body.get("user", {}).get("name", "unknown"),
            body.get("user", {}).get("id", ""))

    async def _begin_interaction(
        self, ack, body: dict, action: dict, kind: str, *, team_scoped: bool = True
    ) -> Optional[Tuple[str, str, str, dict, str, str, str, str]]:
        """Ack a button click, unpack it and authorize the clicker.
        Returns ``(team_id, action_id, value, message, msg_ts, channel_id, user_name, user_id)`` or
        None (logged) when the user is not authorized."""
        from . import adapter as _adapter

        await ack()
        team_id = self._event_team_id({}, body)
        action_id, value, message, msg_ts, channel_id, user_name, user_id = (
            self._interaction_fields(body, action))
        auth_kwargs: _adapter.Dict[str, _adapter.Any] = {"channel_id": channel_id, "user_name": user_name}
        if team_scoped:
            auth_kwargs["team_id"] = team_id
        if not self._is_interactive_user_authorized(user_id, **auth_kwargs):
            _adapter.logger.warning(
                "[Slack] Unauthorized %s click by %s (%s) - ignoring", kind, user_name, user_id)
            return None
        return team_id, action_id, value, message, msg_ts, channel_id, user_name, user_id

    @staticmethod
    def _section_text(message: dict, limit: Optional[int] = 3000) -> str:
        """First ``section`` block text, truncated: Slack re-escapes HTML entities in
        interaction payloads, which can push it past the 3000-char cap."""
        original_text = ""
        for block in message.get("blocks", []):
            if block.get("type") == "section":
                original_text = (block.get("text") or {}).get("text", "")
                break
        return original_text[:limit] if limit is not None else original_text

    async def _finalize_interactive_message(
        self, channel_id: str, msg_ts: str, original_text: str, decision_text: str,
        placeholder: str, label: str, team_id: Optional[str] = None, sanitize: bool = True) -> None:
        """Rewrite a button prompt to show the outcome and drop the buttons."""
        from . import adapter as _adapter

        updated_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": original_text or placeholder}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": decision_text}]}]
        try:
            await self._get_client(channel_id, team_id=team_id).chat_update(
                channel=channel_id, ts=msg_ts, text=decision_text,
                blocks=_adapter.sanitize_blocks(updated_blocks) if sanitize else updated_blocks)
        except Exception as e:
            _adapter.logger.warning("[Slack] Failed to update %s message: %s", label, e)

    async def _handle_slash_confirm_action(self, ack, body, action) -> None:
        """Handle a slash-confirm button click from Block Kit."""
        from . import adapter as _adapter

        started = await self._begin_interaction(ack, body, action, "slash-confirm")
        if started is None:
            return
        team_id, action_id, value, message, msg_ts, channel_id, user_name, user_id = started
        if "|" not in value:
            _adapter.logger.warning("[Slack] Malformed slash-confirm value: %s", value)
            return
        session_key, confirm_id = value.split("|", 1)
        choice = self._CONFIRM_CHOICES.get(action_id, "cancel")
        decision_text = self._CONFIRM_DECISIONS[choice].format(user=user_name)
        await self._finalize_interactive_message(
            channel_id, msg_ts, self._section_text(message), decision_text,
            "Confirmation prompt", "slash-confirm", team_id or None)
        try:
            from tools import slash_confirm as _slash_confirm_mod
            result_text = await _slash_confirm_mod.resolve(session_key, confirm_id, choice)
            if result_text:
                post_kwargs: _adapter.Dict[str, _adapter.Any] = {"channel": channel_id, "text": result_text}
                thread_ts = message.get("thread_ts") or msg_ts  # stay in the same thread
                if thread_ts:
                    post_kwargs["thread_ts"] = thread_ts
                await self._get_client(channel_id, team_id=team_id or None).chat_postMessage(
                    **post_kwargs)
            _adapter.logger.info(
                "Slack button resolved slash-confirm for session %s (choice=%s, user=%s)",
                session_key, choice, user_name)
        except Exception as exc:
            _adapter.logger.error(
                "Failed to resolve slash-confirm from Slack button: %s", exc, exc_info=True)

    async def _handle_feedback_action(self, ack, body, action) -> None:
        """Ack Slack AI feedback button clicks and log the choice."""
        from . import adapter as _adapter

        await ack()
        value = str(action.get("value") or "")
        message = body.get("message", {}) or {}
        channel_id = (body.get("channel") or {}).get("id", "")
        user_id = (body.get("user") or {}).get("id", "")
        _adapter.logger.info(
            "[Slack] Feedback button clicked: value=%s user=%s channel=%s ts=%s", value, user_id,
            channel_id, message.get("ts", ""))

    async def _handle_approval_action(self, ack, body, action) -> None:
        """Handle an approval button click from Block Kit."""
        from . import adapter as _adapter

        started = await self._begin_interaction(ack, body, action, "approval")
        if started is None:
            return
        team_id, action_id, session_key, message, msg_ts, channel_id, user_name, user_id = started
        choice = self._APPROVAL_CHOICES.get(action_id, "deny")
        # Double-click guard (atomic pop). Also accept the bare ts: the approval may
        # have been stored without a team id while the click carries one.
        approval_key = self._workspace_message_marker(team_id, msg_ts)
        if msg_ts in self._approval_resolved:
            approval_key = msg_ts
        if self._approval_resolved.pop(approval_key, True):
            return
        # Resolve FIRST (unblocks the agent); render after so a click past the
        # timeout (count == 0) shows "expired", not "approved".
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice)
            _adapter.logger.info(
                "Slack button resolved %d approval(s) for session %s (choice=%s, user=%s)", count,
                session_key, choice, user_name)
        except Exception as exc:
            _adapter.logger.error("Failed to resolve gateway approval from Slack button: %s", exc)
            count = 0
        decision_text = self._APPROVAL_DECISIONS[choice].format(user=user_name)
        if not count:
            decision_text = (
                "⌛ Approval expired — command was not run (already timed out or resolved elsewhere)"
            )
        await self._finalize_interactive_message(
            channel_id, msg_ts, self._section_text(message), decision_text,
            "Command approval request", "approval", team_id or None)

    async def _update_clarify_message(
        self, channel_id: str, msg_ts: str, question_text: str, decision_text: str) -> None:
        """Rewrite a clarify message to show the outcome and drop the buttons."""
        await self._finalize_interactive_message(
            channel_id, msg_ts, question_text, decision_text, "Clarification", "clarify", sanitize=False
        )

    async def _handle_clarify_action(self, ack, body, action) -> None:
        """Handle a clarify button click (a choice or "Other") from Block Kit."""
        from . import adapter as _adapter

        started = await self._begin_interaction(ack, body, action, "clarify", team_scoped=False)
        if started is None:
            return
        _team_id, action_id, value, message, msg_ts, channel_id, user_name, user_id = started
        if "|" not in value:  # value packs ``clarify_id|<idx|other>``
            _adapter.logger.warning("[Slack] Malformed clarify value: %s", value)
            return
        clarify_id, token = value.split("|", 1)
        # Double-click guard — atomic pop (mirrors approval).
        if self._clarify_resolved.pop(msg_ts, True):
            return
        original_text = self._section_text(message, limit=None)
        from tools import clarify_gateway as _clarify_mod
        # "Other" → text-capture mode: mark_awaiting_text flips the entry and the
        # gateway's text-intercept resolves it from the user's next message.
        expired_text = f"⏳ This prompt expired — please send a new request. (by {user_name})"
        if action_id == "hermes_clarify_other" or token == "other":
            if not _clarify_mod.mark_awaiting_text(clarify_id):
                # Entry evicted/gateway restarted — a typed answer would go nowhere.
                await self._update_clarify_message(channel_id, msg_ts, original_text, expired_text)
                return
            await self._update_clarify_message(
                channel_id, msg_ts, original_text, f"✏️ Awaiting typed answer from {user_name}…")
            return
        try:
            idx = int(token)
        except (ValueError, TypeError):
            _adapter.logger.warning("[Slack] Invalid clarify choice token: %s", token)
            return
        # Canonical choice text from the entry; positional fallback on timeout/reset race.
        resolved_text: _adapter.Optional[str] = None
        try:
            entry = _clarify_mod._entries.get(clarify_id)  # type: ignore[attr-defined]
            if entry and entry.choices and 0 <= idx < len(entry.choices):
                resolved_text = str(entry.choices[idx])
        except Exception:
            resolved_text = None
        if resolved_text is None:
            resolved_text = f"choice {idx + 1}"
        if _clarify_mod.resolve_gateway_clarify(clarify_id, resolved_text):
            await self._update_clarify_message(
                channel_id, msg_ts, original_text, f"✅ {user_name}: {resolved_text}")
            # Privacy: choice text may carry user context — INFO gets metadata only.
            _adapter.logger.info(
                "Slack button resolved clarify (id=%s, choice_index=%d, user=%s)", clarify_id, idx,
                user_name)
            _adapter.logger.debug("Slack clarify choice text (id=%s): %.100r", clarify_id, resolved_text)
        else:
            # Entry evicted/gateway restarted — show expiry, not a misleading ✓.
            await self._update_clarify_message(channel_id, msg_ts, original_text, expired_text)
            _adapter.logger.warning(
                "[Slack] clarify resolve returned False (id=%s) — expired/reset", clarify_id)
