"""Discord prompts methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Dict, Optional
from gateway.platforms.base import SendResult
try:
    import discord
    from discord import Message as DiscordMessage
except ImportError:
    discord = None
    DiscordMessage = Any


class DiscordPromptsMixin:
    def _self_contained_prompt_content(
        self, header: str, body: str, *, code_block: bool = False, tail: str = ""
    ) -> str:
        """Plain content mirroring an embed's payload.
        Embeds can be invisible/detached on web/mobile, so ``content`` carries the payload."""
        from . import adapter as _adapter

        body = str(body or "")
        if code_block:
            prefix = f"{header}\n```bash\n"
            suffix = f"\n```{tail}"
        else:
            prefix = f"{header}\n\n"
            suffix = tail
        truncated_suffix = "\n... [truncated]"
        budget = max(0, self.MAX_MESSAGE_LENGTH - len(prefix) - len(suffix))
        if len(body) > budget:
            body = body[: max(0, budget - len(truncated_suffix))] + truncated_suffix
        return f"{prefix}{body}{suffix}"

    def _approval_mention_content(self) -> Optional[str]:
        """User mentions for approval prompts, gated on ``discord.approval_mentions``
        (``DISCORD_APPROVAL_MENTIONS``). Only numeric allowlist entries; default off."""
        from . import adapter as _adapter

        if not _adapter._env_bool("DISCORD_APPROVAL_MENTIONS", False):
            return None
        user_ids = sorted(uid for uid in self._allowed_user_ids if str(uid).isdigit())
        if not user_ids:
            return None
        return " ".join(f"<@{uid}>" for uid in user_ids)

    async def _send_prompt(
        self, chat_id: str, metadata: Optional[dict], build, *, fail_log: Optional[str] = None,
    ) -> SendResult:
        """Shared tail for interactive prompts: resolve target channel, call ``build(channel) ->
        (send_kwargs, view)``, send, remember the message on the view. ``fail_log`` labels failures."""
        from . import adapter as _adapter

        if not self._client or not _adapter.DISCORD_AVAILABLE:
            return _adapter.SendResult(success=False, error="Not connected")
        try:
            channel = await self._resolve_channel(_adapter._prompt_target_id(chat_id, metadata))
            send_kwargs, view = build(channel)
            msg = await channel.send(**send_kwargs)
            if view is not None:
                view._message = msg
            return _adapter.SendResult(success=True, message_id=str(msg.id))
        except Exception as e:
            if fail_log:
                _adapter.logger.warning("[%s] %s failed: %s", self.name, fail_log, e)
            return _adapter.SendResult(success=False, error=str(e))

    @staticmethod
    def _embed_body(text: str, limit: int = 4088) -> str:
        """Trim to Discord's 4096-char embed description limit (conservatively)."""
        return text if len(text) <= limit else text[: limit - 3] + "..."

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[dict] = None, allow_permanent: bool = True, allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Button-based exec approval prompt; buttons call ``resolve_gateway_approval()`` (not /approve)."""
        from . import adapter as _adapter

        def _build(_channel):
            # Payload in plain content: embeds can be invisible/detached on web/mobile.
            reason_budget = 300
            reason_display = str(description or "dangerous command")
            if len(reason_display) > reason_budget:
                reason_display = reason_display[: reason_budget - 15] + "... [truncated]"
            prompt_prefix = (
                "⚠️ **Command Approval Required**\n\n"
                "Do you want Hermes to run this command?\n\n"
                "**Requested command:**\n```bash\n"
            )
            if smart_denied:
                prompt_prefix += "**Smart DENY:** owner override applies to this one operation only.\n\n"
            mention_content = self._approval_mention_content()
            if mention_content:
                prompt_prefix = f"{mention_content}\n{prompt_prefix}"
            prompt_tail = f"\n```\n**Reason:** {reason_display}"
            truncated_suffix = "\n... [truncated]"
            command_budget = max(0, self.MAX_MESSAGE_LENGTH - len(prompt_prefix) - len(prompt_tail))
            content_cmd_display = str(command or "")
            if len(content_cmd_display) > command_budget:
                content_cmd_display = content_cmd_display[: max(0, command_budget - len(truncated_suffix))] + truncated_suffix
            content = f"{prompt_prefix}{content_cmd_display}{prompt_tail}"
            embed = _adapter.discord.Embed(
                title="⚠️ Command Approval Required",
                description=f"```\n{self._embed_body(str(command or ''))}\n```",
                color=_adapter.discord.Color.orange(),
            )
            embed.add_field(name="Reason", value=reason_display, inline=False)
            require_admin, admin_user_ids = _adapter._resolve_exec_approval_admin_gate(getattr(self.config, "extra", None))
            view = _adapter.ExecApprovalView(
                session_key=session_key, allowed_user_ids=self._allowed_user_ids,
                allowed_role_ids=self._allowed_role_ids, require_admin=require_admin,
                admin_user_ids=admin_user_ids, allow_permanent=allow_permanent,
                allow_session=allow_session, smart_denied=smart_denied,
            )
            send_kwargs: _adapter.Dict[str, _adapter.Any] = {"content": content, "embed": embed, "view": view}
            if mention_content:
                allowed_mentions_cls = getattr(_adapter.discord, "AllowedMentions", None)
                if allowed_mentions_cls is not None:
                    send_kwargs["allowed_mentions"] = allowed_mentions_cls(
                        users=True, roles=False, everyone=False, replied_user=False,
                    )
            return send_kwargs, view
        return await self._send_prompt(chat_id, metadata, _build)

    async def send_slash_confirm(
        self, chat_id: str, title: str, message: str, session_key: str,
        confirm_id: str, metadata: Optional[dict] = None,
    ) -> SendResult:
        """Send a three-button slash-command confirmation prompt."""
        from . import adapter as _adapter

        def _build(_channel):
            embed = _adapter.discord.Embed(
                title=title or "Confirm", description=self._embed_body(message), color=_adapter.discord.Color.orange(),
            )
            content = self._self_contained_prompt_content(f"**{title or 'Confirm'}**", message)
            view = _adapter.SlashConfirmView(
                session_key=session_key, confirm_id=confirm_id,
                allowed_user_ids=self._allowed_user_ids, allowed_role_ids=self._allowed_role_ids,
            )
            return {"content": content, "embed": embed, "view": view}, view
        return await self._send_prompt(chat_id, metadata, _build)

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Clarify prompt: one button per choice plus ``✏️ Other`` (text-capture); with no choices the
        gateway's text-intercept captures the next message. Dict choices (LLMs emit
        ``[{"description": ...}]``) are unwrapped via ``label``/``description``/``text``/``title``."""
        from . import adapter as _adapter

        def _flatten_choice(c):
            if c is None:
                return ""
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, dict):
                # 'name'/'value' excluded: Discord-component-shaped fields would leak raw enum values.
                for key in ("label", "description", "text", "title"):
                    v = c.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                return ""
            if isinstance(c, (list, tuple)):
                return " ".join(_flatten_choice(x) for x in c).strip()
            return str(c).strip()

        def _build(_channel):
            embed = _adapter.discord.Embed(
                title="❓ Hermes needs your input",
                description=self._embed_body(str(question or "").strip()),
                color=_adapter.discord.Color.orange(),
            )
            # 5 buttons × 5 rows = 25; one slot is reserved for "Other".
            clean_choices = [s for s in (_flatten_choice(c) for c in (choices or [])) if s][:24]
            if clean_choices:
                hint = "Pick one below, or click ✏️ Other to type a custom answer."
                embed.add_field(name="Choices", value=hint, inline=False)
                view = _adapter.ClarifyChoiceView(
                    choices=clean_choices, clarify_id=clarify_id,
                    allowed_user_ids=self._allowed_user_ids,
                    allowed_role_ids=self._allowed_role_ids,
                )
            else:
                hint = "Reply in this channel with your answer."
                embed.add_field(name="Reply", value=hint, inline=False)
                view = None
            content = self._self_contained_prompt_content(
                "❓ **Hermes needs your input**", str(question or "").strip(), tail=f"\n\n{hint}",
            )
            send_kwargs = {"content": content, "embed": embed}
            if view:
                send_kwargs["view"] = view
            return send_kwargs, view
        return await self._send_prompt(chat_id, metadata, _build, fail_log="send_clarify")

    async def send_update_prompt(
        self, chat_id: str, prompt: str, default: str = "", session_key: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Yes/No prompt for the gateway ``/update`` watcher when ``hermes update --gateway`` needs input."""
        from . import adapter as _adapter

        def _build(_channel):
            default_hint = f" (default: {default})" if default else ""
            embed = _adapter.discord.Embed(
                title="⚕ Update Needs Your Input", description=f"{prompt}{default_hint}", color=_adapter.discord.Color.gold(),
            )
            view = _adapter.UpdatePromptView(
                session_key=session_key, allowed_user_ids=self._allowed_user_ids,
                allowed_role_ids=self._allowed_role_ids,
            )
            content = self._self_contained_prompt_content("⚕ **Update Needs Your Input**", f"{prompt}{default_hint}")
            return {"content": content, "embed": embed, "view": view}, view
        result = await self._send_prompt(chat_id, metadata, _build)
        if result.success and _adapter._metadata_marks_nonconversational(metadata):
            await self._nonconversational_messages.mark_many([result.message_id])
        return result

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str,
        session_key: str, on_model_selected, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Two-step select-menu model picker (provider → model) via ``ModelPickerView``."""
        from . import adapter as _adapter

        def _build(_channel):
            try:
                from hermes_cli.providers import get_label
                provider_label = get_label(current_provider)
            except Exception:
                provider_label = current_provider
            embed = _adapter.discord.Embed(
                title="⚙ Model Configuration",
                description=(
                    f"Current model: `{current_model or 'unknown'}`\n"
                    f"Provider: {provider_label}\n\n"
                    f"Select a provider:"
                ),
                color=_adapter.discord.Color.blue(),
            )
            view = _adapter.ModelPickerView(
                providers=providers, current_model=current_model, current_provider=current_provider,
                session_key=session_key, on_model_selected=on_model_selected,
                allowed_user_ids=self._allowed_user_ids, allowed_role_ids=self._allowed_role_ids,
            )
            return {"embed": embed, "view": view}, view
        return await self._send_prompt(chat_id, metadata, _build, fail_log="send_model_picker")

    supports_choice_pages = True
    choice_pages_edit_in_place = True

    async def send_choice_picker(
        self,
        chat_id: str,
        title: str,
        choices: list,
        session_key: str,
        on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a flat select-menu choice picker (one selection → one value).

        Generic single-level companion to ``send_model_picker`` used by
        `/reasoning`, `/fast`, and any future finite-choice command. Each
        choice dict: ``{"value": str, "label": str, "is_current": bool}``.
        """
        from . import adapter as _adapter
        from . import choice_picker as _choice_picker

        return await _choice_picker.send_choice_picker(
            self,
            chat_id,
            title,
            choices,
            session_key,
            on_choice_selected,
            metadata,
            discord_sdk=_adapter.discord,
            discord_available=_adapter.DISCORD_AVAILABLE,
            view_class=getattr(_adapter, "ChoicePickerView", None),
            logger=_adapter.logger,
        )
