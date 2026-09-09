"""Discord SDK view construction, shared by initial import and lazy dependency binding."""

from __future__ import annotations

from typing import Any, List, Optional
try:
    import discord
    from discord import Message as DiscordMessage
except ImportError:
    discord = None
    DiscordMessage = Any

def define_discord_view_classes():
    """Register Discord UI view classes as module globals.
    Called at module load and after a lazy install so the classes exist whenever DISCORD_AVAILABLE."""

    from . import adapter as _adapter

    class _HermesView(_adapter.discord.ui.View):
        """Shared plumbing for Hermes component views: allowlist auth, single-use
        ``resolved`` flag, ``_message`` handle for timeout edits."""

        def __init__(self, allowed_user_ids: set, allowed_role_ids: Optional[set], *, timeout):
            super().__init__(timeout=timeout)
            self.allowed_user_ids = allowed_user_ids
            self.allowed_role_ids = allowed_role_ids or set()
            self.resolved = False
            self._message = None

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            return _adapter._component_check_auth(interaction, self.allowed_user_ids, self.allowed_role_ids)

        async def _gate(self, interaction: discord.Interaction, *, resolved_msg: Optional[str], unauth_msg: str) -> bool:
            """Reject (ephemerally) an already-resolved or unauthorized click; True when it may proceed."""
            if resolved_msg is not None and self.resolved:
                await interaction.response.send_message(resolved_msg, ephemeral=True)
                return False
            if not self._check_auth(interaction):
                await interaction.response.send_message(unauth_msg, ephemeral=True)
                return False
            return True

        def _disable_all(self) -> None:
            for child in self.children:
                child.disabled = True

        @staticmethod
        def _first_embed(message):
            return message.embeds[0] if message.embeds else None

        async def _expire_embed(self, footer: str) -> None:
            """Grey out the original message's embed after a timeout (best effort)."""
            msg = self._message
            if msg:
                try:
                    embed = self._first_embed(msg)
                    if embed:
                        embed.color = _adapter.discord.Color.greyple()
                        embed.set_footer(text=footer)
                    await msg.edit(embed=embed, view=self)
                except Exception:
                    pass  # message deleted or too old to edit

        async def _finalize_embed(self, interaction: discord.Interaction, color, footer: str) -> None:
            """Mark resolved, stamp the embed (color + footer), disable buttons, edit in place."""
            self.resolved = True
            embed = self._first_embed(interaction.message)
            if embed:
                embed.color = color
                embed.set_footer(text=footer)
            self._disable_all()
            await interaction.response.edit_message(embed=embed, view=self)

        async def on_timeout(self):
            self.resolved = True
            self._disable_all()
            await self._expire_embed("⏱ Prompt expired — no action taken")

    class ExecApprovalView(_HermesView):
        """Allow Once / Allow Session / Always Allow / Deny buttons for a dangerous command.
        Clicks call ``resolve_gateway_approval()`` — the same mechanism as the text ``/approve`` flow."""

        def __init__(
            self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None,
            require_admin: bool = False, admin_user_ids: Optional[set] = None,
            allow_permanent: bool = True, allow_session: bool = True, smart_denied: bool = False,
        ):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_adapter._read_discord_prompt_timeout())
            self.session_key = session_key
            self.require_admin = require_admin
            self.admin_user_ids = {str(a).strip() for a in (admin_user_ids or set()) if str(a).strip()}
            if smart_denied or not allow_session:
                self.remove_item(self.allow_session)
                self.remove_item(self.allow_always)
            elif not allow_permanent:
                self.remove_item(self.allow_always)

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            """Base admission always required; with ``require_admin`` the clicker must
            also be an admin. Fails closed (logged once) when no admins are configured."""
            if not super()._check_auth(interaction):
                return False
            if not self.require_admin:
                return True
            user = getattr(interaction, "user", None)
            try:
                uid = str(getattr(user, "id", "") or "")
            except Exception:
                uid = ""
            if uid and uid in self.admin_user_ids:
                return True
            if not self.admin_user_ids:
                _adapter.logger.warning(
                    "[Discord] require_admin_for_exec_approval is enabled but "
                    "no admins are configured (allow_admin_from is empty) — "
                    "exec approval buttons are disabled for everyone. Add "
                    "admin user IDs under the discord platform's "
                    "allow_admin_from, or disable the toggle."
                )
            return False

        async def _resolve(self, interaction: discord.Interaction, choice: str, color: discord.Color, label: str):
            """Resolve the approval via the gateway approval queue and update the embed."""
            if not await self._gate(
                interaction, resolved_msg="This approval has already been resolved~",
                unauth_msg="You're not authorized to approve commands~",
            ):
                return
            self.resolved = True
            # Unblock the waiting agent thread FIRST. A click after the approval
            # wait timed out (count == 0) must not claim "Approved".
            try:
                from tools.approval import resolve_gateway_approval
                count = resolve_gateway_approval(self.session_key, choice)
                _adapter.logger.info(
                    "Discord button resolved %d approval(s) for session %s (choice=%s, user=%s)",
                    count, self.session_key, choice, interaction.user.display_name,
                )
            except Exception as exc:
                _adapter.logger.error("Failed to resolve gateway approval from button: %s", exc)
                count = 0
            if not count:
                color = _adapter.discord.Color.dark_grey()
                label = "⌛ Approval expired — command was not run (already timed out or resolved elsewhere)"
            await self._finalize_embed(
                interaction, color, f"{label} by {interaction.user.display_name}" if count else label)

        @_adapter.discord.ui.button(label="Allow Once", style=_adapter.discord.ButtonStyle.green)
        async def allow_once(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "once", _adapter.discord.Color.green(), "Approved once")

        @_adapter.discord.ui.button(label="Allow Session", style=_adapter.discord.ButtonStyle.grey)
        async def allow_session(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "session", _adapter.discord.Color.blue(), "Approved for session")

        @_adapter.discord.ui.button(label="Always Allow", style=_adapter.discord.ButtonStyle.blurple)
        async def allow_always(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "always", _adapter.discord.Color.purple(), "Approved permanently")

        @_adapter.discord.ui.button(label="Deny", style=_adapter.discord.ButtonStyle.red)
        async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "deny", _adapter.discord.Color.red(), "Denied")

    class SlashConfirmView(_HermesView):
        """Approve Once / Always Approve / Cancel for slash-command confirmations (``/reload-mcp``,
        ``GatewayRunner._request_slash_confirm``); clicks call ``tools.slash_confirm.resolve(...)``."""

        def __init__(self, session_key: str, confirm_id: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_adapter._read_discord_prompt_timeout())
            self.session_key = session_key
            self.confirm_id = confirm_id

        async def _resolve(self, interaction: discord.Interaction, choice: str, color: discord.Color, label: str):
            if not await self._gate(
                interaction, resolved_msg="This prompt has already been resolved~",
                unauth_msg="You're not authorized to answer this prompt~",
            ):
                return
            await self._finalize_embed(interaction, color, f"{label} by {interaction.user.display_name}")
            # A returned follow-up message is posted in the same channel.
            try:
                from tools import slash_confirm as _slash_confirm_mod
                result_text = await _slash_confirm_mod.resolve(self.session_key, self.confirm_id, choice)
                if result_text:
                    await interaction.followup.send(result_text)
                _adapter.logger.info(
                    "Discord button resolved slash-confirm for session %s "
                    "(choice=%s, user=%s)",
                    self.session_key, choice, interaction.user.display_name,
                )
            except Exception as exc:
                _adapter.logger.error("Discord slash-confirm resolve failed: %s", exc, exc_info=True)

        @_adapter.discord.ui.button(label="Approve Once", style=_adapter.discord.ButtonStyle.green)
        async def approve_once(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "once", _adapter.discord.Color.green(), "Approved once")

        @_adapter.discord.ui.button(label="Always Approve", style=_adapter.discord.ButtonStyle.blurple)
        async def approve_always(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "always", _adapter.discord.Color.purple(), "Always approved")

        @_adapter.discord.ui.button(label="Cancel", style=_adapter.discord.ButtonStyle.red)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._resolve(interaction, "cancel", _adapter.discord.Color.greyple(), "Cancelled")

    class UpdatePromptView(_HermesView):
        """Yes/No buttons for ``hermes update`` prompts; the answer is written to
        ``.update_response`` for the detached update process to pick up."""

        def __init__(self, session_key: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_adapter._read_discord_prompt_timeout())
            self.session_key = session_key

        async def _respond(self, interaction: discord.Interaction, answer: str, color: discord.Color, label: str):
            if not await self._gate(interaction, resolved_msg="Already answered~", unauth_msg="You're not authorized~"):
                return
            await self._finalize_embed(interaction, color, f"{label} by {interaction.user.display_name}")
            try:
                from hermes_constants import get_hermes_home
                response_path = get_hermes_home() / ".update_response"
                tmp = response_path.with_suffix(".tmp")
                tmp.write_text(answer, encoding="utf-8")
                tmp.replace(response_path)
                _adapter.logger.info("Discord update prompt answered '%s' by %s", answer, interaction.user.display_name)
            except Exception as exc:
                _adapter.logger.error("Failed to write update response: %s", exc)

        @_adapter.discord.ui.button(label="Yes", style=_adapter.discord.ButtonStyle.green, emoji="✓")
        async def yes_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._respond(interaction, "y", _adapter.discord.Color.green(), "Yes")

        @_adapter.discord.ui.button(label="No", style=_adapter.discord.ButtonStyle.red, emoji="✗")
        async def no_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._respond(interaction, "n", _adapter.discord.Color.red(), "No")

    class ModelPickerView(_HermesView):
        """Two-step select-menu model picker: provider dropdown → model dropdown,
        editing the original message in place. Times out after 2 minutes."""

        def __init__(
            self, providers: list, current_model: str, current_provider: str, session_key: str,
            on_model_selected, allowed_user_ids: set, allowed_role_ids: Optional[set] = None,
        ):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=120)
            self.providers = providers
            self.current_model = current_model
            self.current_provider = current_provider
            self.session_key = session_key
            self.on_model_selected = on_model_selected
            self._selected_provider: str = ""
            self._pending_expensive_model: str = ""
            self._build_provider_select()

        def _add_button(self, label: str, style, custom_id: str, callback) -> None:
            btn = _adapter.discord.ui.Button(label=label, style=style, custom_id=custom_id)
            btn.callback = callback
            self.add_item(btn)

        def _add_select(self, placeholder: str, options: list, custom_id: str, callback) -> None:
            select = _adapter.discord.ui.Select(placeholder=placeholder, options=options, custom_id=custom_id)
            select.callback = callback
            self.add_item(select)

        async def _edit(self, interaction: discord.Interaction, description: str, *, view=..., **embed_kw) -> None:
            """Edit the picker message in place with a config embed (``view`` defaults to self)."""
            await interaction.response.edit_message(
                embed=self._config_embed(description, **embed_kw), view=self if view is ... else view,
            )

        def _build_provider_select(self):
            """Build the provider dropdown menu."""
            self.clear_items()
            options = []
            for p in self.providers:
                count = p.get("total_models", len(p.get("models", [])))
                options.append(_adapter.discord.SelectOption(
                    label=_adapter._truncate_discord_component_text(f"{p['name']} ({count} models)", _adapter._DISCORD_SELECT_FIELD_LIMIT),
                    value=p["slug"], description="current" if p.get("is_current") else None,
                ))
            if not options:
                return
            self._add_select(
                "Choose a provider...", options[:_adapter._DISCORD_SELECT_MAX_OPTIONS], "model_provider_select",
                self._on_provider_selected,
            )
            self._add_button("Cancel", _adapter.discord.ButtonStyle.red, "model_cancel", self._on_cancel)

        def _build_model_select(self, provider_slug: str):
            """Model dropdown(s) for one provider.
            Select caps at 25 options and View at 5 rows (2 reserved for Back/Cancel), so models are
            partitioned across up to 3 selects (75) rather than truncated (tail entries would vanish)."""
            self.clear_items()
            provider = next((p for p in self.providers if p["slug"] == provider_slug), None)
            if not provider:
                return
            models = provider.get("models", [])
            if not models:
                return
            chunks = [
                models[i : i + _adapter._DISCORD_SELECT_MAX_OPTIONS]
                for i in range(0, len(models), _adapter._DISCORD_SELECT_MAX_OPTIONS)
            ][: _adapter._DISCORD_SELECT_MAX_ROWS - 2]
            placeholder_base = f"Choose a model from {provider.get('name', provider_slug)}"
            for idx, chunk in enumerate(chunks):
                options = [
                    _adapter.discord.SelectOption(
                        label=_adapter._truncate_discord_component_text(model_id.split("/")[-1], _adapter._DISCORD_SELECT_FIELD_LIMIT),
                        value=_adapter._truncate_discord_component_text(model_id, _adapter._DISCORD_SELECT_FIELD_LIMIT),
                    )
                    for model_id in chunk
                ]
                suffix = f" ({idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                self._add_select(
                    f"{placeholder_base}{suffix}...", options, f"model_model_select_{idx}", self._on_model_selected)
            self._add_button("◀ Back", _adapter.discord.ButtonStyle.grey, "model_back", self._on_back)
            self._add_button("Cancel", _adapter.discord.ButtonStyle.red, "model_cancel2", self._on_cancel)

        def _build_expensive_confirm(self, model_id: str):
            """Build confirmation buttons for unusually expensive models."""
            self.clear_items()
            self._pending_expensive_model = model_id
            self._add_button("Switch anyway", _adapter.discord.ButtonStyle.red, "model_expensive_confirm", self._on_expensive_confirm)
            self._add_button("Cancel", _adapter.discord.ButtonStyle.grey, "model_expensive_cancel", self._on_cancel)

        async def _expensive_warning_for(self, model_id: str):
            try:
                from hermes_cli.model_selection_guards import combined_selection_warning
                # Pricing lookup can hit models.dev on a cache miss — keep it off the event loop.
                return await _adapter.asyncio.to_thread(combined_selection_warning, model_id, provider=self._selected_provider)
            except Exception:
                return None

        def _config_embed(self, description: str, *, title: str = "⚙ Model Configuration", color=None):
            return _adapter.discord.Embed(title=title, description=description, color=_adapter.discord.Color.blue() if color is None else color)

        async def _on_provider_selected(self, interaction: discord.Interaction):
            if not await self._gate(interaction, resolved_msg=None, unauth_msg="You're not authorized~"):
                return
            provider_slug = interaction.data["values"][0]
            self._selected_provider = provider_slug
            provider = next((p for p in self.providers if p["slug"] == provider_slug), None)
            pname = provider.get("name", provider_slug) if provider else provider_slug
            self._build_model_select(provider_slug)
            # `shown` counts models actually rendered across the partitioned selects (≤ 75).
            total = provider.get("total_models", 0) if provider else 0
            shown = min(len(provider.get("models", [])), _adapter._DISCORD_MODEL_SELECT_CAPACITY) if provider else 0
            extra = f"\n*{total - shown} more available — type `/model <name>` directly*" if total > shown else ""
            await self._edit(interaction, f"Provider: **{pname}**\nSelect a model:{extra}")

        async def _switch_selected_model(self, interaction: discord.Interaction, model_id: str):
            if not await self._gate(interaction, resolved_msg="Already resolved~", unauth_msg="You're not authorized~"):
                return
            self.resolved = True
            self.clear_items()
            await self._edit(interaction, f"Switching to `{model_id}`...", title="⚙ Switching Model", view=None)
            try:
                result_text = await self.on_model_selected(str(interaction.channel_id), model_id, self._selected_provider)
            except Exception as exc:
                result_text = f"Error switching model: {exc}"
            await interaction.edit_original_response(
                embed=self._config_embed(result_text, title="⚙ Model Switched", color=_adapter.discord.Color.green()),
                view=None,
            )

        async def _on_model_selected(self, interaction: discord.Interaction):
            if not await self._gate(interaction, resolved_msg="Already resolved~", unauth_msg="You're not authorized~"):
                return
            model_id = interaction.data["values"][0]
            warning = await self._expensive_warning_for(model_id)
            if warning is not None:
                self._build_expensive_confirm(model_id)
                await self._edit(interaction, warning.message, title=f"⚠ {warning.title}", color=_adapter.discord.Color.red())
                return
            await self._switch_selected_model(interaction, model_id)

        async def _on_expensive_confirm(self, interaction: discord.Interaction):
            if not await self._gate(interaction, resolved_msg=None, unauth_msg="You're not authorized~"):
                return
            if not self._pending_expensive_model:
                await interaction.response.send_message("Model selection expired.", ephemeral=True)
                return
            await self._switch_selected_model(interaction, self._pending_expensive_model)

        async def _on_back(self, interaction: discord.Interaction):
            if not await self._gate(interaction, resolved_msg=None, unauth_msg="You're not authorized~"):
                return
            self._build_provider_select()
            try:
                from hermes_cli.providers import get_label
                provider_label = get_label(self.current_provider)
            except Exception:
                provider_label = self.current_provider
            await self._edit(
                interaction,
                f"Current model: `{self.current_model or 'unknown'}`\nProvider: {provider_label}\n\nSelect a provider:",
            )

        async def _on_cancel(self, interaction: discord.Interaction):
            self.resolved = True
            self.clear_items()
            await self._edit(interaction, "Model selection cancelled.", color=_adapter.discord.Color.greyple())

        async def on_timeout(self):
            self.resolved = True
            self.clear_items()
            msg = self._message
            if msg:
                try:
                    embed = self._config_embed("⏱ Selection expired — no model change.", color=_adapter.discord.Color.greyple())
                    await msg.edit(embed=embed, view=self)
                except Exception:
                    pass

    from .choice_picker import define_choice_picker_view

    ChoicePickerView = define_choice_picker_view(
        discord_sdk=_adapter.discord,
        component_check_auth=_adapter._component_check_auth,
        truncate_component_text=_adapter._truncate_discord_component_text,
        logger=_adapter.logger,
        max_options=_adapter._DISCORD_SELECT_MAX_OPTIONS,
        field_limit=_adapter._DISCORD_SELECT_FIELD_LIMIT,
    )

    class ClarifyChoiceView(_HermesView):
        """One button per clarify choice (max 24) plus ``✏️ Other``. A numeric click resolves the
        gateway clarify entry immediately; ``Other`` flips to text-capture (next message answers).
        Single-use: after the first valid click all buttons disable."""

        def __init__(self, choices: List[str], clarify_id: str, allowed_user_ids: set, allowed_role_ids: Optional[set] = None):
            super().__init__(allowed_user_ids, allowed_role_ids, timeout=_adapter._read_discord_prompt_timeout())
            self.choices = list(choices)[:24]
            self.clarify_id = clarify_id
            for index, choice in enumerate(self.choices):
                button = _adapter.discord.ui.Button(
                    label=self._button_label(index, choice), style=_adapter.discord.ButtonStyle.primary,
                    custom_id=f"clarify:{clarify_id}:{index}",
                )
                button.callback = self._make_choice_callback(index, choice)
                self.add_item(button)
            other_btn = _adapter.discord.ui.Button(
                label="✏️ Other (type answer)", style=_adapter.discord.ButtonStyle.secondary,
                custom_id=f"clarify:{clarify_id}:other",
            )
            other_btn.callback = self._on_other
            self.add_item(other_btn)

        @staticmethod
        def _button_label(index: int, choice: str) -> str:
            """``"N. <choice>"`` within Discord's 80-char (UTF-16) label cap.
            Mobile wraps early, so long choices cut at a word boundary in the trailing half, else a
            soft boundary (``- , . )``, inclusive), else hard."""
            prefix = f"{index + 1}. "
            budget = _adapter._DISCORD_BUTTON_LABEL_LIMIT - _adapter.utf16_len(prefix)
            if _adapter.utf16_len(choice) <= budget:
                return f"{prefix}{choice}"
            truncated = _adapter._prefix_within_utf16_limit(choice, max(0, budget - _adapter.utf16_len(_adapter._DISCORD_ELLIPSIS))).rstrip()
            cut_at = -1
            space = truncated.rfind(" ")
            if space >= len(truncated) // 2:
                cut_at = space
            if cut_at < 0:
                latest_soft = max((truncated.rfind(s) for s in ("-", ",", ".", ")")), default=-1)
                if latest_soft >= len(truncated) // 2:
                    cut_at = latest_soft + 1
            if cut_at > 0:
                truncated = truncated[:cut_at]
            return f"{prefix}{truncated.rstrip() + _adapter._DISCORD_ELLIPSIS}"

        def _make_choice_callback(self, index: int, choice: str):
            async def _callback(interaction: "discord.Interaction"):
                await self._resolve_choice(interaction, index, choice)
            return _callback

        async def _finish(self, interaction: "discord.Interaction", color, footer: str, *, log_edit_failure: bool) -> None:
            """Disable the buttons and stamp the embed; fall back to a bare defer."""
            self.resolved = True
            self._disable_all()
            embed = self._first_embed(interaction.message) if interaction.message else None
            if embed:
                embed.color = color
                embed.set_footer(text=footer)
            try:
                await interaction.response.edit_message(embed=embed, view=self)
            except Exception:
                if log_edit_failure:
                    _adapter.logger.debug("Discord clarify edit_message failed for %s", self.clarify_id, exc_info=True)
                try:
                    await interaction.response.defer()
                except Exception:
                    pass

        async def _resolve_choice(self, interaction: "discord.Interaction", index: int, choice: str) -> None:
            """Resolve the clarify with a chosen option."""
            if not await self._gate(
                interaction, resolved_msg="This prompt has already been answered~",
                unauth_msg="You're not authorized to answer this prompt~",
            ):
                return
            display_name = getattr(getattr(interaction, "user", None), "display_name", "user")
            await self._finish(interaction, _adapter.discord.Color.green(), f"Answered by {display_name}: {choice}", log_edit_failure=True)
            # Round-trip the canonical choice text from the entry, not the button label.
            resolved_text: _adapter.Optional[str] = None
            try:
                from tools.clarify_gateway import _entries as _clarify_entries  # type: ignore
                entry = _clarify_entries.get(self.clarify_id)
                if entry and entry.choices and 0 <= index < len(entry.choices):
                    resolved_text = entry.choices[index]
            except Exception:
                resolved_text = None
            if resolved_text is None:
                resolved_text = choice
            try:
                from tools.clarify_gateway import resolve_gateway_clarify
                resolved = resolve_gateway_clarify(self.clarify_id, resolved_text)
                _adapter.logger.info(
                    "Discord clarify button resolved (id=%s, choice=%r, user=%s, ok=%s)",
                    self.clarify_id, resolved_text,
                    getattr(getattr(interaction, "user", None), "display_name", "?"), resolved,
                )
            except Exception as exc:
                _adapter.logger.error("Discord clarify resolve_gateway_clarify failed (id=%s): %s", self.clarify_id, exc)

        async def _on_other(self, interaction: "discord.Interaction") -> None:
            """Flip the clarify entry into text-capture mode."""
            if not await self._gate(
                interaction, resolved_msg="This prompt has already been answered~",
                unauth_msg="You're not authorized to answer this prompt~",
            ):
                return
            # Don't pop: the gateway text-intercept needs the entry until the user types.
            try:
                from tools.clarify_gateway import mark_awaiting_text
                mark_awaiting_text(self.clarify_id)
            except Exception as exc:
                _adapter.logger.warning("Discord clarify mark_awaiting_text failed (id=%s): %s", self.clarify_id, exc)
            display_name = getattr(getattr(interaction, "user", None), "display_name", "user")
            await self._finish(interaction, _adapter.discord.Color.blue(), f"Awaiting typed response from {display_name}…", log_edit_failure=False)

    return ExecApprovalView, SlashConfirmView, UpdatePromptView, ModelPickerView, ClarifyChoiceView, ChoicePickerView
