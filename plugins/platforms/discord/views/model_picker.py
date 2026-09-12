"""Interactive Discord model picker view."""

from __future__ import annotations

import asyncio
from typing import Optional

from .. import adapter as _adapter
from .base import _HermesView

discord = _adapter.discord
logger = _adapter.logger
_DISCORD_SELECT_MAX_OPTIONS = _adapter._DISCORD_SELECT_MAX_OPTIONS
_DISCORD_SELECT_MAX_ROWS = _adapter._DISCORD_SELECT_MAX_ROWS
_DISCORD_SELECT_FIELD_LIMIT = _adapter._DISCORD_SELECT_FIELD_LIMIT
_DISCORD_MODEL_SELECT_CAPACITY = _adapter._DISCORD_MODEL_SELECT_CAPACITY
_truncate_discord_component_text = _adapter._truncate_discord_component_text
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
        self.session_key = session_key
        self.on_model_selected = on_model_selected
        self._selected_provider: str = ""
        self._pending_expensive_model: str = ""
        self._build_provider_select()

    def _add_button(self, label: str, style, custom_id: str, callback) -> None:
        btn = discord.ui.Button(label=label, style=style, custom_id=custom_id)
        btn.callback = callback
        self.add_item(btn)

    def _add_select(self, placeholder: str, options: list, custom_id: str, callback) -> None:
        select = discord.ui.Select(placeholder=placeholder, options=options, custom_id=custom_id)
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
            options.append(discord.SelectOption(
                label=_truncate_discord_component_text(f"{p['name']} ({count} models)", _DISCORD_SELECT_FIELD_LIMIT),
                value=p["slug"], description="current" if p.get("is_current") else None,
            ))
        if not options:
            return
        self._add_select(
            "Choose a provider...", options[:_DISCORD_SELECT_MAX_OPTIONS], "model_provider_select",
            self._on_provider_selected,
        )
        self._add_button("Cancel", discord.ButtonStyle.red, "model_cancel", self._on_cancel)

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
            models[i : i + _DISCORD_SELECT_MAX_OPTIONS]
            for i in range(0, len(models), _DISCORD_SELECT_MAX_OPTIONS)
        ][: _DISCORD_SELECT_MAX_ROWS - 2]
        placeholder_base = f"Choose a model from {provider.get('name', provider_slug)}"
        for idx, chunk in enumerate(chunks):
            options = [
                discord.SelectOption(
                    label=_truncate_discord_component_text(model_id.split("/")[-1], _DISCORD_SELECT_FIELD_LIMIT),
                    value=_truncate_discord_component_text(model_id, _DISCORD_SELECT_FIELD_LIMIT),
                )
                for model_id in chunk
            ]
            suffix = f" ({idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            self._add_select(
                f"{placeholder_base}{suffix}...", options, f"model_model_select_{idx}", self._on_model_selected)
        self._add_button("◀ Back", discord.ButtonStyle.grey, "model_back", self._on_back)
        self._add_button("Cancel", discord.ButtonStyle.red, "model_cancel2", self._on_cancel)

    def _build_expensive_confirm(self, model_id: str):
        """Build confirmation buttons for unusually expensive models."""
        self.clear_items()
        self._pending_expensive_model = model_id
        self._add_button("Switch anyway", discord.ButtonStyle.red, "model_expensive_confirm", self._on_expensive_confirm)
        self._add_button("Cancel", discord.ButtonStyle.grey, "model_expensive_cancel", self._on_cancel)

    async def _expensive_warning_for(self, model_id: str):
        try:
            from hermes_cli.model_selection_guards import combined_selection_warning
            # Pricing lookup can hit models.dev on a cache miss — keep it off the event loop.
            return await asyncio.to_thread(combined_selection_warning, model_id, provider=self._selected_provider)
        except Exception:
            return None

    def _config_embed(self, description: str, *, title: str = "⚙ Model Configuration", color=None):
        return discord.Embed(title=title, description=description, color=discord.Color.blue() if color is None else color)

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
        shown = min(len(provider.get("models", [])), _DISCORD_MODEL_SELECT_CAPACITY) if provider else 0
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
            embed=self._config_embed(result_text, title="⚙ Model Switched", color=discord.Color.green()),
            view=None,
        )

    async def _on_model_selected(self, interaction: discord.Interaction):
        if not await self._gate(interaction, resolved_msg="Already resolved~", unauth_msg="You're not authorized~"):
            return
        model_id = interaction.data["values"][0]
        warning = await self._expensive_warning_for(model_id)
        if warning is not None:
            self._build_expensive_confirm(model_id)
            await self._edit(interaction, warning.message, title=f"⚠ {warning.title}", color=discord.Color.red())
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
        await self._edit(interaction, "Model selection cancelled.", color=discord.Color.greyple())

    async def on_timeout(self):
        self.resolved = True
        self.clear_items()
        msg = self._message
        if msg:
            try:
                embed = self._config_embed("⏱ Selection expired — no model change.", color=discord.Color.greyple())
                await msg.edit(embed=embed, view=self)
            except Exception:
                pass
