from __future__ import annotations

"""Finite-choice picker support for the Discord platform adapter."""

from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult


# Rebound by ``define_choice_picker_view`` without importing the optional SDK.
discord: Any = None


async def send_choice_picker(
    adapter: Any,
    chat_id: str,
    title: str,
    choices: list,
    session_key: str,
    on_choice_selected: Any,
    metadata: Optional[dict[str, Any]] = None,
    *,
    discord_sdk: Any,
    discord_available: bool,
    view_class: Optional[type],
    logger: Any,
) -> SendResult:
    """Send a flat select-menu choice picker (one selection to one value)."""
    if not adapter._client or not discord_available:
        return SendResult(success=False, error="Not connected")

    try:
        target_id = chat_id
        if metadata and metadata.get("thread_id"):
            target_id = metadata["thread_id"]

        channel = adapter._client.get_channel(int(target_id))
        if not channel:
            channel = await adapter._client.fetch_channel(int(target_id))

        navigation = any(choice.get("full_width") for choice in choices)
        first_line = title.splitlines()[0] if title else "Choose an option"
        embed = discord_sdk.Embed(
            title=first_line if navigation else f"⚙ {first_line}",
            description="\n".join(title.splitlines()[1:]) or None,
            color=discord_sdk.Color.blue(),
        )

        view = view_class(
            choices=choices,
            on_choice_selected=on_choice_selected,
            allowed_user_ids=adapter._allowed_user_ids,
            allowed_role_ids=adapter._allowed_role_ids,
            requester_user_id=str((metadata or {}).get("requester_user_id") or "")
            or None,
            navigation=navigation,
        )

        msg = await channel.send(embed=embed, view=view)
        view._message = msg
        return SendResult(success=True, message_id=str(msg.id))
    except Exception as exc:
        logger.warning("[%s] send_choice_picker failed: %s", adapter.name, exc)
        return SendResult(success=False, error=str(exc))


def define_choice_picker_view(
    *,
    discord_sdk: Any,
    component_check_auth: Callable[..., bool],
    truncate_component_text: Callable[[str, int], str],
    logger: Any,
    max_options: int,
    field_limit: int,
) -> type:
    """Build the view class after discord.py is present."""
    global discord
    discord = discord_sdk

    class ChoicePickerView(discord.ui.View):
        """Flat select-menu view for finite-choice commands (/reasoning, /fast).

        One dropdown, one selection, done — the generic single-level companion
        to ``ModelPickerView``. Auth gating mirrors ``ExecApprovalView``.
        Times out after 2 minutes.
        """

        def __init__(
            self,
            choices: list,
            on_choice_selected,
            allowed_user_ids: set,
            allowed_role_ids: Optional[set] = None,
            requester_user_id: Optional[str] = None,
            navigation: bool = False,
        ):
            super().__init__(timeout=120)
            self.choices = list(choices)[:max_options]
            self.on_choice_selected = on_choice_selected
            self.allowed_user_ids = allowed_user_ids
            self.allowed_role_ids = allowed_role_ids or set()
            self.requester_user_id = requester_user_id
            self.navigation = navigation
            self.resolved = False
            self._message = None

            options = []
            for choice in self.choices:
                label = str(choice.get("label") or choice.get("value") or "")
                options.append(
                    discord.SelectOption(
                        label=truncate_component_text(label, field_limit),
                        value=str(choice.get("value") or ""),
                        description="current" if choice.get("is_current") else None,
                    )
                )
            select = discord.ui.Select(
                placeholder="Choose an option...",
                options=options,
            )
            select.callback = self._on_select
            self.add_item(select)

        def _check_auth(self, interaction: discord.Interaction) -> bool:
            if self.requester_user_id and self.requester_user_id != str(
                getattr(getattr(interaction, "user", None), "id", "")
            ):
                return False
            return component_check_auth(
                interaction, self.allowed_user_ids, self.allowed_role_ids
            )

        async def _on_select(self, interaction: discord.Interaction):
            if not self._check_auth(interaction):
                await interaction.response.send_message(
                    (
                        "⛔ You are not authorized to use this menu."
                        if self.navigation
                        else "⛔ You are not authorized to change this setting."
                    ),
                    ephemeral=True,
                )
                return
            if self.resolved:
                await interaction.response.defer()
                return
            self.resolved = True

            value = interaction.data.get("values", [""])[0]
            try:
                result_text = await self.on_choice_selected(
                    str(interaction.channel_id), value
                )
            except Exception as exc:
                logger.error("Choice picker selection failed: %s", exc)
                result_text = f"Error applying selection: {exc}"

            embed = discord.Embed(
                description=result_text,
                color=(
                    discord.Color.blue() if self.navigation else discord.Color.green()
                ),
            )
            self.clear_items()
            self.stop()
            await interaction.response.edit_message(embed=embed, view=self)

        async def on_timeout(self):
            if self.resolved:
                return
            msg = self._message
            if msg is not None:
                try:
                    embed = discord.Embed(
                        description=(
                            "⏱ Menu expired — run the command again."
                            if self.navigation
                            else "⏱ Selection expired — no change made."
                        ),
                        color=discord.Color.greyple(),
                    )
                    self.clear_items()
                    await msg.edit(embed=embed, view=self)
                except Exception:
                    pass

    return ChoicePickerView
