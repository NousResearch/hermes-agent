from __future__ import annotations

"""Finite-choice picker support for the Discord platform adapter."""

import asyncio
import secrets
import time
from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult
from gateway.choice_picker import (
    ChoicePage,
    ChoiceProgress,
    PAGE_TIMEOUT_SECONDS,
    choice_action,
    choice_index,
    choice_label,
)


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
        reusable = (metadata or {}).get("choice_pages") is True
        if reusable:
            if not (metadata or {}).get("requester_user_id"):
                return SendResult(success=False, error="Requester required")
            choices = ChoicePage(title, choices).choices
        target_id = chat_id
        if metadata and metadata.get("thread_id"):
            target_id = metadata["thread_id"]

        channel = adapter._client.get_channel(int(target_id))
        if not channel:
            channel = await adapter._client.fetch_channel(int(target_id))

        navigation = reusable or any(choice.get("full_width") for choice in choices)
        first_line = title.splitlines()[0] if title else "Choose an option"
        embed = discord_sdk.Embed(
            title=first_line[:256] if reusable or navigation else f"⚙ {first_line}",
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
            **({"reusable": True, "channel_id": str(target_id)} if reusable else {}),
        )

        msg = await channel.send(embed=embed, view=view)
        view._message = msg
        if reusable:
            view.arm_expiry()
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
            reusable: bool = False,
            channel_id: Optional[str] = None,
        ):
            super().__init__(timeout=120)
            if reusable:
                if not requester_user_id:
                    raise ValueError("Requester required")
                choices = ChoicePage("Choose an option", choices).choices
            self.choices = list(choices)[:max_options]
            self.on_choice_selected = on_choice_selected
            self.allowed_user_ids = allowed_user_ids
            self.allowed_role_ids = allowed_role_ids or set()
            self.requester_user_id = requester_user_id
            self.navigation = navigation or reusable
            self.resolved = False
            self.busy = False
            self.token = secrets.token_hex(4) if reusable else ""
            self.revision = 0
            self.channel_id = channel_id
            self.expires_at = (
                None if reusable else time.monotonic() + PAGE_TIMEOUT_SECONDS
            )
            self._message = None
            self._expiry_handle = None
            self._render_choices()

        def arm_expiry(self):
            if self.expires_at is None:
                self.expires_at = time.monotonic() + PAGE_TIMEOUT_SECONDS
            self._expiry_handle = asyncio.get_running_loop().call_later(
                max(0, self.expires_at - time.monotonic()),
                lambda: asyncio.create_task(self.on_timeout()),
            )

        def _stop(self):
            if self._expiry_handle:
                self._expiry_handle.cancel()
                self._expiry_handle = None
            self.stop()

        def _render_choices(self):
            self.clear_items()
            options = []
            for index, choice in enumerate(self.choices):
                label = (
                    choice_label(choice, field_limit)
                    if self.token
                    else str(choice.get("label") or choice.get("value") or "")
                )
                options.append(
                    discord.SelectOption(
                        label=truncate_component_text(label, field_limit),
                        value=(
                            choice_action(self.token, self.revision, index)
                            if self.token
                            else str(choice.get("value") or "")
                        ),
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

        def _is_retired(self) -> bool:
            is_finished = getattr(self, "is_finished", None)
            return self.resolved or (
                callable(is_finished) and is_finished() is True
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
            if self.token and (
                self._message is None
                or getattr(getattr(interaction, "message", None), "id", None)
                != self._message.id
                or str(interaction.channel_id) != self.channel_id
            ):
                await interaction.response.send_message(
                    "This menu belongs to another conversation.", ephemeral=True
                )
                return
            if self.token and time.monotonic() > self.expires_at:
                await interaction.response.defer()
                await self.on_timeout()
                return
            if self.resolved or self.busy:
                await interaction.response.defer()
                return

            values = interaction.data.get("values")
            if (
                not isinstance(values, list)
                or len(values) != 1
                or not isinstance(values[0], str)
            ):
                await interaction.response.defer()
                return
            value = values[0]
            if self.token:
                index = choice_index(
                    value, self.token, self.revision, len(self.choices)
                )
                if index is None:
                    await interaction.response.send_message(
                        "This page is no longer current.", ephemeral=True
                    )
                    return
                value = str(self.choices[index]["value"])
            elif value not in [
                str(choice.get("value") or "") for choice in self.choices
            ]:
                await interaction.response.defer()
                return
            self.busy = True
            if not self.token:
                self.resolved = True
            try:
                await interaction.response.defer()
                if self.token:
                    if self._is_retired():
                        return
                    if time.monotonic() > self.expires_at:
                        await self.on_timeout()
                        return
                result_text = await self.on_choice_selected(
                    str(interaction.channel_id), value
                )
                if isinstance(result_text, ChoiceProgress):
                    if not self.token:
                        raise TypeError("Reusable choice pages were not enabled")
                    if self._is_retired():
                        return
                    self.clear_items()
                    await interaction.edit_original_response(
                        embed=discord.Embed(
                            description=result_text.text, color=discord.Color.blue()
                        ),
                        view=self,
                    )
                    if self._is_retired() or time.monotonic() > self.expires_at:
                        await self._close_expired()
                        return
                    result_text = await result_text.complete()
                if self.token and self._is_retired():
                    return
                if self.token and time.monotonic() > self.expires_at:
                    await self.on_timeout()
                    return
                if isinstance(result_text, ChoicePage):
                    if not self.token:
                        raise TypeError("Reusable choice pages were not enabled")
                    self.choices = result_text.choices
                    self.revision += 1
                    self._render_choices()
                    lines = result_text.title.splitlines()
                    embed = discord.Embed(
                        title=lines[0][:256],
                        description="\n".join(lines[1:]) or None,
                        color=discord.Color.blue(),
                    )
                    await interaction.edit_original_response(embed=embed, view=self)
                    if self._is_retired() or time.monotonic() > self.expires_at:
                        await self._close_expired()
                        return
                    self.busy = False
                    return
                if not isinstance(result_text, str):
                    raise TypeError("Choice callback must return text or ChoicePage")
            except asyncio.CancelledError:
                self.resolved = True
                self.clear_items()
                self._stop()
                if self.token and self._message is not None:
                    try:
                        await self._message.edit(view=self)
                    except Exception:
                        pass
                raise
            except Exception as exc:
                logger.error("Choice picker selection failed: %s", exc)
                result_text = (
                    "Unable to update menu."
                    if self.token
                    else f"Error applying selection: {exc}"
                )

            if self.token and self._is_retired():
                return
            embed = discord.Embed(
                description=result_text,
                color=(
                    discord.Color.blue() if self.navigation else discord.Color.green()
                ),
            )
            self.clear_items()
            self.resolved = True
            self._stop()
            await interaction.edit_original_response(embed=embed, view=self)

        async def on_timeout(self):
            if self.resolved:
                return
            await self._close_expired()

        async def _close_expired(self):
            self.resolved = True
            self.clear_items()
            self._stop()
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
