"""Focused tests for the Discord quick-action palette."""

from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import (
    DISCORD_QUICK_ACTION_COMMANDS,
    DISCORD_QUICK_ACTION_CONFIRM_COMMANDS,
    DISCORD_QUICK_ACTION_PRIMARY_COMMANDS,
    _quick_action_prompt,
    _quick_action_row,
)


EXPECTED_COMMANDS = (
    "status", "usage", "help",
    "model", "agents", "personality",
    "whoami", "insights", "new",
    "retry", "undo", "stop",
    "compress", "fast", "yolo",
)


def test_quick_action_metadata_is_fixed_and_compact():
    assert DISCORD_QUICK_ACTION_COMMANDS == EXPECTED_COMMANDS
    assert DISCORD_QUICK_ACTION_CONFIRM_COMMANDS == frozenset({"stop", "yolo"})
    assert DISCORD_QUICK_ACTION_PRIMARY_COMMANDS == frozenset({
        "model", "personality", "retry", "compress", "fast",
    })


def test_quick_action_layout_fits_discord_button_rows():
    rows = {command: _quick_action_row(command) for command in DISCORD_QUICK_ACTION_COMMANDS}
    assert max(rows.values()) <= 4
    assert [command for command, row in rows.items() if row == 0] == ["status", "usage", "help"]
    assert [command for command, row in rows.items() if row == 4] == ["compress", "fast", "yolo"]


def test_quick_action_confirmation_prompts_are_specific():
    assert _quick_action_prompt("stop") == "Stop the active Hermes response in this Discord thread?"
    assert _quick_action_prompt("yolo") == (
        "Enable YOLO mode for this session and skip dangerous-command approvals?"
    )


def _view_or_skip():
    if not hasattr(discord_platform, "CommandQuickActionsView"):
        pytest.skip("discord.py UI classes are not available")
    adapter = discord_platform.DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    return adapter, discord_platform.CommandQuickActionsView(adapter)


def _button_for(view, command_name: str):
    button = next(child for child in view.children if child.command_name == command_name)
    # The lightweight discord.py test double does not attach item.view like
    # real discord.py does, so set it explicitly for direct callback tests.
    button.view = view
    return button


async def _click_quick_button(button, interaction):
    callback = getattr(type(button), "callback", None)
    if not callable(callback):
        pytest.skip("discord.py Button callback binding is not available")
    await callback(button, interaction)


def test_quick_action_button_styles_use_semantic_groups():
    _adapter, view = _view_or_skip()
    buttons = {child.command_name: child for child in view.children}
    for command in DISCORD_QUICK_ACTION_CONFIRM_COMMANDS:
        assert buttons[command].style == discord_platform.discord.ButtonStyle.red
    for command in DISCORD_QUICK_ACTION_PRIMARY_COMMANDS:
        assert buttons[command].style == discord_platform.discord.ButtonStyle.primary


@pytest.mark.asyncio
async def test_quick_action_button_dispatches_through_slash_pipeline(monkeypatch):
    adapter, view = _view_or_skip()
    dispatch = AsyncMock()
    monkeypatch.setattr(discord_platform, "_dispatch_quick_action_slash", dispatch)
    status_button = _button_for(view, "status")

    interaction = object()
    await _click_quick_button(status_button, interaction)

    dispatch.assert_awaited_once_with(
        adapter,
        interaction,
        "/status",
        cleanup_response=False,
    )


@pytest.mark.asyncio
async def test_model_quick_action_opens_existing_interactive_picker_flow_without_deleting_palette(monkeypatch):
    adapter, view = _view_or_skip()
    dispatch = AsyncMock()
    monkeypatch.setattr(discord_platform, "_dispatch_quick_action_slash", dispatch)
    model_button = _button_for(view, "model")

    interaction = object()
    await _click_quick_button(model_button, interaction)

    dispatch.assert_awaited_once_with(
        adapter,
        interaction,
        "/model",
        cleanup_response=False,
    )


@pytest.mark.asyncio
async def test_fast_quick_action_opens_select_menu():
    adapter, view = _view_or_skip()
    adapter._run_simple_slash = AsyncMock()
    fast_button = _button_for(view, "fast")

    response = type("ResponseStub", (), {})()
    response.send_message = AsyncMock()
    interaction = type("InteractionStub", (), {"response": response})()
    await _click_quick_button(fast_button, interaction)

    adapter._run_simple_slash.assert_not_awaited()
    response.send_message.assert_awaited_once()
    kwargs = response.send_message.await_args.kwargs
    assert kwargs["ephemeral"] is True
    assert isinstance(kwargs["view"], discord_platform.FastQuickActionView)
    select = kwargs["view"].children[0]
    assert [option.value for option in select.options] == ["status", "fast", "normal"]


@pytest.mark.asyncio
async def test_personality_quick_action_opens_select_menu(monkeypatch):
    adapter, view = _view_or_skip()
    adapter._run_simple_slash = AsyncMock()
    monkeypatch.setattr(
        discord_platform,
        "_load_quick_action_personalities",
        lambda: {"friendly": "warm", "concise": {"description": "short"}},
        raising=False,
    )
    personality_button = _button_for(view, "personality")

    response = type("ResponseStub", (), {})()
    response.send_message = AsyncMock()
    interaction = type("InteractionStub", (), {"response": response})()
    await _click_quick_button(personality_button, interaction)

    adapter._run_simple_slash.assert_not_awaited()
    response.send_message.assert_awaited_once()
    kwargs = response.send_message.await_args.kwargs
    assert kwargs["ephemeral"] is True
    assert isinstance(kwargs["view"], discord_platform.PersonalityQuickActionView)
    select = kwargs["view"].children[0]
    assert [option.value for option in select.options] == ["none", "friendly", "concise"]


@pytest.mark.asyncio
async def test_personality_quick_action_rejects_unauthorized_before_loading_config(monkeypatch):
    adapter, view = _view_or_skip()
    adapter._allowed_user_ids = {"111"}
    adapter._notify_unauthorized_slash = AsyncMock()
    load_personalities = Mock(return_value={"secret": {"description": "do not leak"}})
    monkeypatch.setattr(
        discord_platform,
        "_load_quick_action_personalities",
        load_personalities,
        raising=False,
    )
    personality_button = _button_for(view, "personality")
    response = type("ResponseStub", (), {})()
    response.send_message = AsyncMock()
    user = type("UserStub", (), {"id": 222, "name": "outsider"})()
    interaction = type(
        "InteractionStub",
        (),
        {
            "response": response,
            "user": user,
            "channel": None,
            "channel_id": 333,
            "guild": None,
            "guild_id": 444,
        },
    )()

    await _click_quick_button(personality_button, interaction)

    load_personalities.assert_not_called()
    response.send_message.assert_awaited_once_with(
        "You're not authorized to use this command.",
        ephemeral=True,
    )


@pytest.mark.asyncio
async def test_quick_action_selects_dispatch_existing_commands(monkeypatch):
    if not hasattr(discord_platform, "FastQuickActionView"):
        pytest.skip("discord.py UI classes are not available")
    adapter = discord_platform.DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    dispatch = AsyncMock()
    monkeypatch.setattr(discord_platform, "_dispatch_quick_action_slash", dispatch)
    view = discord_platform.FastQuickActionView(adapter)
    interaction = type(
        "InteractionStub",
        (),
        {
            "data": {"values": ["normal"]},
            "response": type("ResponseStub", (), {"send_message": AsyncMock()})(),
            "edit_original_response": AsyncMock(),
        },
    )()

    await view._on_selected(interaction)

    dispatch.assert_awaited_once_with(
        adapter,
        interaction,
        "/fast normal",
        cleanup_response=False,
    )
    interaction.edit_original_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirmed_quick_action_dispatches_after_component_confirmation(monkeypatch):
    if not hasattr(discord_platform, "QuickActionConfirmView"):
        pytest.skip("discord.py UI classes are not available")
    adapter = discord_platform.DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    dispatch = AsyncMock()
    monkeypatch.setattr(discord_platform, "_dispatch_quick_action_slash", dispatch)
    view = discord_platform.QuickActionConfirmView(adapter, "stop")
    interaction = type("InteractionStub", (), {"response": type("ResponseStub", (), {})()})()
    interaction.response.send_message = AsyncMock()

    confirm_button = next(
        (child for child in view.children if getattr(child, "label", None) == "Confirm"),
        None,
    )
    if confirm_button is None or not callable(getattr(confirm_button, "callback", None)):
        pytest.skip("discord.py Button callback binding is not available")
    await confirm_button.callback(interaction)

    dispatch.assert_awaited_once_with(
        adapter,
        interaction,
        "/stop",
        cleanup_response=True,
    )
