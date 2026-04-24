"""Regression tests for the Discord /model picker."""

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest


def _ensure_discord_mock():
    existing = sys.modules.get("discord")
    if isinstance(existing, ModuleType) and getattr(existing, "__file__", None):
        return

    class _FakeView:
        def __init__(self, timeout=None):
            self.timeout = timeout
            self.children = []

        def add_item(self, item):
            self.children.append(item)

        def clear_items(self):
            self.children.clear()

    class _FakeSelect:
        def __init__(self, *, placeholder, options, custom_id):
            self.placeholder = placeholder
            self.options = options
            self.custom_id = custom_id
            self.callback = None
            self.disabled = False

    class _FakeButton:
        def __init__(self, *, label, style, custom_id=None, emoji=None, url=None, disabled=False, row=None, sku_id=None):
            self.label = label
            self.style = style
            self.custom_id = custom_id
            self.emoji = emoji
            self.url = url
            self.disabled = disabled
            self.row = row
            self.sku_id = sku_id
            self.callback = None

    class _FakeSelectOption:
        def __init__(self, *, label, value, description=None):
            self.label = label
            self.value = value
            self.description = description

    class _FakeEmbed:
        def __init__(self, *, title, description, color):
            self.title = title
            self.description = description
            self.color = color

    class _FakeColor:
        @staticmethod
        def green():
            return "green"

        @staticmethod
        def blue():
            return "blue"

        @staticmethod
        def red():
            return "red"

        @staticmethod
        def greyple():
            return "greyple"

    class _FakeButtonStyle:
        green = "green"
        grey = "grey"
        red = "red"
        blurple = "blurple"

    discord_mod = sys.modules.get("discord") or MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.Message = type("Message", (), {})
    discord_mod.SelectOption = _FakeSelectOption
    discord_mod.Embed = _FakeEmbed
    discord_mod.Color = _FakeColor
    discord_mod.ButtonStyle = _FakeButtonStyle
    discord_mod.app_commands = getattr(
        discord_mod,
        "app_commands",
        SimpleNamespace(describe=lambda **kwargs: (lambda fn: fn)),
    )
    discord_mod.ui = SimpleNamespace(
        View=_FakeView,
        Select=_FakeSelect,
        Button=_FakeButton,
        button=lambda **kwargs: (lambda fn: fn),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules["discord"] = discord_mod
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import ModelPickerView


@pytest.mark.asyncio
async def test_model_picker_clears_controls_before_running_switch_callback():
    events: list[object] = []

    async def on_model_selected(chat_id: str, model_id: str, provider_slug: str) -> str:
        events.append(("switch", chat_id, model_id, provider_slug))
        return "Model switched"

    async def edit_message(**kwargs):
        events.append(
            (
                "initial-edit",
                kwargs["embed"].title,
                kwargs["embed"].description,
                kwargs["view"],
            )
        )

    async def edit_original_response(**kwargs):
        events.append(("final-edit", kwargs["embed"].title, kwargs["embed"].description, kwargs["view"]))

    view = ModelPickerView(
        providers=[
            {
                "slug": "copilot",
                "name": "GitHub Copilot",
                "models": ["gpt-5.4"],
                "total_models": 1,
                "is_current": True,
            }
        ],
        current_model="gpt-5-mini",
        current_provider="copilot",
        session_key="session-1",
        on_model_selected=on_model_selected,
        allowed_user_ids=set(),
    )
    view._selected_provider = "copilot"

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=123),
        channel_id=456,
        data={"values": ["gpt-5.4"]},
        response=SimpleNamespace(
            defer=AsyncMock(),
            send_message=AsyncMock(),
            edit_message=AsyncMock(side_effect=edit_message),
        ),
        edit_original_response=AsyncMock(side_effect=edit_original_response),
    )

    await view._on_model_selected(interaction)

    assert events == [
        ("initial-edit", "⚙ Switching Model", "Switching to `gpt-5.4`...", None),
        ("switch", "456", "gpt-5.4", "copilot"),
        ("final-edit", "⚙ Model Switched", "Model switched", None),
    ]
    interaction.response.edit_message.assert_awaited_once()
    interaction.response.defer.assert_not_called()
    interaction.edit_original_response.assert_awaited_once()
