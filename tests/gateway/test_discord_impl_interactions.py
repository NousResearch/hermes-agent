from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest

from gateway.platforms.base import MessageType


class FakeView:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.children = []


class FakeEmbed:
    def __init__(self):
        self.color = None
        self.footer_text = None

    def set_footer(self, *, text):
        self.footer_text = text


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator


def _load_interactions_module():
    class FakeIntents:
        @staticmethod
        def default():
            return SimpleNamespace()

    discord_mod = ModuleType("discord")
    discord_mod.__file__ = "mock-discord.py"
    discord_mod.Message = type("Message", (), {})
    discord_mod.Intents = FakeIntents
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.ui = SimpleNamespace(
        View=FakeView,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(green=1, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(
        green=lambda: "green",
        blue=lambda: "blue",
        red=lambda: "red",
        orange=lambda: "orange",
    )
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(
        is_loaded=lambda: True,
        load_opus=lambda *_args, **_kwargs: None,
        Decoder=MagicMock,
    )
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    sys.modules["discord"] = discord_mod
    ext_mod = ModuleType("discord.ext")
    commands_mod = ModuleType("discord.ext.commands")
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules["discord.ext"] = ext_mod
    sys.modules["discord.ext.commands"] = commands_mod
    return importlib.reload(importlib.import_module("gateway.platforms.discord_impl.interactions"))


interactions = _load_interactions_module()


def _build_adapter():
    return SimpleNamespace(
        build_source=lambda **kwargs: SimpleNamespace(**kwargs),
        _run_simple_slash=AsyncMock(),
        _build_slash_event=MagicMock(return_value=SimpleNamespace()),
        handle_message=AsyncMock(),
        _handle_thread_create_slash=AsyncMock(),
    )


def test_build_slash_event_constructs_message_event():
    adapter = _build_adapter()
    interaction = SimpleNamespace(
        channel=SimpleNamespace(
            name="general",
            guild=SimpleNamespace(name="Hermes"),
            topic="topic",
        ),
        channel_id=123,
        user=SimpleNamespace(id=42, display_name="Jezza"),
    )

    event = interactions.build_slash_event(adapter, interaction, "/status")

    assert event.text == "/status"
    assert event.message_type is MessageType.COMMAND
    assert event.source.chat_id == "123"
    assert event.source.chat_name == "Hermes / #general"
    assert event.source.chat_topic == "topic"


def test_register_slash_commands_registers_expected_names():
    tree = FakeTree()
    adapter = _build_adapter()

    interactions.register_slash_commands(tree, adapter)

    assert set(tree.commands) == {
        "new",
        "reset",
        "model",
        "reasoning",
        "personality",
        "retry",
        "undo",
        "status",
        "sethome",
        "stop",
        "compress",
        "title",
        "resume",
        "usage",
        "provider",
        "help",
        "insights",
        "reload-mcp",
        "voice",
        "update",
        "thread",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "expected_color", "expected_footer", "permanent_calls"),
    [
        ("allow_once", "green", "allow_once by Jezza", 0),
        ("allow_always", "blue", "allow_always by Jezza", 1),
        ("deny", "red", "deny by Jezza", 0),
    ],
)
async def test_exec_approval_view_button_callbacks_resolve_correctly(
    monkeypatch,
    method_name,
    expected_color,
    expected_footer,
    permanent_calls,
):
    approve_permanent = MagicMock()
    approval_mod = ModuleType("tools.approval")
    approval_mod.approve_permanent = approve_permanent
    monkeypatch.setitem(sys.modules, "tools.approval", approval_mod)

    view = interactions.ExecApprovalView("approval-1", {"42"})
    view.children = [SimpleNamespace(disabled=False), SimpleNamespace(disabled=False)]
    embed = FakeEmbed()
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=42, display_name="Jezza"),
        message=SimpleNamespace(embeds=[embed]),
        response=SimpleNamespace(
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
        ),
    )

    await getattr(view, method_name)(interaction, None)

    assert view.resolved is True
    assert embed.color == expected_color
    assert embed.footer_text == expected_footer
    assert all(child.disabled for child in view.children)
    interaction.response.edit_message.assert_awaited_once_with(embed=embed, view=view)
    assert approve_permanent.call_count == permanent_calls


@pytest.mark.asyncio
async def test_exec_approval_view_rejects_unauthorized_user():
    view = interactions.ExecApprovalView("approval-1", {"42"})
    view.children = [SimpleNamespace(disabled=False)]
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=7, display_name="Mallory"),
        message=SimpleNamespace(embeds=[FakeEmbed()]),
        response=SimpleNamespace(
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
        ),
    )

    await view.deny(interaction, None)

    assert view.resolved is False
    interaction.response.edit_message.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once_with(
        "You're not authorized to approve commands~",
        ephemeral=True,
    )
