from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.platforms.base import MessageType


class FakeView:
    def __init__(self, timeout=None):
        self.timeout = timeout
        self.children = []

    def add_item(self, item):
        item.view = self
        self.children.append(item)


class FakeButton:
    def __init__(
        self,
        *,
        label=None,
        style=None,
        custom_id=None,
        row=None,
        disabled=False,
        emoji=None,
        url=None,
    ):
        self.label = label
        self.style = style
        self.custom_id = custom_id
        self.row = row
        self.disabled = disabled
        self.emoji = emoji
        self.url = url
        self.view = None


class FakeSelect:
    def __init__(
        self,
        *,
        placeholder=None,
        min_values=1,
        max_values=1,
        options=None,
        custom_id=None,
        row=None,
        disabled=False,
    ):
        self.placeholder = placeholder
        self.min_values = min_values
        self.max_values = max_values
        self.options = list(options or [])
        self.custom_id = custom_id
        self.row = row
        self.disabled = disabled
        self.values = []
        self.view = None


class FakeUserSelect(FakeSelect):
    pass


class FakeRoleSelect(FakeSelect):
    pass


class FakeMentionableSelect(FakeSelect):
    pass


class FakeChannelSelect(FakeSelect):
    pass


class FakeModal:
    def __init__(self, *, title=None, custom_id=None, timeout=None):
        self.title = title
        self.custom_id = custom_id
        self.timeout = timeout
        self.children = []

    def add_item(self, item):
        self.children.append(item)


class FakeTextInput:
    def __init__(
        self,
        *,
        label,
        placeholder=None,
        default=None,
        required=True,
        min_length=None,
        max_length=None,
        style=None,
    ):
        self.label = label
        self.placeholder = placeholder
        self.default = default
        self.required = required
        self.min_length = min_length
        self.max_length = max_length
        self.style = style
        self.value = default or ""


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
        Button=FakeButton,
        Select=FakeSelect,
        UserSelect=FakeUserSelect,
        RoleSelect=FakeRoleSelect,
        MentionableSelect=FakeMentionableSelect,
        ChannelSelect=FakeChannelSelect,
        Modal=FakeModal,
        TextInput=FakeTextInput,
        button=lambda *a, **k: (lambda fn: fn),
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        primary=1,
        secondary=2,
        success=3,
        danger=4,
        link=5,
        green=3,
        blurple=1,
        red=4,
    )
    discord_mod.Color = SimpleNamespace(
        green=lambda: "green",
        blue=lambda: "blue",
        red=lambda: "red",
        orange=lambda: "orange",
    )
    discord_mod.SelectOption = lambda **kwargs: SimpleNamespace(**kwargs)
    discord_mod.TextStyle = SimpleNamespace(short="short", paragraph="paragraph")
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        autocomplete=lambda **kwargs: (lambda fn: fn),
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
    for module_name in (
        "gateway.platforms.discord_impl.components",
        "gateway.platforms.discord_impl.native_commands",
        "gateway.platforms.discord_impl.interactions",
    ):
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
    return sys.modules["gateway.platforms.discord_impl.interactions"]


interactions = _load_interactions_module()


def _build_adapter():
    return SimpleNamespace(
        build_source=lambda **kwargs: SessionSource(platform=Platform.DISCORD, **kwargs),
        _run_simple_slash=AsyncMock(),
        _invoke_native_slash_command=AsyncMock(),
        _send_native_slash_content=AsyncMock(),
        _build_slash_event=MagicMock(return_value=SimpleNamespace()),
        handle_message=AsyncMock(),
        _handle_thread_create_slash=AsyncMock(),
        _resolve_exec_approval=AsyncMock(return_value="resolved"),
        _component_runtime=interactions.create_component_runtime(),
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
    assert isinstance(event.metadata.get("session_source"), SessionSource)
    assert event.metadata["session_source"].session_namespace == "slash:42"
    assert event.metadata["command_target_source"].chat_id == "123"


def test_register_slash_commands_registers_expected_names():
    tree = FakeTree()
    adapter = _build_adapter()

    interactions.register_slash_commands(tree, adapter)

    assert set(tree.commands) == {
        "new",
        "reset",
        "help",
        "commands",
        "context",
        "export-session",
        "export",
        "whoami",
        "focus",
        "unfocus",
        "agents",
        "session",
        "id",
        "approve",
        "allowlist",
        "config",
        "debug",
        "model",
        "models",
        "reasoning",
        "think",
        "personality",
        "retry",
        "undo",
        "status",
        "sethome",
        "stop",
        "compact",
        "compress",
        "title",
        "resume",
        "usage",
        "provider",
        "help",
        "insights",
        "reload-mcp",
        "skill",
        "subagents",
        "kill",
        "steer",
        "tell",
        "acp",
        "bash",
        "voice",
        "vc",
        "send",
        "activation",
        "update",
        "restart",
        "dock-telegram",
        "dock-discord",
        "dock-slack",
        "thread",
    }


def test_extract_inline_shortcut_finds_first_supported_command():
    command, remaining = interactions.native_commands.extract_inline_shortcut("hey /status please")

    assert command == "status"
    assert remaining == "hey please"


def test_extract_inline_shortcut_returns_none_for_plain_text():
    command, remaining = interactions.native_commands.extract_inline_shortcut("just chatting here")

    assert command is None
    assert remaining == "just chatting here"


@pytest.mark.asyncio
async def test_native_dispatch_opens_fallback_menu_for_missing_discrete_arg():
    adapter = _build_adapter()
    spec = next(spec for spec in interactions.native_commands.get_command_specs() if spec.name == "think")
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=42, display_name="Jezza"),
        message=None,
        response=SimpleNamespace(
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
            is_done=lambda: False,
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await interactions.native_commands._dispatch(adapter, interaction, spec, effort="")

    interaction.response.send_message.assert_awaited_once()
    args = interaction.response.send_message.await_args.args
    kwargs = interaction.response.send_message.await_args.kwargs
    assert kwargs["ephemeral"] is True
    assert "Choose `effort` for `/think`." == args[0]
    assert kwargs["view"] is not None
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_dispatch_route_sends_returned_response():
    adapter = _build_adapter()
    adapter._invoke_native_slash_command.return_value = "focused"
    spec = next(spec for spec in interactions.native_commands.get_command_specs() if spec.name == "focus")
    interaction = SimpleNamespace(
        response=SimpleNamespace(
            defer=AsyncMock(),
            send_message=AsyncMock(),
            is_done=lambda: True,
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await interactions.native_commands._dispatch(adapter, interaction, spec, name="release-room")

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    adapter._invoke_native_slash_command.assert_awaited_once_with(
        interaction,
        "/focus release-room",
    )
    adapter._send_native_slash_content.assert_awaited_once_with(interaction, "focused")


@pytest.mark.asyncio
async def test_native_autocomplete_filters_discrete_choices():
    adapter = _build_adapter()
    spec = next(spec for spec in interactions.native_commands.get_command_specs() if spec.name == "reasoning")
    arg = spec.args[0]

    result = await interactions.native_commands._autocomplete_choices(
        adapter,
        spec,
        arg,
        SimpleNamespace(),
        "hi",
    )

    assert [choice.value for choice in result] == ["off", "hide", "high", "xhigh"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "expected_color", "expected_footer"),
    [
        ("allow_once", "green", "allow_once by Jezza"),
        ("allow_always", "blue", "allow_always by Jezza"),
        ("deny", "red", "deny by Jezza"),
    ],
)
async def test_exec_approval_view_button_callbacks_resolve_correctly(
    method_name,
    expected_color,
    expected_footer,
):
    adapter = _build_adapter()
    view = interactions.create_exec_approval_view(adapter, "approval-1", {"42"})
    embed = FakeEmbed()
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=42, display_name="Jezza"),
        message=SimpleNamespace(embeds=[embed]),
        response=SimpleNamespace(
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    button_labels = {
        "allow_once": "Allow Once",
        "allow_always": "Always Allow",
        "deny": "Deny",
    }
    button = next(child for child in view.children if child.label == button_labels[method_name])
    await button.callback(interaction)

    assert embed.color == expected_color
    assert embed.footer_text == expected_footer.replace("_", "-")
    assert all(child.disabled for child in view.children)
    interaction.response.edit_message.assert_awaited_once_with(embed=embed, view=view)
    interaction.followup.send.assert_awaited_once_with("resolved", ephemeral=True)
    assert adapter._resolve_exec_approval.await_count == 1
    decision = adapter._resolve_exec_approval.await_args.kwargs["decision"]
    assert decision == expected_footer.split(" by ")[0].replace("_", "-")
    assert adapter._resolve_exec_approval.await_args.kwargs["approval_id"] == "approval-1"


@pytest.mark.asyncio
async def test_exec_approval_view_rejects_unauthorized_user():
    adapter = _build_adapter()
    view = interactions.create_exec_approval_view(adapter, "approval-1", {"42"})
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=7, display_name="Mallory"),
        message=SimpleNamespace(embeds=[FakeEmbed()]),
        response=SimpleNamespace(
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
        ),
    )

    button = next(child for child in view.children if child.label == "Deny")
    await button.callback(interaction)

    interaction.response.edit_message.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once_with(
        "You're not authorized to use this interaction~",
        ephemeral=True,
    )
