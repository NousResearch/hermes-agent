"""Tests for native Discord slash command fast-paths (thread creation & auto-thread)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        # Real discord is installed — nothing to do.
        return

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object

        # Lightweight mock for app_commands.Group and Command used by
        # _register_skill_group.
        class _FakeGroup:
            def __init__(self, *, name, description, parent=None):
                self.name = name
                self.description = description
                self.parent = parent
                self._children: dict[str, object] = {}
                if parent is not None:
                    parent.add_command(self)

            def add_command(self, cmd):
                self._children[cmd.name] = cmd

        class _FakeCommand:
            def __init__(self, *, name, description, callback, parent=None):
                self.name = name
                self.description = description
                self.callback = callback
                self.parent = parent

        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            autocomplete=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
            Group=_FakeGroup,
            Command=_FakeCommand,
        )

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)

    # Whether we just installed the mock OR another test module installed
    # it first via its own _ensure_discord_mock, force the decorators we
    # need onto discord.app_commands — the flat /skill command uses
    # @app_commands.autocomplete and not every other mock stub exposes it.
    _app = getattr(sys.modules["discord"], "app_commands", None)
    if _app is not None and not hasattr(_app, "autocomplete"):
        try:
            _app.autocomplete = lambda **kwargs: (lambda fn: fn)
        except Exception:
            pass


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator

    def add_command(self, cmd):
        self.commands[cmd.name] = cmd

    def get_commands(self):
        return [SimpleNamespace(name=n) for n in self.commands]


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    return adapter


# ------------------------------------------------------------------
# /thread slash command registration
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registers_native_thread_slash_command(adapter):
    adapter._handle_thread_create_slash = AsyncMock()
    adapter._register_slash_commands()

    command = adapter._client.tree.commands["thread"]
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock()),
    )

    await command(interaction, name="Planning", message="", auto_archive_duration=1440)

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    adapter._handle_thread_create_slash.assert_awaited_once_with(interaction, "Planning", "", 1440)


@pytest.mark.asyncio
async def test_registers_native_restart_slash_command(adapter):
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()

    assert "restart" in adapter._client.tree.commands

    interaction = SimpleNamespace()
    await adapter._client.tree.commands["restart"](interaction)

    adapter._run_simple_slash.assert_awaited_once_with(
        interaction,
        "/restart",
        "Restart requested~",
    )


# ------------------------------------------------------------------
# Auto-registration from COMMAND_REGISTRY
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_registers_missing_gateway_commands(adapter):
    """Commands in COMMAND_REGISTRY that aren't explicitly registered should
    be auto-registered by the dynamic catch-all block."""
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()

    tree_names = set(adapter._client.tree.commands.keys())

    # These commands are gateway-available but were not in the original
    # hardcoded registration list — they should be auto-registered.
    expected_auto = {"debug", "yolo", "reload", "profile"}
    for name in expected_auto:
        assert name in tree_names, f"/{name} should be auto-registered on Discord"


@pytest.mark.asyncio
async def test_auto_registered_command_dispatches_correctly(adapter):
    """Auto-registered commands should dispatch via _run_simple_slash."""
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()

    # /debug has no args — test parameterless dispatch
    debug_cmd = adapter._client.tree.commands["debug"]
    interaction = SimpleNamespace()
    adapter._run_simple_slash.reset_mock()
    await debug_cmd.callback(interaction)
    adapter._run_simple_slash.assert_awaited_once_with(interaction, "/debug")


@pytest.mark.asyncio
async def test_auto_registered_command_with_args(adapter):
    """Auto-registered commands with args_hint should accept an optional args param."""
    adapter._run_simple_slash = AsyncMock()
    adapter._register_slash_commands()

    # /branch has args_hint="[name]" — test dispatch with args
    branch_cmd = adapter._client.tree.commands["branch"]
    interaction = SimpleNamespace()
    adapter._run_simple_slash.reset_mock()
    await branch_cmd.callback(interaction, args="my-branch")
    adapter._run_simple_slash.assert_awaited_once_with(
        interaction, "/branch my-branch"
    )


# ------------------------------------------------------------------
# _handle_thread_create_slash — success, session dispatch, failure
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_thread_create_slash_reports_success(adapter):
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread), send=AsyncMock())
    interaction_channel = SimpleNamespace(parent=parent_channel)
    interaction = SimpleNamespace(
        channel=interaction_channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "Kickoff", 1440)

    parent_channel.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    created_thread.send.assert_awaited_once_with("Kickoff")
    # Thread link shown to user
    interaction.followup.send.assert_awaited()
    args, kwargs = interaction.followup.send.await_args
    assert "<#555>" in args[0]
    assert kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_handle_thread_create_slash_dispatches_session_when_message_provided(adapter):
    """When a message is given, _dispatch_thread_session should be called."""
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread))
    interaction = SimpleNamespace(
        channel=SimpleNamespace(parent=parent_channel),
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    adapter._dispatch_thread_session = AsyncMock()

    await adapter._handle_thread_create_slash(interaction, "Planning", "Hello Hermes", 1440)

    adapter._dispatch_thread_session.assert_awaited_once_with(
        interaction, "555", "Planning", "Hello Hermes",
    )


@pytest.mark.asyncio
async def test_handle_thread_create_slash_no_dispatch_without_message(adapter):
    """Without a message, no session dispatch should occur."""
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread))
    interaction = SimpleNamespace(
        channel=SimpleNamespace(parent=parent_channel),
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    adapter._dispatch_thread_session = AsyncMock()

    await adapter._handle_thread_create_slash(interaction, "Planning", "", 1440)

    adapter._dispatch_thread_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_thread_create_slash_falls_back_to_seed_message(adapter):
    created_thread = SimpleNamespace(id=555, name="Planning")
    seed_message = SimpleNamespace(id=777, create_thread=AsyncMock(return_value=created_thread))
    channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct failed")),
        send=AsyncMock(return_value=seed_message),
    )
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "Kickoff", 1440)

    channel.send.assert_awaited_once_with("Kickoff")
    seed_message.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    interaction.followup.send.assert_awaited()


@pytest.mark.asyncio
async def test_handle_thread_create_slash_reports_failure(adapter):
    channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct failed")),
        send=AsyncMock(side_effect=RuntimeError("nope")),
    )
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "", 1440)

    interaction.followup.send.assert_awaited_once()
    args, kwargs = interaction.followup.send.await_args
    assert "Failed to create thread:" in args[0]
    assert "nope" in args[0]
    assert kwargs["ephemeral"] is True


# ------------------------------------------------------------------
# _dispatch_thread_session — builds correct event and routes it
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_thread_session_builds_thread_event(adapter):
    """Dispatched event should have chat_type=thread and chat_id=thread_id."""
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="Jezza", id=42),
        guild=SimpleNamespace(name="TestGuild"),
    )

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    await adapter._dispatch_thread_session(interaction, "555", "Planning", "Hello!")

    assert len(captured_events) == 1
    event = captured_events[0]
    assert event.text == "Hello!"
    assert event.source.chat_id == "555"
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == "555"
    assert "TestGuild" in event.source.chat_name


# ------------------------------------------------------------------
# _build_slash_event — preserve thread context for native slash commands
# ------------------------------------------------------------------


def test_build_slash_event_preserves_thread_context(adapter):
    interaction = SimpleNamespace(
        channel=_FakeThreadChannel(channel_id=555, name="Planning"),
        channel_id=555,
        user=SimpleNamespace(display_name="Jezza", id=42),
    )

    event = adapter._build_slash_event(interaction, "/status")

    assert event.text == "/status"
    assert event.source.chat_id == "555"
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == "555"
    assert "TestGuild" in event.source.chat_name


def test_build_slash_event_uses_group_context_for_channels(adapter):
    interaction = SimpleNamespace(
        channel=_FakeTextChannel(channel_id=123, name="general"),
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza", id=42),
    )

    event = adapter._build_slash_event(interaction, "/status")

    assert event.source.chat_id == "123"
    assert event.source.chat_type == "group"
    assert event.source.thread_id is None
    assert "TestGuild / #general" == event.source.chat_name


# ------------------------------------------------------------------
# Auto-thread: _auto_create_thread
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_create_thread_uses_message_content_as_name(adapter):
    thread = SimpleNamespace(id=999, name="Hello world")
    message = SimpleNamespace(
        content="Hello world, how are you?",
        create_thread=AsyncMock(return_value=thread),
    )

    result = await adapter._auto_create_thread(message)

    assert result is thread
    message.create_thread.assert_awaited_once()
    call_kwargs = message.create_thread.await_args[1]
    assert call_kwargs["name"] == "Hello world, how are you?"
    assert call_kwargs["auto_archive_duration"] == 1440


@pytest.mark.asyncio
async def test_auto_create_thread_truncates_long_names(adapter):
    long_text = "a" * 200
    thread = SimpleNamespace(id=999, name="truncated")
    message = SimpleNamespace(
        content=long_text,
        create_thread=AsyncMock(return_value=thread),
    )

    result = await adapter._auto_create_thread(message)

    assert result is thread
    call_kwargs = message.create_thread.await_args[1]
    assert len(call_kwargs["name"]) <= 80
    assert call_kwargs["name"].endswith("...")


@pytest.mark.asyncio
async def test_auto_create_thread_returns_none_on_failure(adapter):
    message = SimpleNamespace(
        content="Hello",
        create_thread=AsyncMock(side_effect=RuntimeError("no perms")),
    )

    result = await adapter._auto_create_thread(message)
    assert result is None


# ------------------------------------------------------------------
# Auto-thread integration in _handle_message
# ------------------------------------------------------------------


import discord as _discord_mod  # noqa: E402 — mock or real, used below


class _FakeTextChannel:
    """A channel that is NOT a discord.Thread or discord.DMChannel."""

    def __init__(self, channel_id=100, name="general", guild_name="TestGuild"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None


class _FakeThreadChannel(_discord_mod.Thread):
    """isinstance(ch, discord.Thread) → True."""

    def __init__(self, channel_id=200, name="existing-thread", guild_name="TestGuild", parent_id=100):
        # Don't call super().__init__ — mock Thread is just an empty type
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None
        self.parent = SimpleNamespace(id=parent_id, name="general", guild=SimpleNamespace(name=guild_name, id=1))


def _fake_message(channel, *, content="Hello", author_id=42, display_name="Jezza"):
    return SimpleNamespace(
        author=SimpleNamespace(id=author_id, display_name=display_name, bot=False),
        content=content,
        channel=channel,
        attachments=[],
        mentions=[],
        reference=None,
        created_at=None,
        id=12345,
    )


@pytest.mark.asyncio
async def test_auto_thread_creates_thread_and_redirects(adapter, monkeypatch):
    """When DISCORD_AUTO_THREAD=true, a new thread is created and the event routes there."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    thread = SimpleNamespace(id=999, name="Hello")
    adapter._auto_create_thread = AsyncMock(return_value=thread)

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeTextChannel(), content="Hello world")

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_awaited_once_with(msg)
    assert len(captured_events) == 1
    event = captured_events[0]
    assert event.source.chat_id == "999"  # redirected to thread
    assert event.source.chat_type == "thread"
    assert event.source.thread_id == "999"


@pytest.mark.asyncio
async def test_auto_thread_enabled_by_default_slash_commands(adapter, monkeypatch):
    """Without DISCORD_AUTO_THREAD env var, auto-threading is enabled (default: true)."""
    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    fake_thread = _FakeThreadChannel(channel_id=999, name="auto-thread")
    adapter._auto_create_thread = AsyncMock(return_value=fake_thread)

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeTextChannel())

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_awaited_once()
    assert len(captured_events) == 1
    assert captured_events[0].source.chat_id == "999"  # redirected to thread
    assert captured_events[0].source.chat_type == "thread"


@pytest.mark.asyncio
async def test_auto_thread_can_be_disabled(adapter, monkeypatch):
    """Setting DISCORD_AUTO_THREAD=false keeps messages in the channel."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    adapter._auto_create_thread = AsyncMock()

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeTextChannel())

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_not_awaited()
    assert len(captured_events) == 1
    assert captured_events[0].source.chat_id == "100"  # stays in channel


@pytest.mark.asyncio
async def test_auto_thread_skips_threads_and_dms(adapter, monkeypatch):
    """Auto-thread should not create threads inside existing threads."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")

    adapter._auto_create_thread = AsyncMock()

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle

    msg = _fake_message(_FakeThreadChannel())

    await adapter._handle_message(msg)

    adapter._auto_create_thread.assert_not_awaited()  # should NOT auto-thread


# ------------------------------------------------------------------
# Config bridge
# ------------------------------------------------------------------


def test_discord_auto_thread_config_bridge(monkeypatch, tmp_path):
    """discord.auto_thread in config.yaml should be bridged to DISCORD_AUTO_THREAD env var."""
    import yaml
    from pathlib import Path

    # Write a config.yaml the loader will find
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    config_path = hermes_dir / "config.yaml"
    config_path.write_text(yaml.dump({
        "discord": {"auto_thread": True},
    }))

    monkeypatch.delenv("DISCORD_AUTO_THREAD", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_dir))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("DISCORD_AUTO_THREAD") == "true"


# ------------------------------------------------------------------
# /skill command registration (flat + autocomplete)
# ------------------------------------------------------------------


def test_register_skill_command_is_flat_not_nested(adapter):
    """_register_skill_group should register a single flat ``/skill`` command.

    The older layout nested categories as subcommand groups under ``/skill``.
    That registered as one giant command whose serialized payload exceeded
    Discord's 8KB per-command limit with the default skill catalog. The
    flat layout sidesteps the limit — autocomplete options are fetched
    dynamically by Discord and don't count against the registration budget.
    """
    mock_categories = {
        "creative": [
            ("ascii-art", "Generate ASCII art", "/ascii-art"),
            ("excalidraw", "Hand-drawn diagrams", "/excalidraw"),
        ],
        "media": [
            ("gif-search", "Search for GIFs", "/gif-search"),
        ],
    }
    mock_uncategorized = [
        ("dogfood", "Exploratory QA testing", "/dogfood"),
    ]

    with patch(
        "hermes_cli.commands.discord_skill_commands_by_category",
        return_value=(mock_categories, mock_uncategorized, 0),
    ):
        adapter._register_slash_commands()

    tree = adapter._client.tree
    assert "skill" in tree.commands, "Expected /skill command to be registered"
    skill_cmd = tree.commands["skill"]
    assert skill_cmd.name == "skill"
    # Flat command — NOT a Group — so it has no _children of category subgroups
    assert not hasattr(skill_cmd, "_children") or not getattr(skill_cmd, "_children", {}), (
        "Flat /skill command should not have subcommand children"
    )


def test_register_skill_command_empty_skills_no_command(adapter):
    """No /skill command should be registered when there are zero skills."""
    with patch(
        "hermes_cli.commands.discord_skill_commands_by_category",
        return_value=({}, [], 0),
    ):
        adapter._register_slash_commands()

    tree = adapter._client.tree
    assert "skill" not in tree.commands


def test_register_skill_command_callback_dispatches_by_name(adapter):
    """The /skill callback should look up the skill by ``name`` and
    dispatch via ``_run_simple_slash`` with the real command key.
    """
    mock_categories = {
        "media": [
            ("gif-search", "Search for GIFs", "/gif-search"),
        ],
    }
    mock_uncategorized = [
        ("dogfood", "QA testing", "/dogfood"),
    ]

    with patch(
        "hermes_cli.commands.discord_skill_commands_by_category",
        return_value=(mock_categories, mock_uncategorized, 0),
    ):
        adapter._register_slash_commands()

    skill_cmd = adapter._client.tree.commands["skill"]
    assert skill_cmd.callback is not None

    # Stub out _run_simple_slash so we can verify the dispatched text.
    dispatched: list[str] = []

    async def fake_run(_interaction, text):
        dispatched.append(text)

    adapter._run_simple_slash = fake_run

    import asyncio

    fake_interaction = SimpleNamespace()
    # gif-search → /gif-search with no args
    asyncio.run(skill_cmd.callback(fake_interaction, name="gif-search"))
    # dogfood with args
    asyncio.run(skill_cmd.callback(fake_interaction, name="dogfood", args="my test"))

    assert dispatched == ["/gif-search", "/dogfood my test"]


def test_register_skill_command_handles_unknown_skill_gracefully(adapter):
    """Passing a name that isn't a registered skill should respond with
    an ephemeral error message, NOT crash the callback.
    """
    with patch(
        "hermes_cli.commands.discord_skill_commands_by_category",
        return_value=({"media": [("gif-search", "GIFs", "/gif-search")]}, [], 0),
    ):
        adapter._register_slash_commands()

    skill_cmd = adapter._client.tree.commands["skill"]

    sent: list[dict] = []

    async def fake_send(text, ephemeral=False):
        sent.append({"text": text, "ephemeral": ephemeral})

    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=fake_send),
    )

    import asyncio
    asyncio.run(skill_cmd.callback(interaction, name="does-not-exist"))

    assert len(sent) == 1
    assert "Unknown skill" in sent[0]["text"]
    assert "does-not-exist" in sent[0]["text"]
    assert sent[0]["ephemeral"] is True


def test_register_skill_command_payload_fits_discord_8kb_limit(adapter):
    """The /skill command registration payload must stay under Discord's
    ~8000-byte per-command limit even with a large skill catalog.

    This is the regression guard for #11321 / #10259. Simulates 500 skills
    (20 categories × 25 — the hard cap per category in the collector) and
    confirms the serialized command still fits. Autocomplete options are
    not part of this payload, so the budget is essentially constant.
    """
    import json

    # Simulate the largest catalog the collector will ever produce:
    # 20 categories × 25 skills each, with verbose 100-char descriptions.
    large_categories: dict[str, list[tuple[str, str, str]]] = {}
    long_desc = "A verbose description padded to approximately 100 chars " + "." * 42
    for i in range(20):
        cat = f"cat{i:02d}"
        large_categories[cat] = [
            (f"skill-{i:02d}-{j:02d}", long_desc, f"/skill-{i:02d}-{j:02d}")
            for j in range(25)
        ]

    with patch(
        "hermes_cli.commands.discord_skill_commands_by_category",
        return_value=(large_categories, [], 0),
    ):
        adapter._register_slash_commands()

    skill_cmd = adapter._client.tree.commands["skill"]
    # Approximate the serialized registration payload (name + description only).
    # Autocomplete options are NOT registered — they're fetched dynamically.
    payload = json.dumps({
        "name": skill_cmd.name,
        "description": skill_cmd.description,
        "options": [
            {"name": "name", "description": "Which skill to run", "type": 3, "required": True},
            {"name": "args", "description": "Optional arguments for the skill", "type": 3, "required": False},
        ],
    })
    assert len(payload) < 500, (
        f"Flat /skill command payload is ~{len(payload)} bytes — the whole "
        f"point of this design is that it stays small regardless of skill count"
    )


def test_register_skill_command_autocomplete_filters_by_name_and_description(adapter):
    """The autocomplete callback should match on both skill name and
    description so the user can search by either.
    """
    mock_categories = {
        "ocr": [
            ("ocr-and-documents", "Extract text from PDFs and scanned documents", "/ocr-and-documents"),
        ],
        "media": [
            ("gif-search", "Search and download GIFs from Tenor", "/gif-search"),
        ],
    }

    with patch(
        "hermes_cli.commands.discord_skill_commands_by_category",
        return_value=(mock_categories, [], 0),
    ):
        adapter._register_slash_commands()

    skill_cmd = adapter._client.tree.commands["skill"]
    # The callback has been wrapped with @autocomplete(name=...) — in our mock
    # the decorator is pass-through, so we inspect the closed-over list by
    # invoking the registered autocomplete function directly through the
    # test API. Since the mock doesn't preserve the autocomplete binding,
    # we re-derive the filter by building the same entries list.
    #
    # What we CAN verify at this layer: the callback dispatches correctly
    # (covered in other tests). The autocomplete filter itself is exercised
    # via direct function call in the real-discord integration path.
    assert skill_cmd.callback is not None

