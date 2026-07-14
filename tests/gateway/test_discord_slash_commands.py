"""Tests for native Discord slash command fast-paths (thread creation & auto-thread)."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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

from gateway.platforms.base import MessageType  # noqa: E402
from plugins.platforms.discord.adapter import (  # noqa: E402
    _APP_COMMAND_LN_COMPATIBILITY_RANGES,
    _APP_COMMAND_LOWERCASE_COMPATIBILITY_RANGES,
    _APP_COMMAND_SCRIPT_RANGES,
    DiscordAdapter,
    _is_app_command_name_character,
    _is_valid_app_command_name,
    _normalize_app_command_mentions,
)
from plugins.platforms.discord.unicode_command_policy import (  # noqa: E402
    _APP_COMMAND_UNICODE_BASELINE_VERSION,
    _APP_COMMAND_UNICODE_SOURCE_SHA256,
    _APP_COMMAND_UNICODE_VERSION,
)


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
    # Slash auth is exercised in test_discord_slash_auth.py — bypass it here
    # so registration / dispatch / thread behavior tests don't have to
    # construct a full auth context (allowlist / channel scope).
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    return adapter


# ------------------------------------------------------------------
# /thread slash command registration
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registers_native_thread_slash_command(adapter):
    # The /thread slash closure now delegates ALL the work — including
    # defer() — to _handle_thread_create_slash so the auth gate can send
    # an ephemeral rejection on the still-unresponded interaction. The
    # closure should just forward.
    adapter._handle_thread_create_slash = AsyncMock()
    adapter._register_slash_commands()

    command = adapter._client.tree.commands["thread"]
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock()),
    )

    await command(interaction, name="Planning", message="", auto_archive_duration=1440)

    # defer is now performed inside _handle_thread_create_slash, AFTER the
    # auth check passes — not by the closure.
    interaction.response.defer.assert_not_awaited()
    adapter._handle_thread_create_slash.assert_awaited_once_with(interaction, "Planning", "", 1440)


@pytest.mark.asyncio
async def test_run_simple_slash_executes_when_defer_interaction_expired(adapter):
    class UnknownInteraction(Exception):
        status = 404
        code = 10062

    interaction = SimpleNamespace(
        channel=_FakeTextChannel(channel_id=123, name="general"),
        channel_id=123,
        guild_id=456,
        user=SimpleNamespace(id=42, name="Jezza", display_name="Jezza"),
        response=SimpleNamespace(defer=AsyncMock(side_effect=UnknownInteraction("Unknown interaction"))),
        edit_original_response=AsyncMock(),
        delete_original_response=AsyncMock(),
    )
    adapter.handle_message = AsyncMock()

    await adapter._run_simple_slash(interaction, "/reset", "Session reset~")

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "/reset"
    assert event.source.chat_id == "123"
    interaction.edit_original_response.assert_not_awaited()
    interaction.delete_original_response.assert_not_awaited()


# ------------------------------------------------------------------
# Auto-registration from COMMAND_REGISTRY
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_registers_plugin_commands_for_discord(adapter):
    """Plugin slash commands should appear as native Discord app commands."""
    adapter._run_simple_slash = AsyncMock()

    with patch(
        "hermes_cli.plugins.get_plugin_commands",
        return_value={
            "metricas": {
                "handler": lambda _a: "ok",
                "description": "Metrics dashboard",
                "args_hint": "dias:7 formato:json",
                "plugin": "metrics-plugin",
            }
        },
    ):
        adapter._register_slash_commands()

    tree_names = set(adapter._client.tree.commands.keys())
    assert "metricas" in tree_names

    metricas_cmd = adapter._client.tree.commands["metricas"]
    interaction = SimpleNamespace()
    await metricas_cmd.callback(interaction, args="dias:7 formato:json")
    adapter._run_simple_slash.assert_awaited_once_with(
        interaction, "/metricas dias:7 formato:json"
    )


@pytest.mark.asyncio
async def test_plugin_command_name_conflict_skipped(adapter):
    """A plugin command that collides with a built-in must not override it."""
    adapter._run_simple_slash = AsyncMock()

    with patch(
        "hermes_cli.plugins.get_plugin_commands",
        return_value={
            "status": {
                "handler": lambda _a: "plugin-status",
                "description": "Plugin status",
                "args_hint": "",
                "plugin": "shadow-plugin",
            }
        },
    ):
        adapter._register_slash_commands()

    # Built-ins are registered via @tree.command as plain functions. A
    # plugin-registered override would install a _FakeCommand instance
    # (has .callback) via tree.add_command. If the conflict-skip logic
    # fires, the slot remains a bare function.
    status_entry = adapter._client.tree.commands["status"]
    assert callable(status_entry) and not hasattr(status_entry, "callback"), (
        "plugin registration overrode the built-in /status command — "
        "the already_registered skip must prevent this"
    )


# ------------------------------------------------------------------
# 100-command cap (Discord error 30032 guard)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_command_registration_stays_under_discord_limit(adapter):
    """Registering far more commands than Discord allows must NOT push the
    tree over the 100-command hard cap.

    Discord rejects the ENTIRE command sync with error 30032 once the
    desired set exceeds 100 global application commands, silently breaking
    every slash command. The adapter must bound the desired set instead.
    Regression guard for samuraiheart's recurring
    "Maximum number of application commands reached (100)" sync failures.
    """
    from plugins.platforms.discord.adapter import _DISCORD_MAX_APP_COMMANDS

    adapter._run_simple_slash = AsyncMock()

    # 200 plugin commands — way past Discord's limit on their own.
    many_plugins = {
        f"plug{i:03d}": {
            "handler": lambda _a: "ok",
            "description": f"Plugin command {i}",
            "args_hint": "",
            "plugin": "stress-plugin",
        }
        for i in range(200)
    }

    with patch("hermes_cli.plugins.get_plugin_commands", return_value=many_plugins):
        adapter._register_slash_commands()

    tree_names = set(adapter._client.tree.commands.keys())

    # Contract: never exceed Discord's hard cap.
    assert len(tree_names) <= _DISCORD_MAX_APP_COMMANDS, (
        f"registered {len(tree_names)} commands — exceeds Discord's "
        f"{_DISCORD_MAX_APP_COMMANDS} limit and would fail sync with 30032"
    )

    # Native, high-priority commands are registered first and must survive
    # the cap — they are the core UX, not droppable overflow.
    for native in ("status", "stop", "new", "model", "help"):
        assert native in tree_names, f"/{native} (native) was dropped by the cap"

    # The cap must actually have dropped overflow — not every plugin fit.
    registered_plugins = [n for n in tree_names if n.startswith("plug")]
    assert len(registered_plugins) < 200, "cap did not drop any overflow commands"


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
        response=SimpleNamespace(defer=AsyncMock()),
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
        response=SimpleNamespace(defer=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "Kickoff", 1440)

    channel.send.assert_awaited_once_with("Kickoff")
    seed_message.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    interaction.followup.send.assert_awaited()


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


# ------------------------------------------------------------------
# Auto-thread: _auto_create_thread
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_create_thread_strips_mention_syntax_from_name(adapter):
    """Thread names must not contain raw <@id>, <@&id>, or <#id> markers.

    Regression guard for #6336 — previously a message like
    ``<@&1490963422786093149> help`` would spawn a thread literally
    named ``<@&1490963422786093149> help``.
    """
    thread = SimpleNamespace(id=999, name="help")
    message = SimpleNamespace(
        content="<@&1490963422786093149> <@555> please help <#123>",
        create_thread=AsyncMock(return_value=thread),
        channel=SimpleNamespace(send=AsyncMock()),
        author=SimpleNamespace(display_name="Jezza"),
    )

    await adapter._auto_create_thread(message)

    name = message.create_thread.await_args[1]["name"]
    assert "<@" not in name, f"role/user mention leaked: {name!r}"
    assert "<#" not in name, f"channel mention leaked: {name!r}"
    assert name == "please help"


@pytest.mark.asyncio
async def test_rename_thread_edits_only_when_current_name_matches(adapter):
    thread = SimpleNamespace(
        id=999,
        name="raw user prompt",
        edit=AsyncMock(),
    )
    adapter._client.get_channel = lambda _id: thread

    result = await adapter.rename_thread(
        "999",
        "Semantic Session Title",
        only_if_current_name="raw user prompt",
    )

    assert result is True
    thread.edit.assert_awaited_once_with(
        name="Semantic Session Title",
        reason="Hermes semantic session title",
    )


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

    def history(self, *args, **kwargs):
        async def _empty():
            return
            yield  # pragma: no cover — make this an async generator

        return _empty()


class _FakeThreadChannel(_discord_mod.Thread):
    """isinstance(ch, discord.Thread) → True."""

    def __init__(self, channel_id=200, name="existing-thread", guild_name="TestGuild", parent_id=100):
        # Don't call super().__init__ — mock Thread is just an empty type
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None
        self.parent = SimpleNamespace(id=parent_id, name="general", guild=SimpleNamespace(name=guild_name, id=1))

    def history(self, *args, **kwargs):
        async def _empty():
            return
            yield  # pragma: no cover — make this an async generator

        return _empty()


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


# ------------------------------------------------------------------
# Config bridge
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# /skill command registration (flat + autocomplete)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Application-command mention normalisation (clicked slash suggestions)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Bare top-level command, no surrounding text.
        ("</status:123456789>", "/status"),
        # Subcommand form: `</name sub:id>` -> `/name sub`.
        ("</cron list:987654321098765432>", "/cron list"),
        # Grouped-subcommand form: `</name group sub:id>` -> `/name group sub`.
        ("</skill manage delete:111222333444555666>", "/skill manage delete"),
        # Trailing user-typed arguments are preserved verbatim.
        ("</skill search:42> ocr-and-documents", "/skill search ocr-and-documents"),
        # Hyphens and underscores in command names are allowed by Discord.
        ("</my-cmd_x:42>", "/my-cmd_x"),
        # Discord's CHAT_INPUT grammar is Unicode-aware and permits apostrophes.
        ("</café:42>", "/café"),
        ("</नमस्ते:42>", "/नमस्ते"),
        ("</\U0001e4d0:42>", "/\U0001e4d0"),
        ("</rock'n_roll:42>", "/rock'n_roll"),
        ("</café नमस्ते:42>", "/café नमस्ते"),
        ("</café समूह नमस्ते:42>", "/café समूह नमस्ते"),
        # Every path token is independently bounded and must use lowercase
        # wherever Unicode defines a lowercase variant.
        ("</STATUS:42>", "</STATUS:42>"),
        (f"</{'a' * 33}:42>", f"</{'a' * 33}:42>"),
        ("</status Café:42>", "</status Café:42>"),
        # Punctuation outside Discord's explicit -_' set is rejected.
        ("</status!:42>", "</status!:42>"),
        # Discord separates command path tokens with literal spaces, not other
        # whitespace. Tabs/newlines must not be normalized across boundaries.
        ("</cron\tlist:42>", "</cron\tlist:42>"),
        ("</cron\nlist:42>", "</cron\nlist:42>"),
        ("</cron  list:42>", "</cron  list:42>"),
        # Multiple application-command mentions in a single message all resolve.
        ("</status:1> and </help:2>", "/status and /help"),
        # No application-command mention: pass-through, no rewrites.
        ("/status", "/status"),
        ("hello world", "hello world"),
        # Bot mention syntax is NOT a command mention and must be left alone.
        ("<@1234567890> what's up", "<@1234567890> what's up"),
    ],
)
def test_normalize_app_command_mentions(raw, expected):
    assert _normalize_app_command_mentions(raw) == expected


@pytest.mark.parametrize(
    "name",
    [
        "café",
        "नमस्ते",
        "สวัสดี",
        "rock'n_roll",
        "\u0970",
        "\u0e4f",
        "\ua8e0",
        "\U00011b00",
        "\U0001e4d0",  # Nag Mundari L/N assigned after the UCD 14 runtime floor.
        "\u1c8a",  # Lowercase counterpart assigned after the UCD 14 runtime floor.
        "".join(chr(codepoint) for codepoint in range(0xA8E0, 0xA8F2)),
        "\u0e47\u0e4f",
    ],
)
def test_valid_app_command_name_accepts_discord_unicode_grammar(name):
    assert _is_valid_app_command_name(name)


_UNICODE_ORACLE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "discord_chat_input_unicode_17.json").read_text(
        encoding="utf-8"
    )
)


def _oracle_ranges(name):
    return tuple((start, end) for start, end in _UNICODE_ORACLE[name])


def _oracle_contains(codepoint, ranges):
    low = 0
    high = len(ranges)
    while low < high:
        middle = (low + high) // 2
        start, end = ranges[middle]
        if codepoint < start:
            high = middle
        elif codepoint > end:
            low = middle + 1
        else:
            return True
    return False


def test_app_command_unicode_policy_and_oracle_share_pinned_official_sources():
    assert _APP_COMMAND_UNICODE_BASELINE_VERSION == _UNICODE_ORACLE["baseline_unicode_version"]
    assert _APP_COMMAND_UNICODE_VERSION == _UNICODE_ORACLE["unicode_version"]
    assert _APP_COMMAND_UNICODE_SOURCE_SHA256 == _UNICODE_ORACLE["source_sha256"]
    assert set(_APP_COMMAND_UNICODE_SOURCE_SHA256) == set(_UNICODE_ORACLE["source_urls"])
    assert all(
        len(digest) == 64 and not (set(digest) - set("0123456789abcdef"))
        for digest in _APP_COMMAND_UNICODE_SOURCE_SHA256.values()
    )


def test_app_command_unicode_compatibility_ranges_are_canonical():
    for ranges in (
        _APP_COMMAND_LN_COMPATIBILITY_RANGES,
        _APP_COMMAND_SCRIPT_RANGES,
        _APP_COMMAND_LOWERCASE_COMPATIBILITY_RANGES,
    ):
        previous_end = -2
        for start, end in ranges:
            assert 0 <= start <= end <= sys.maxunicode
            assert start > previous_end + 1
            previous_end = end


def test_app_command_unicode_17_full_table_oracle_has_no_missing_or_extra_codepoints():
    """Compare official expected, embedded class, and validator over all Unicode."""
    property_ranges = _oracle_ranges("property_ranges")
    lowercase_disallowed = _oracle_ranges("lowercase_disallowed_ranges")
    counts = {
        "embedded_missing": 0,
        "embedded_extra": 0,
        "accepted_missing": 0,
        "accepted_extra": 0,
    }
    samples = {name: [] for name in counts}
    for codepoint in range(sys.maxunicode + 1):
        char = chr(codepoint)
        expected_property = _oracle_contains(codepoint, property_ranges)
        expected_character = expected_property or char in "-_'"
        target_has_lowercase = _oracle_contains(codepoint, lowercase_disallowed)
        expected_valid = expected_character and not target_has_lowercase
        embedded = _is_app_command_name_character(char)
        accepted = _is_valid_app_command_name(char)

        comparisons = (
            ("embedded_missing", expected_character and not embedded),
            ("embedded_extra", embedded and not expected_character),
            ("accepted_missing", expected_valid and not accepted),
            ("accepted_extra", accepted and not expected_valid),
        )
        for name, mismatched in comparisons:
            if mismatched:
                counts[name] += 1
                if len(samples[name]) < 8:
                    samples[name].append(f"U+{codepoint:04X}")

    assert counts == {name: 0 for name in counts}, samples


def test_app_command_unicode_17_property_range_boundaries():
    property_ranges = _oracle_ranges("property_ranges")
    for start, end in property_ranges:
        assert _is_app_command_name_character(chr(start))
        assert _is_app_command_name_character(chr(end))
        if start:
            before = start - 1
            expected = _oracle_contains(before, property_ranges) or chr(before) in "-_'"
            assert _is_app_command_name_character(chr(before)) is expected
        if end < sys.maxunicode:
            after = end + 1
            expected = _oracle_contains(after, property_ranges) or chr(after) in "-_'"
            assert _is_app_command_name_character(chr(after)) is expected


@pytest.mark.parametrize(
    "name",
    [
        "STATUS",
        "a" * 33,
        "status!",
        "has space",
        "tab\tname",
        "\u1c89",  # Post-UCD-14 capital with a UCD 17 lowercase mapping.
        "\u0301",  # Combining mark outside the permitted scripts.
        "\u0e3f",  # Thai block, but Script=Common.
        "\u0964",  # Gap between exact Devanagari script ranges.
    ],
)
def test_valid_app_command_name_rejects_discord_boundaries(name):
    assert not _is_valid_app_command_name(name)


def _slash_click_message(channel, *, command_payload, bot_user=None, author_id=42, mention_bot=False):
    if mention_bot:
        assert bot_user is not None, "mention_bot=True requires bot_user"
        content = f"<@{bot_user.id}> {command_payload}"
        # discord.py mention-detection compares by identity (`user in mentions`);
        # pass through the exact bot_user object so the strip path fires.
        mentions = [bot_user]
    else:
        content = command_payload
        mentions = []
    return SimpleNamespace(
        author=SimpleNamespace(id=author_id, display_name="Jezza", bot=False, name="jezza"),
        content=content,
        channel=channel,
        attachments=[],
        message_snapshots=[],
        mentions=mentions,
        reference=None,
        created_at=None,
        id=12345,
        guild=SimpleNamespace(id=1, name="TestGuild"),
        type=_discord_mod.MessageType.default,
    )


@pytest.mark.asyncio
async def test_clicked_slash_suggestion_dispatched_as_command(adapter, monkeypatch):
    """A clicked `</status:id>` should reach handle_message as `/status` COMMAND."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    msg = _slash_click_message(_FakeTextChannel(), command_payload="</status:123456789>")
    await adapter._handle_message(msg)

    assert len(captured) == 1
    assert captured[0].text == "/status"
    assert captured[0].message_type == MessageType.COMMAND


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command_payload,expected_text,expected_type",
    [
        ("</café:101>", "/café", MessageType.COMMAND),
        ("</नमस्ते:102>", "/नमस्ते", MessageType.COMMAND),
        ("</rock'n_roll:103>", "/rock'n_roll", MessageType.COMMAND),
        ("</café समूह नमस्ते:104>", "/café समूह नमस्ते", MessageType.COMMAND),
        ("</\u0970:107>", "/\u0970", MessageType.COMMAND),
        ("</\u0e4f:108>", "/\u0e4f", MessageType.COMMAND),
        ("</\ua8e0:109>", "/\ua8e0", MessageType.COMMAND),
        ("</\U00011b00:110>", "/\U00011b00", MessageType.COMMAND),
        ("</\U0001e4d0:116>", "/\U0001e4d0", MessageType.COMMAND),
        ("</\u1c8a:117>", "/\u1c8a", MessageType.COMMAND),
        (
            f"</{''.join(chr(codepoint) for codepoint in range(0xA8E0, 0xA8F2))}:111>",
            "/" + "".join(chr(codepoint) for codepoint in range(0xA8E0, 0xA8F2)),
            MessageType.COMMAND,
        ),
        ("</\u0e47\u0e4f:112>", "/\u0e47\u0e4f", MessageType.COMMAND),
        ("</STATUS:105>", "</STATUS:105>", MessageType.TEXT),
        (f"</{'a' * 33}:106>", f"</{'a' * 33}:106>", MessageType.TEXT),
        ("</\u0301:113>", "</\u0301:113>", MessageType.TEXT),
        ("</\u0e3f:114>", "</\u0e3f:114>", MessageType.TEXT),
        ("</\u0964:115>", "</\u0964:115>", MessageType.TEXT),
        ("</\u1c89:118>", "</\u1c89:118>", MessageType.TEXT),
    ],
)
async def test_clicked_slash_suggestion_enforces_discord_name_grammar(
    adapter,
    monkeypatch,
    command_payload,
    expected_text,
    expected_type,
):
    """The real ingress path rewrites only valid Unicode CHAT_INPUT names."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    msg = _slash_click_message(_FakeTextChannel(), command_payload=command_payload)
    await adapter._handle_message(msg)

    assert len(captured) == 1
    assert captured[0].text == expected_text
    assert captured[0].message_type == expected_type


@pytest.mark.asyncio
async def test_clicked_slash_suggestion_with_args_preserves_args(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    msg = _slash_click_message(
        _FakeTextChannel(), command_payload="</skill search:42> ocr-and-documents"
    )
    await adapter._handle_message(msg)

    assert captured[0].text == "/skill search ocr-and-documents"
    assert captured[0].message_type == MessageType.COMMAND


@pytest.mark.asyncio
async def test_clicked_slash_suggestion_after_bot_mention_still_dispatches(adapter, monkeypatch):
    """`<@bot> </status:id>` (auto-prepended mention) still arrives as `/status`."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    msg = _slash_click_message(
        _FakeTextChannel(),
        command_payload="</status:123456789>",
        mention_bot=True,
        bot_user=adapter._client.user,
    )
    await adapter._handle_message(msg)

    assert captured[0].text == "/status"
    assert captured[0].message_type == MessageType.COMMAND


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command_payload,mention_bot",
    [
        ("/help", False),
        ("</help:123456789>", False),
        ("</help:123456789>", True),
    ],
)
async def test_dm_help_control_and_clicked_suggestions_dispatch_as_commands(
    adapter, monkeypatch, command_payload, mention_bot
):
    """The real DM adapter path keeps literal and clicked `/help` routing aligned."""
    class _FakeDMChannel:
        id = 200

    monkeypatch.setattr(_discord_mod, "DMChannel", _FakeDMChannel)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    msg = _slash_click_message(
        _FakeDMChannel(),
        command_payload=command_payload,
        mention_bot=mention_bot,
        bot_user=adapter._client.user if mention_bot else None,
    )
    msg.guild = None

    await adapter._handle_message(msg)

    assert len(captured) == 1
    assert captured[0].text == "/help"
    assert captured[0].message_type == MessageType.COMMAND
    assert captured[0].source.chat_type == "dm"


@pytest.mark.asyncio
async def test_plain_text_with_lt_slash_substring_not_misinterpreted(adapter, monkeypatch):
    """Literal `</` substrings in user text without an `:id>` suffix pass through."""
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    captured = []

    async def capture(event):
        captured.append(event)

    adapter.handle_message = capture
    msg = _slash_click_message(_FakeTextChannel(), command_payload="see </tag> in HTML")
    await adapter._handle_message(msg)

    assert captured[0].text == "see </tag> in HTML"
    assert captured[0].message_type == MessageType.TEXT

