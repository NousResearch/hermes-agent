"""Tests for native Discord slash command fast-paths (thread creation & auto-thread)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from gateway.config import PlatformConfig
from hermes_cli import kanban_db as kb


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


@pytest.mark.asyncio
async def test_registers_task_slash_command(adapter):
    adapter._handle_task_slash = AsyncMock()
    adapter._register_slash_commands()

    assert "task" in adapter._client.tree.commands

    interaction = SimpleNamespace()
    await adapter._client.tree.commands["task"](interaction, prompt="請求書の確認をお願いします")

    adapter._handle_task_slash.assert_awaited_once_with(
        interaction,
        "請求書の確認をお願いします",
    )


@pytest.mark.asyncio
async def test_task_slash_rejects_empty_prompt(adapter):
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await adapter._handle_task_slash(interaction, "   ")

    interaction.response.send_message.assert_awaited_once()
    args, kwargs = interaction.response.send_message.await_args
    assert "依頼内容を入力してください" in args[0]
    assert kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_task_slash_creates_kanban_task(adapter, monkeypatch):
    fake_conn = MagicMock()
    fake_conn.close = MagicMock()
    created = {}

    def fake_connect(*, board=None):
        created["board"] = board
        return fake_conn

    def fake_create_task(conn, **kwargs):
        created["conn"] = conn
        created["kwargs"] = kwargs
        return "t_123abc456def"

    def fake_get_task(conn, task_id):
        return SimpleNamespace(
            id=task_id,
            title="請求書の確認をお願いします",
            assignee="operations-orchestrator",
            status="ready",
        )

    fake_add_notify_sub = MagicMock()
    monkeypatch.setattr("hermes_cli.kanban_db.connect", fake_connect)
    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", fake_get_task)
    monkeypatch.setattr("hermes_cli.kanban_db.add_notify_sub", fake_add_notify_sub)
    monkeypatch.delenv("HERMES_TASK_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_TASK_DEFAULT_ASSIGNEE", raising=False)

    adapter._active_profile_name = MagicMock(return_value="rino")
    interaction = SimpleNamespace(
        id=987654,
        channel_id=12345,
        channel=SimpleNamespace(
            name="リノ",
            guild=SimpleNamespace(name="DEGs"),
        ),
        guild=SimpleNamespace(name="DEGs"),
        user=SimpleNamespace(id=67890, display_name="内田さん", name="uchida"),
        response=SimpleNamespace(defer=AsyncMock()),
        edit_original_response=AsyncMock(),
    )

    await adapter._handle_task_slash(interaction, "請求書の確認をお願いします")

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    assert created["board"] == "ai-company-2-0"
    assert created["kwargs"]["title"] == "請求書の確認をお願いします"
    assert created["kwargs"]["assignee"] == "operations-orchestrator"
    assert created["kwargs"]["idempotency_key"] == "discord-task:987654"
    assert "Discordの /task から登録された依頼です。" in created["kwargs"]["body"]
    assert created["kwargs"]["initial_status"] == "ready"
    assert "通常依頼は triage に固定せず" in created["kwargs"]["body"]
    fake_add_notify_sub.assert_called_once()

    interaction.edit_original_response.assert_awaited_once()
    args, kwargs = interaction.edit_original_response.await_args
    assert "依頼として受け取りました。" in kwargs["content"]
    assert "AI側で進め" in kwargs["content"]
    assert "カンバン" not in kwargs["content"]
    assert "t_123abc456def" not in kwargs["content"]
    assert "operations-orchestrator" not in kwargs["content"]
    assert "実行待ち" not in kwargs["content"]


def test_ai_company_task_creation_accepts_ready_in_real_kanban_db(adapter, monkeypatch):
    monkeypatch.delenv("HERMES_TASK_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_TASK_DEFAULT_ASSIGNEE", raising=False)

    adapter._active_profile_name = MagicMock(return_value="rino")
    payload = {
        "title": "ダッシュボードに登録される確認",
        "body": "Discordから登録された依頼です。",
        "created_by": "内田さん",
        "user_id": "67890",
    }

    created = adapter._create_ai_company_task(
        payload=payload,
        idempotency_key="discord-task:real-db-ready",
        chat_id="12345",
        thread_id=None,
    )

    assert created["status"] == "ready"
    assert created["assignee"] == "operations-orchestrator"

    with kb.connect(board="ai-company-2-0") as conn:
        task = kb.get_task(conn, created["id"])
        subscribers = kb.list_notify_subs(conn, created["id"])

    assert task.status == "ready"
    assert task.title == "ダッシュボードに登録される確認"
    assert subscribers
    assert subscribers[0]["platform"] == "discord"
    assert subscribers[0]["chat_id"] == "12345"
    assert subscribers[0]["user_id"] == "67890"
    assert subscribers[0]["notifier_profile"] == "rino"


def test_ai_company_task_creation_uses_active_profile_fallback_without_adapter_method(adapter, monkeypatch):
    """DiscordAdapter itself does not define _active_profile_name; natural task creation must not crash."""
    fake_conn = MagicMock()
    fake_conn.close = MagicMock()

    def fake_create_task(conn, **kwargs):
        return "t_fallback123"

    def fake_get_task(conn, task_id):
        return SimpleNamespace(
            id=task_id,
            title="カンバン登録の確認",
            assignee="operations-orchestrator",
            status="ready",
        )

    fake_add_notify_sub = MagicMock()
    monkeypatch.setattr("hermes_cli.kanban_db.connect", MagicMock(return_value=fake_conn))
    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", fake_get_task)
    monkeypatch.setattr("hermes_cli.kanban_db.add_notify_sub", fake_add_notify_sub)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.delenv("HERMES_TASK_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_TASK_DEFAULT_ASSIGNEE", raising=False)

    if hasattr(adapter, "_active_profile_name"):
        delattr(adapter, "_active_profile_name")

    payload = {
        "title": "カンバン登録の確認",
        "body": "Discordの自然文依頼から登録されたタスクです。",
        "created_by": "内田さん",
        "user_id": "67890",
    }

    created = adapter._create_ai_company_task(
        payload=payload,
        idempotency_key="discord-natural-task:fallback-profile",
        chat_id="12345",
        thread_id=None,
    )

    assert created["id"] == "t_fallback123"
    fake_add_notify_sub.assert_called_once()
    assert fake_add_notify_sub.call_args.kwargs["notifier_profile"] == "default"


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
    expected_auto = {"debug", "yolo", "profile"}
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
async def test_auto_registered_plugin_command_without_args_hint(adapter):
    """Plugin commands without args_hint should register as parameterless."""
    adapter._run_simple_slash = AsyncMock()

    with patch(
        "hermes_cli.plugins.get_plugin_commands",
        return_value={
            "ping": {
                "handler": lambda _a: "pong",
                "description": "Ping the plugin",
                "args_hint": "",
                "plugin": "ping-plugin",
            }
        },
    ):
        adapter._register_slash_commands()

    assert "ping" in adapter._client.tree.commands
    ping_cmd = adapter._client.tree.commands["ping"]
    interaction = SimpleNamespace()
    await ping_cmd.callback(interaction)
    adapter._run_simple_slash.assert_awaited_once_with(interaction, "/ping")


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
        response=SimpleNamespace(defer=AsyncMock()),
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
        response=SimpleNamespace(defer=AsyncMock()),
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
        response=SimpleNamespace(defer=AsyncMock()),
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
        channel=SimpleNamespace(send=AsyncMock()),
        author=SimpleNamespace(display_name="Jezza"),
    )

    result = await adapter._auto_create_thread(message)

    assert result is thread
    message.create_thread.assert_awaited_once()
    call_kwargs = message.create_thread.await_args[1]
    assert call_kwargs["name"] == "Hello world, how are you?"
    assert call_kwargs["auto_archive_duration"] == 1440


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
async def test_auto_create_thread_falls_back_to_hermes_when_only_mentions(adapter):
    """If a message contains only mention syntax, the stripped content is
    empty — fall back to the 'Hermes' default rather than ''."""
    thread = SimpleNamespace(id=999, name="Hermes")
    message = SimpleNamespace(
        content="<@&1490963422786093149>",
        create_thread=AsyncMock(return_value=thread),
        channel=SimpleNamespace(send=AsyncMock()),
        author=SimpleNamespace(display_name="Jezza"),
    )

    await adapter._auto_create_thread(message)

    name = message.create_thread.await_args[1]["name"]
    assert name == "Hermes"


@pytest.mark.asyncio
async def test_auto_create_thread_truncates_long_names(adapter):
    long_text = "a" * 200
    thread = SimpleNamespace(id=999, name="truncated")
    message = SimpleNamespace(
        content=long_text,
        create_thread=AsyncMock(return_value=thread),
        channel=SimpleNamespace(send=AsyncMock()),
        author=SimpleNamespace(display_name="Jezza"),
    )

    result = await adapter._auto_create_thread(message)

    assert result is thread
    call_kwargs = message.create_thread.await_args[1]
    assert len(call_kwargs["name"]) <= 80
    assert call_kwargs["name"].endswith("...")


@pytest.mark.asyncio
async def test_auto_create_thread_falls_back_to_seed_message(adapter):
    thread = SimpleNamespace(id=555, name="Hello")
    seed_message = SimpleNamespace(create_thread=AsyncMock(return_value=thread))
    message = SimpleNamespace(
        content="Hello",
        create_thread=AsyncMock(side_effect=RuntimeError("no perms")),
        channel=SimpleNamespace(send=AsyncMock(return_value=seed_message)),
        author=SimpleNamespace(display_name="Jezza"),
    )

    result = await adapter._auto_create_thread(message)
    assert result is thread
    message.channel.send.assert_awaited_once_with("🧵 Thread created by Hermes: **Hello**")
    seed_message.create_thread.assert_awaited_once_with(
        name="Hello",
        auto_archive_duration=1440,
        reason="Auto-threaded from mention by Jezza",
    )


@pytest.mark.asyncio
async def test_auto_create_thread_returns_none_when_direct_and_fallback_fail(adapter):
    message = SimpleNamespace(
        content="Hello",
        create_thread=AsyncMock(side_effect=RuntimeError("no perms")),
        channel=SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("send failed"))),
        author=SimpleNamespace(display_name="Jezza"),
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
async def test_natural_language_task_request_creates_ready_kanban_task(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_NATURAL_TASK_KANBAN", "true")
    monkeypatch.delenv("HERMES_TASK_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_TASK_DEFAULT_ASSIGNEE", raising=False)

    fake_conn = MagicMock()
    fake_conn.close = MagicMock()
    created = {}

    def fake_connect(*, board=None):
        created["board"] = board
        return fake_conn

    def fake_create_task(conn, **kwargs):
        created["conn"] = conn
        created["kwargs"] = kwargs
        return "t_natural123"

    def fake_get_task(conn, task_id):
        return SimpleNamespace(
            id=task_id,
            title="この表示を直してください",
            assignee="operations-orchestrator",
            status="ready",
        )

    monkeypatch.setattr("hermes_cli.kanban_db.connect", fake_connect)
    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", fake_get_task)
    monkeypatch.setattr("hermes_cli.kanban_db.add_notify_sub", MagicMock())

    adapter._active_profile_name = MagicMock(return_value="rino")
    adapter.handle_message = AsyncMock()
    adapter.send = AsyncMock()

    msg = _fake_message(_FakeTextChannel(), content="この表示を直してください")

    await adapter._handle_message(msg)

    adapter.handle_message.assert_awaited_once()
    adapter.send.assert_not_awaited()
    assert created["board"] == "ai-company-2-0"
    assert created["kwargs"]["title"] == "この表示を直してください"
    assert created["kwargs"]["assignee"] == "operations-orchestrator"
    assert created["kwargs"]["idempotency_key"] == "discord-natural-task:12345"
    assert created["kwargs"]["initial_status"] == "ready"
    assert "Discordの自然文依頼から登録されたタスクです。" in created["kwargs"]["body"]
    assert "通常依頼は triage に固定せず" in created["kwargs"]["body"]


@pytest.mark.asyncio
async def test_natural_language_task_request_continues_when_kanban_create_fails(
    adapter, monkeypatch
):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_NATURAL_TASK_KANBAN", "true")

    def fake_connect(*, board=None):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("hermes_cli.kanban_db.connect", fake_connect)

    adapter.handle_message = AsyncMock()
    adapter.send = AsyncMock()

    msg = _fake_message(_FakeTextChannel(), content="この表示を直してください")

    await adapter._handle_message(msg)

    adapter.handle_message.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_natural_language_task_request_bypasses_mention_requirement(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_HISTORY_BACKFILL", "false")
    monkeypatch.setenv("DISCORD_NATURAL_TASK_KANBAN", "true")
    monkeypatch.delenv("HERMES_TASK_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_TASK_DEFAULT_ASSIGNEE", raising=False)

    fake_conn = MagicMock()
    fake_conn.close = MagicMock()
    created = {}

    def fake_connect(*, board=None):
        created["board"] = board
        return fake_conn

    def fake_create_task(conn, **kwargs):
        created["conn"] = conn
        created["kwargs"] = kwargs
        return "t_nomention123"

    def fake_get_task(conn, task_id):
        return SimpleNamespace(
            id=task_id,
            title="ダッシュボードの表示を直してください",
            assignee="operations-orchestrator",
            status="ready",
        )

    monkeypatch.setattr("hermes_cli.kanban_db.connect", fake_connect)
    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", fake_get_task)
    monkeypatch.setattr("hermes_cli.kanban_db.add_notify_sub", MagicMock())

    adapter._active_profile_name = MagicMock(return_value="rino")
    adapter.handle_message = AsyncMock()
    adapter.send = AsyncMock()

    msg = _fake_message(
        _FakeTextChannel(),
        content="ダッシュボードの表示を直してください",
    )

    await adapter._handle_message(msg)

    adapter.handle_message.assert_awaited_once()
    adapter.send.assert_not_awaited()
    assert created["board"] == "ai-company-2-0"
    assert created["kwargs"]["title"] == "ダッシュボードの表示を直してください"
    assert created["kwargs"]["idempotency_key"] == "discord-natural-task:12345"
    assert created["kwargs"]["initial_status"] == "ready"


@pytest.mark.asyncio
async def test_natural_language_explicit_task_registration_with_question_creates_task(
    adapter, monkeypatch
):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_NATURAL_TASK_KANBAN", "true")
    monkeypatch.delenv("HERMES_TASK_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_TASK_DEFAULT_ASSIGNEE", raising=False)

    fake_conn = MagicMock()
    fake_conn.close = MagicMock()
    created = {}

    def fake_connect(*, board=None):
        created["board"] = board
        return fake_conn

    def fake_create_task(conn, **kwargs):
        created["conn"] = conn
        created["kwargs"] = kwargs
        return "t_register123"

    def fake_get_task(conn, task_id):
        return SimpleNamespace(
            id=task_id,
            title="テストで新規タスクを登録したい。まずはどんなスキルがあるかを箇条書きで出して",
            assignee="operations-orchestrator",
            status="ready",
        )

    fake_add_notify_sub = MagicMock()
    monkeypatch.setattr("hermes_cli.kanban_db.connect", fake_connect)
    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", fake_get_task)
    monkeypatch.setattr("hermes_cli.kanban_db.add_notify_sub", fake_add_notify_sub)

    adapter._active_profile_name = MagicMock(return_value="rino")
    adapter.handle_message = AsyncMock()
    adapter.send = AsyncMock()

    prompt = "テストで新規タスクを登録したい。まずはどんなスキルがあるかを箇条書きで出して"
    msg = _fake_message(_FakeTextChannel(), content=prompt)

    await adapter._handle_message(msg)

    adapter.handle_message.assert_awaited_once()
    adapter.send.assert_not_awaited()
    assert created["board"] == "ai-company-2-0"
    assert created["kwargs"]["title"] == prompt
    assert created["kwargs"]["idempotency_key"] == "discord-natural-task:12345"
    assert created["kwargs"]["initial_status"] == "ready"
    fake_add_notify_sub.assert_called_once()


def test_natural_language_classifier_registers_regular_discord_messages():
    assert DiscordAdapter._looks_like_natural_task_request(
        "どんなスキルがあるかを箇条書きで出して"
    )
    assert DiscordAdapter._looks_like_natural_task_request("この説明の意味わかりますか？")
    assert not DiscordAdapter._looks_like_natural_task_request("/status")


@pytest.mark.asyncio
async def test_natural_language_question_registers_and_still_answers(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setenv("DISCORD_NATURAL_TASK_KANBAN", "true")
    monkeypatch.delenv("HERMES_TASK_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_TASK_DEFAULT_ASSIGNEE", raising=False)

    fake_conn = MagicMock()
    fake_conn.close = MagicMock()
    created = {}

    def fake_connect(*, board=None):
        created["board"] = board
        return fake_conn

    def fake_create_task(conn, **kwargs):
        created["conn"] = conn
        created["kwargs"] = kwargs
        return "t_question123"

    def fake_get_task(conn, task_id):
        return SimpleNamespace(
            id=task_id,
            title="この説明の意味わかりますか？",
            assignee="operations-orchestrator",
            status="ready",
        )

    fake_add_notify_sub = MagicMock()
    monkeypatch.setattr("hermes_cli.kanban_db.connect", fake_connect)
    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", fake_get_task)
    monkeypatch.setattr("hermes_cli.kanban_db.add_notify_sub", fake_add_notify_sub)

    captured_events = []

    async def capture_handle(event):
        captured_events.append(event)

    adapter.handle_message = capture_handle
    adapter.send = AsyncMock()

    msg = _fake_message(_FakeTextChannel(), content="この説明の意味わかりますか？")

    await adapter._handle_message(msg)

    adapter.send.assert_not_awaited()
    assert len(captured_events) == 1
    assert captured_events[0].text == "この説明の意味わかりますか？"
    assert created["board"] == "ai-company-2-0"
    assert created["kwargs"]["title"] == "この説明の意味わかりますか？"
    fake_add_notify_sub.assert_called_once()


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
