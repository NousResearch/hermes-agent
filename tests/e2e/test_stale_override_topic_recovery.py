"""Real-ingress regression for stale notices in Telegram DM topic mode."""

import asyncio
import dataclasses
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import (
    AsyncSessionStore,
    SessionSource,
    SessionStore,
    build_session_key,
)
from gateway.stale_override_notice import (
    OverrideNoticeDecision,
    StaleOverrideNoticeConfig,
)
from plugins.platforms.telegram.adapter import TelegramAdapter


def _write_profile_skill(hermes_home, name, body):
    skill_dir = hermes_home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: Test skill {name}",
                "---",
                f"# {name}",
                "",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


def _runner_with_store(config, store, db):
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._running_agents_ts = {}
    runner._pending_model_notes = {}
    runner._update_prompt_pending = {}
    runner._session_db = SimpleNamespace(_db=db)
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._stale_override_pending = {}
    runner._is_user_authorized = lambda _source: True
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def _busy_topic_command_runner(
    tmp_path,
    monkeypatch,
    *,
    quick_commands=None,
    busy_mode="steer",
):
    """Build a stripped Telegram source whose canonical topic lane is busy."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import gateway.run as gateway_run_module

    monkeypatch.setattr(gateway_run_module, "_hermes_home", hermes_home)
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        quick_commands=quick_commands or {},
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)
    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        thread_id="42",
    )
    canonical_entry = store.get_or_create_session(canonical_source)
    canonical_key = canonical_entry.session_key
    raw_source = dataclasses.replace(canonical_source, thread_id=None)
    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=canonical_entry.session_id,
    )

    class ActiveAgent:
        def __init__(self):
            self.steered = []

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

        def steer(self, text):
            self.steered.append(text)
            return True

    runner = _runner_with_store(config, store, db)
    runner._busy_input_mode = busy_mode
    active_agent = ActiveAgent()
    active_state = runner._session_state(canonical_key)
    active_state.turn.agent = active_agent
    active_state.turn.started_ts = time.time() - 10
    adapter = SimpleNamespace(_pending_messages={}, send=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("recovered busy command dispatched a second turn")
    )
    return runner, raw_source, canonical_key, active_agent, adapter


@pytest.mark.asyncio
async def test_topic_recovered_ingress_uses_canonical_clock_entry(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        stale_override_notice=StaleOverrideNoticeConfig(
            mode="info_only",
            idle_minutes=1,
            channels=("*",),
        ),
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)

    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        user_name="topic user",
        thread_id="42",
    )
    entry = store.get_or_create_session(canonical_source)
    canonical_key = entry.session_key
    raw_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        user_name="topic user",
        thread_id=None,
    )
    raw_key = build_session_key(raw_source)
    assert raw_key != canonical_key
    store.set_session_metadata(
        canonical_key,
        "stale_override_last_completed_at",
        time.time() - 3600,
    )

    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=entry.session_id,
    )

    runner = _runner_with_store(config, store, db)
    runner._stale_override_decision = MagicMock(
        return_value=OverrideNoticeDecision(
            model_stale=True,
            current_route="provider/custom",
            default_route="provider/default",
        )
    )

    seen = {}

    async def _agent(event, source, session_key, generation):
        seen.update(
            source=source,
            session_key=session_key,
            generation=generation,
        )
        return "agent response"

    runner._handle_message_with_agent = _agent
    adapter = TelegramAdapter(config.platforms[Platform.TELEGRAM])
    adapter.send = AsyncMock(
        return_value=SendResult(success=True, message_id="e2e-response")
    )
    adapter.send_typing = AsyncMock()
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.TELEGRAM] = adapter

    event = MessageEvent(
        text="ordinary message",
        message_id="inbound-message",
        source=raw_source,
    )
    result = await runner._handle_message(event)
    assert result == "agent response"

    completion_callback = adapter.pop_post_delivery_callback(
        canonical_key,
        generation=seen["generation"],
    )
    assert completion_callback is not None
    await completion_callback()

    assert event.source.thread_id == "42"
    assert seen["source"].thread_id == "42"
    assert seen["session_key"] == canonical_key
    runner._stale_override_decision.assert_called_once()
    assert (
        runner._stale_override_decision.call_args.kwargs["session_key"] == canonical_key
    )
    assert (
        store.get_session_metadata(raw_key, "stale_override_last_completed_at") is None
    )

    # Completion metadata uses the single-entry DB UPSERT and survives a fresh
    # SessionStore load without requiring a full sessions.json rewrite.
    restarted = SessionStore(config.sessions_dir, config)
    completed_at = restarted.get_session_metadata(
        canonical_key, "stale_override_last_completed_at"
    )
    assert completed_at > time.time() - 10


@pytest.mark.asyncio
async def test_topic_recovered_busy_ingress_preserves_original_active_agent(
    tmp_path, monkeypatch
):
    """A stripped topic follow-up must enter the canonical busy path."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "0")
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)
    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        user_name="topic user",
        thread_id="42",
    )
    canonical_entry = store.get_or_create_session(canonical_source)
    canonical_key = canonical_entry.session_key
    raw_source = dataclasses.replace(canonical_source, thread_id=None)

    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=canonical_entry.session_id,
    )

    class ActiveAgent:
        def __init__(self):
            self.steered = []

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

        def steer(self, text):
            self.steered.append(text)
            return True

    runner = _runner_with_store(config, store, db)
    runner._busy_input_mode = "steer"
    active_agent = ActiveAgent()
    active_state = runner._session_state(canonical_key)
    active_state.turn.agent = active_agent
    active_state.turn.started_ts = time.time() - 10
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("canonical busy follow-up dispatched a second turn")
    )
    original_claim = runner._claim_active_session_slot
    runner._claim_active_session_slot = MagicMock(wraps=original_claim)

    event = MessageEvent(
        text="steer the active topic turn",
        message_id="busy-follow-up",
        source=raw_source,
    )
    result = await runner._handle_message(event)

    assert result is None
    assert event.source.thread_id == "42"
    assert active_agent.steered == ["steer the active topic turn"]
    runner._claim_active_session_slot.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    assert runner._peek_session_state(canonical_key).turn.agent is active_agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_text", "expected_steer", "expected_queue", "expected_reply"),
    [
        (
            "/steer follow up",
            ["follow up"],
            [],
            "⏩ Steer queued — arrives after the next tool call: 'follow up'",
        ),
        ("/queue follow up", [], ["follow up"], "Queued for the next turn."),
        (
            "/moa compare these",
            [],
            [],
            "Agent is running — wait or /stop first, then run /moa.",
        ),
    ],
)
async def test_stripped_topic_fallthrough_command_reenters_canonical_busy_path(
    tmp_path,
    monkeypatch,
    command_text,
    expected_steer,
    expected_queue,
    expected_reply,
):
    """Late topic recovery must route commands before claim/state mutation."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)
    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        user_name="topic user",
        thread_id="42",
    )
    canonical_entry = store.get_or_create_session(canonical_source)
    canonical_key = canonical_entry.session_key
    raw_source = dataclasses.replace(canonical_source, thread_id=None)
    raw_key = build_session_key(raw_source)

    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=canonical_entry.session_id,
    )

    class ActiveAgent:
        def __init__(self):
            self.steered = []

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

        def steer(self, text):
            self.steered.append(text)
            return True

    runner = _runner_with_store(config, store, db)
    active_agent = ActiveAgent()
    active_state = runner._session_state(canonical_key)
    original_override = {"provider": "test", "model": "original"}
    active_state.conversation.model_override = original_override
    active_state.turn.agent = active_agent
    active_state.turn.started_ts = time.time() - 10
    adapter = SimpleNamespace(_pending_messages={})
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("late canonical command dispatched a second turn")
    )
    original_claim = runner._claim_active_session_slot
    runner._claim_active_session_slot = MagicMock(wraps=original_claim)

    event = MessageEvent(
        text=command_text,
        message_id="busy-command",
        source=raw_source,
    )
    result = await runner._handle_message(event)

    assert result == expected_reply
    assert event.source.thread_id == "42"
    assert active_agent.steered == expected_steer
    queued = adapter._pending_messages.get(canonical_key)
    assert ([queued.text] if queued is not None else []) == expected_queue
    runner._claim_active_session_slot.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    preserved = runner._peek_session_state(canonical_key)
    assert preserved.turn.agent is active_agent
    assert preserved.conversation.model_override is original_override
    assert runner._peek_session_state(raw_key) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_text", "blocked_call"),
    [
        ("/learn https://example.com/source", "learn"),
        ("/init include the deployment notes", "init"),
        ("/blueprint daily-digest", "blueprint"),
        (
            "/blueprint daily-digest topic=AI time=08:00 recurrence=weekdays",
            "blueprint",
        ),
    ],
)
async def test_stripped_topic_side_effectful_command_is_rejected_before_handler_or_ack(
    tmp_path, monkeypatch, command_text, blocked_call
):
    """Canonical busy policy must precede command acks and mutations."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)
    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        thread_id="42",
    )
    canonical_entry = store.get_or_create_session(canonical_source)
    canonical_key = canonical_entry.session_key
    raw_source = dataclasses.replace(canonical_source, thread_id=None)
    raw_key = build_session_key(raw_source)
    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=canonical_entry.session_id,
    )

    class ActiveAgent:
        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    runner = _runner_with_store(config, store, db)
    active_agent = ActiveAgent()
    active_state = runner._session_state(canonical_key)
    original_override = {"provider": "test", "model": "original"}
    active_state.conversation.model_override = original_override
    active_state.turn.agent = active_agent
    active_state.turn.started_ts = time.time() - 10

    adapter = SimpleNamespace(send=AsyncMock(), _pending_messages={})
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("busy command dispatched a second turn")
    )
    runner._claim_active_session_slot = MagicMock(
        side_effect=AssertionError("busy command reached session claim")
    )
    runner._handle_blueprint_command = AsyncMock(
        side_effect=AssertionError("busy /blueprint reached its handler")
    )
    learn_builder = MagicMock(
        side_effect=AssertionError("busy /learn built a persistent skill prompt")
    )
    init_builder = MagicMock(
        side_effect=AssertionError("busy /init scanned or built a project prompt")
    )
    cron_registration = MagicMock(
        side_effect=AssertionError("busy /blueprint registered a cron job")
    )
    monkeypatch.setattr("agent.learn_prompt.build_learn_prompt", learn_builder)
    monkeypatch.setattr(
        "hermes_cli.init_command.build_init_prompt_for_cwd", init_builder
    )
    monkeypatch.setattr(
        "cron.scheduler.create_job_with_scheduler_registration", cron_registration
    )

    event = MessageEvent(text=command_text, source=raw_source)
    result = await runner._handle_message(event)
    canonical_command = command_text.split(maxsplit=1)[0].lstrip("/")

    assert result == (
        f"⏳ Agent is running — `/{canonical_command}` can't run mid-turn. "
        "Wait for the current response or `/stop` first."
    )
    assert event.source.thread_id == "42"
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()
    runner._claim_active_session_slot.assert_not_called()
    cron_registration.assert_not_called()
    if blocked_call == "learn":
        learn_builder.assert_not_called()
    elif blocked_call == "init":
        init_builder.assert_not_called()
    else:
        runner._handle_blueprint_command.assert_not_awaited()
    preserved = runner._peek_session_state(canonical_key)
    assert preserved.turn.agent is active_agent
    assert preserved.conversation.model_override is original_override
    assert runner._peek_session_state(raw_key) is None
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_result",
    [
        {"decision": "handled", "message": "optimistic hook acknowledgement"},
        {"decision": "deny", "message": "hook denial"},
    ],
)
async def test_recovered_busy_registered_command_skips_all_command_hooks(
    tmp_path, monkeypatch, hook_result
):
    """Hooks cannot observe or intercept a command before canonical busy policy."""
    runner, raw_source, _canonical_key, active_agent, adapter = (
        _busy_topic_command_runner(tmp_path, monkeypatch)
    )
    pre_command = MagicMock(
        side_effect=AssertionError("busy command fired pre_command hook")
    )
    monkeypatch.setattr("hermes_cli.plugins.fire_pre_command_hook", pre_command)
    runner.hooks.emit_collect = AsyncMock(return_value=[hook_result])

    result = await runner._handle_message(
        MessageEvent(text="/learn https://example.com/source", source=raw_source)
    )

    assert result == (
        "⏳ Agent is running — `/learn` can't run mid-turn. "
        "Wait for the current response or `/stop` first."
    )
    pre_command.assert_not_called()
    runner.hooks.emit_collect.assert_not_awaited()
    assert active_agent.steered == []
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_busy_quick_exec_uses_canonical_busy_route_without_shell(
    tmp_path, monkeypatch
):
    """A stripped quick exec is input to busy policy, never a shell invocation."""
    runner, raw_source, _canonical_key, active_agent, adapter = (
        _busy_topic_command_runner(
            tmp_path,
            monkeypatch,
            quick_commands={
                "danger": {"type": "exec", "command": "printf side-effect"}
            },
        )
    )
    create_shell = AsyncMock(
        side_effect=AssertionError("busy quick command executed a shell")
    )
    monkeypatch.setattr(asyncio, "create_subprocess_shell", create_shell)

    result = await runner._handle_message(
        MessageEvent(text="/danger now", source=raw_source)
    )

    assert result is None
    create_shell.assert_not_awaited()
    assert active_agent.steered == ["/danger now"]
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["/stop", "/new"])
async def test_recovered_busy_quick_alias_preserves_typed_identity(
    tmp_path, monkeypatch, target
):
    """A quick alias is generic busy input, never its control-command target."""
    runner, raw_source, canonical_key, active_agent, adapter = (
        _busy_topic_command_runner(
            tmp_path,
            monkeypatch,
            quick_commands={"shortcut": {"type": "alias", "target": target}},
        )
    )
    runner._handle_reset_command = AsyncMock(
        side_effect=AssertionError("busy alias dispatched /new")
    )
    runner._interrupt_and_clear_session = AsyncMock(
        side_effect=AssertionError("busy alias dispatched a control target")
    )

    result = await runner._handle_message(
        MessageEvent(text="/shortcut keep this", source=raw_source)
    )

    assert result is None
    assert active_agent.steered == ["/shortcut keep this"]
    assert runner._peek_session_state(canonical_key).turn.agent is active_agent
    runner._handle_reset_command.assert_not_awaited()
    runner._interrupt_and_clear_session.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_text", "quick_commands"),
    [
        ("/quick now", {"quick": {"type": "exec", "command": "false"}}),
        ("/plugin mutate", None),
        ("/unknown value", None),
    ],
)
async def test_recovered_busy_non_builtin_never_touches_plugin_discovery(
    tmp_path, monkeypatch, command_text, quick_commands
):
    runner, raw_source, _canonical_key, active_agent, _adapter = (
        _busy_topic_command_runner(
            tmp_path,
            monkeypatch,
            quick_commands=quick_commands,
        )
    )

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("plugin discovery touched before busy decision")

    monkeypatch.setattr("hermes_cli.commands.is_gateway_known_command", _forbidden)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_commands", _forbidden)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_command_handler", _forbidden)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _forbidden)
    monkeypatch.setattr("hermes_cli.plugins.PluginManager.discover_and_load", _forbidden)

    result = await runner._handle_message(
        MessageEvent(text=command_text, source=raw_source)
    )

    assert result is None
    assert active_agent.steered == [command_text]


@pytest.mark.asyncio
async def test_recovered_busy_plugin_command_never_calls_handler_or_returns_ack(
    tmp_path, monkeypatch
):
    """Plugin discovery and the handler both stay inert before busy routing."""
    runner, raw_source, _canonical_key, active_agent, adapter = (
        _busy_topic_command_runner(tmp_path, monkeypatch)
    )
    plugin_handler = MagicMock(return_value="optimistic plugin acknowledgement")
    plugin_lookup = MagicMock(
        side_effect=AssertionError("busy plugin command triggered discovery")
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler", plugin_lookup
    )

    result = await runner._handle_message(
        MessageEvent(text="/side_effect mutate", source=raw_source)
    )

    assert result is None
    plugin_lookup.assert_not_called()
    plugin_handler.assert_not_called()
    assert active_agent.steered == ["/side_effect mutate"]
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_mode", ["queue", "steer"])
async def test_stripped_topic_dynamic_skill_preserves_busy_mode(
    tmp_path, monkeypatch, busy_mode
):
    """No-CommandDef skill turns must queue/steer under the canonical key."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import gateway.run as gateway_run_module

    monkeypatch.setattr(gateway_run_module, "_hermes_home", hermes_home)
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        },
        sessions_dir=hermes_home / "sessions",
    )
    store = SessionStore(config.sessions_dir, config)
    canonical_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="topic-chat",
        chat_type="dm",
        user_id="topic-user",
        thread_id="42",
    )
    canonical_entry = store.get_or_create_session(canonical_source)
    canonical_key = canonical_entry.session_key
    raw_source = dataclasses.replace(canonical_source, thread_id=None)
    raw_key = build_session_key(raw_source)
    db = store._db
    db.enable_telegram_topic_mode(chat_id="topic-chat", user_id="topic-user")
    db.bind_telegram_topic(
        chat_id="topic-chat",
        thread_id="42",
        user_id="topic-user",
        session_key=canonical_key,
        session_id=canonical_entry.session_id,
    )

    monkeypatch.setattr(
        "agent.skill_commands._skill_commands",
        {"/race-skill": {"name": "race-skill"}},
    )
    import agent.skill_commands as skill_commands_module
    from hermes_constants import hermes_home_key

    monkeypatch.setattr(
        skill_commands_module,
        "_skill_command_snapshots",
        {
            (hermes_home_key(hermes_home), "telegram"): {
                "/race-skill": {"name": "race-skill"},
            }
        },
        raising=False,
    )

    monkeypatch.setattr(
        skill_commands_module,
        "_skill_commands_home",
        skill_commands_module._resolve_skill_commands_home(),
    )
    monkeypatch.setattr(
        skill_commands_module,
        "_skill_commands_platform",
        skill_commands_module._resolve_skill_commands_platform(),
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/race-skill": {"name": "race-skill"}},
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        lambda *_args, **_kwargs: "loaded dynamic skill prompt",
    )
    monkeypatch.setattr(
        "agent.skill_utils.get_disabled_skill_names", lambda **_kwargs: set()
    )
    forbidden_discovery = MagicMock(
        side_effect=AssertionError("dynamic busy routing touched plugin discovery")
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler", forbidden_discovery
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_commands", forbidden_discovery)
    monkeypatch.setattr(
        "hermes_cli.plugins.PluginManager.discover_and_load", forbidden_discovery
    )
    monkeypatch.setattr(
        "hermes_cli.commands.is_gateway_known_command", forbidden_discovery
    )
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", forbidden_discovery)

    class ActiveAgent:
        def __init__(self):
            self.steered = []

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

        def steer(self, text):
            self.steered.append(text)
            return True

    runner = _runner_with_store(config, store, db)
    runner._busy_input_mode = busy_mode
    active_agent = ActiveAgent()
    active_state = runner._session_state(canonical_key)
    active_state.turn.agent = active_agent
    active_state.turn.started_ts = time.time() - 10
    adapter = TelegramAdapter(config.platforms[Platform.TELEGRAM])
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 0
    adapter._busy_text_hard_cap_seconds = 0
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("dynamic skill dispatched a second turn")
    )
    runner._claim_active_session_slot = MagicMock(
        side_effect=AssertionError("dynamic skill reached session claim")
    )

    result = await runner._handle_message(
        MessageEvent(text="/race-skill inspect", source=raw_source)
    )
    await asyncio.sleep(0)

    assert result is None
    if busy_mode == "queue":
        await adapter._flush_text_debounce_now(canonical_key)
        assert adapter._pending_messages[canonical_key].text == (
            "loaded dynamic skill prompt"
        )
        assert active_agent.steered == []
    else:
        assert canonical_key not in adapter._pending_messages
        assert active_agent.steered == ["loaded dynamic skill prompt"]
    assert raw_key not in adapter._pending_messages
    runner._handle_message_with_agent.assert_not_awaited()
    assert runner._peek_session_state(canonical_key).turn.agent is active_agent
    forbidden_discovery.assert_not_called()


@pytest.mark.asyncio
async def test_stripped_topic_ignores_dynamic_skill_cache_from_other_profile(
    tmp_path, monkeypatch
):
    runner, raw_source, canonical_key, active_agent, adapter = (
        _busy_topic_command_runner(tmp_path, monkeypatch, busy_mode="steer")
    )
    import agent.skill_commands as skill_commands_module

    monkeypatch.setattr(
        skill_commands_module,
        "_skill_commands",
        {"/only-a": {"name": "only-a"}},
    )
    from hermes_constants import hermes_home_key

    monkeypatch.setattr(
        skill_commands_module,
        "_skill_command_snapshots",
        {
            (hermes_home_key("/profiles/profile-a"), "telegram"): {
                "/only-a": {"name": "only-a"},
            }
        },
        raising=False,
    )
    monkeypatch.setattr(
        skill_commands_module, "_skill_commands_home", "/profiles/profile-a"
    )
    monkeypatch.setattr(
        skill_commands_module,
        "_skill_commands_platform",
        skill_commands_module._resolve_skill_commands_platform(),
    )
    build_skill = MagicMock(
        side_effect=AssertionError("stale profile cache expanded a skill")
    )
    monkeypatch.setattr(
        skill_commands_module, "build_skill_invocation_message", build_skill
    )

    result = await runner._handle_message(
        MessageEvent(text="/only-a inspect", source=raw_source)
    )

    assert result is None
    assert active_agent.steered == ["/only-a inspect"]
    preserved = runner._peek_session_state(canonical_key)
    assert preserved is not None
    assert preserved.turn.agent is active_agent
    build_skill.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_dynamic_skill_cache_uses_message_platform_in_same_process(
    tmp_path, monkeypatch
):
    """A recovered Telegram turn must ignore a Discord snapshot and use Telegram's."""
    runner, raw_source, canonical_key, active_agent, adapter = (
        _busy_topic_command_runner(tmp_path, monkeypatch, busy_mode="steer")
    )
    import agent.skill_bundles as skill_bundles_module
    import agent.skill_commands as skill_commands_module
    from hermes_constants import hermes_home_key

    hermes_home = tmp_path / ".hermes"
    home_key = hermes_home_key(hermes_home)
    telegram_skill_dir = _write_profile_skill(
        hermes_home,
        "only-telegram",
        "TELEGRAM SNAPSHOT SKILL BODY",
    )
    monkeypatch.setenv("HERMES_PLATFORM", "discord")
    monkeypatch.setattr(skill_bundles_module, "_bundle_cache_snapshots", {})
    monkeypatch.setattr(
        skill_commands_module,
        "_skill_command_snapshots",
        {
            (home_key, "discord"): {
                "/only-discord": {"name": "only-discord"},
            },
            (home_key, "telegram"): {
                "/only-telegram": {
                    "name": "only-telegram",
                    "skill_dir": str(telegram_skill_dir),
                },
            },
        },
    )
    monkeypatch.setattr(
        skill_commands_module,
        "_skill_commands",
        {"/only-discord": {"name": "only-discord"}},
    )
    monkeypatch.setattr(
        skill_commands_module, "_skill_commands_home", home_key
    )
    monkeypatch.setattr(
        skill_commands_module, "_skill_commands_platform", "discord"
    )
    monkeypatch.setattr(
        "agent.skill_utils.get_disabled_skill_names", lambda **_kwargs: set()
    )

    import gateway.run as gateway_run_module

    with gateway_run_module._profile_runtime_scope(hermes_home):
        first = await runner._handle_message(
            MessageEvent(text="/only-discord raw", source=raw_source)
        )
        second = await runner._handle_message(
            MessageEvent(text="/only-telegram run", source=raw_source)
        )

    assert first is None and second is None
    assert active_agent.steered[0] == "/only-discord raw"
    assert "TELEGRAM SNAPSHOT SKILL BODY" in active_agent.steered[1]
    assert "run" in active_agent.steered[1]
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_stripped_topic_dynamic_bundle_expands_without_command_discovery(
    tmp_path, monkeypatch
):
    runner, raw_source, canonical_key, active_agent, adapter = (
        _busy_topic_command_runner(tmp_path, monkeypatch, busy_mode="steer")
    )
    import agent.skill_bundles as skill_bundles_module
    from hermes_constants import hermes_home_key

    hermes_home = tmp_path / ".hermes"
    bundle_skill_dir = _write_profile_skill(
        hermes_home,
        "bundle-skill",
        "TELEGRAM SNAPSHOT BUNDLE SKILL BODY",
    )
    home_key = hermes_home_key(hermes_home)
    monkeypatch.setenv("HERMES_PLATFORM", "discord")
    monkeypatch.setattr(skill_bundles_module, "_bundles_cache", {})
    monkeypatch.setattr(
        skill_bundles_module,
        "_bundle_cache_snapshots",
        {
            (home_key, "discord"): {},
            (home_key, "telegram"): {
                "/race-bundle": {
                    "name": "race-bundle",
                    "slug": "race-bundle",
                    "command": "/race-bundle",
                    "description": "Telegram-only bundle",
                    "skills": [str(bundle_skill_dir)],
                    "instruction": "",
                },
            },
        },
        raising=False,
    )

    monkeypatch.setattr(
        skill_bundles_module,
        "_bundles_cache_dir",
        str(skill_bundles_module._bundles_dir()),
    )
    plugin_handler_lookup = MagicMock(return_value=None)
    plugin_commands_lookup = MagicMock(return_value={})
    known_command_lookup = MagicMock(return_value=False)
    hook_dispatch = MagicMock(return_value=None)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler", plugin_handler_lookup
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_commands", plugin_commands_lookup
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.PluginManager.discover_and_load",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hermes_cli.commands.is_gateway_known_command", known_command_lookup
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", hook_dispatch
    )

    import gateway.run as gateway_run_module

    with gateway_run_module._profile_runtime_scope(hermes_home):
        result = await runner._handle_message(
            MessageEvent(text="/race-bundle inspect", source=raw_source)
        )

    assert result is None
    assert len(active_agent.steered) == 1
    assert "TELEGRAM SNAPSHOT BUNDLE SKILL BODY" in active_agent.steered[0]
    assert "inspect" in active_agent.steered[0]
    assert runner._peek_session_state(canonical_key).turn.agent is active_agent
    plugin_handler_lookup.assert_not_called()
    plugin_commands_lookup.assert_not_called()
    known_command_lookup.assert_not_called()
    hook_dispatch.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_race_routes_to_existing_canonical_agent(tmp_path, monkeypatch):
    """A turn appearing during a late await wins without sentinel corruption."""
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = GatewayConfig(sessions_dir=hermes_home / "sessions")
    store = SessionStore(config.sessions_dir, config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="race-chat",
        chat_type="dm",
        user_id="race-user",
    )
    key = store.get_or_create_session(source).session_key
    runner = _runner_with_store(config, store, store._db)
    runner._busy_input_mode = "steer"
    runner.adapters[Platform.TELEGRAM] = TelegramAdapter(
        PlatformConfig(enabled=True, token="e2e-test-token")
    )
    original_override = {"provider": "test", "model": "original"}
    runner._session_state(key).conversation.model_override = original_override

    class ActiveAgent:
        def __init__(self):
            self.steered = []

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

        def steer(self, text):
            self.steered.append(text)
            return True

    winner = ActiveAgent()

    async def _late_notice(_event, _key):
        state = runner._session_state(key)
        state.turn.agent = winner
        state.turn.started_ts = time.time()
        return False, None

    runner._maybe_handle_stale_override_notice = _late_notice
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("claim race dispatched a second turn")
    )
    result = await runner._handle_message(
        MessageEvent(text="race follow-up", source=source)
    )

    assert result is None
    assert winner.steered == ["race follow-up"]
    runner._handle_message_with_agent.assert_not_awaited()
    state = runner._peek_session_state(key)
    assert state.turn.agent is winner
    assert state.turn.lease is None
    assert state.conversation.model_override is original_override
    assert state.persistent.run_generation == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["lobby", "drain", "stale", "limit"])
async def test_rejected_moa_never_installs_temporary_override(
    tmp_path, monkeypatch, rejection
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = GatewayConfig(sessions_dir=hermes_home / "sessions")
    store = SessionStore(config.sessions_dir, config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="moa-chat",
        chat_type="dm",
        user_id="moa-user",
    )
    key = store.get_or_create_session(source).session_key
    runner = _runner_with_store(config, store, store._db)
    original_override = {"provider": "test", "model": "original"}
    runner._session_state(key).conversation.model_override = original_override
    cached_agent = object()
    runner._agent_cache = {key: cached_agent}
    runner._evict_cached_agent = MagicMock()
    runner._is_telegram_topic_root_lobby = MagicMock(
        return_value=rejection == "lobby"
    )
    if rejection == "drain":
        runner._external_drain_active = True
    if rejection == "stale":
        runner._maybe_handle_stale_override_notice = AsyncMock(
            return_value=(True, "held")
        )
    if rejection == "limit":
        runner._claim_active_session_slot = MagicMock(
            return_value=(None, "session limit")
        )
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("rejected MoA reached dispatch")
    )

    await runner._handle_message(MessageEvent(text="/moa compare", source=source))

    assert runner._session_state(key).conversation.model_override is original_override
    assert runner._agent_cache == {key: cached_agent}
    runner._evict_cached_agent.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_accepted_moa_applies_only_inside_claimed_restore_scope(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = GatewayConfig(sessions_dir=hermes_home / "sessions")
    store = SessionStore(config.sessions_dir, config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="moa-chat",
        chat_type="dm",
        user_id="moa-user",
    )
    key = store.get_or_create_session(source).session_key
    runner = _runner_with_store(config, store, store._db)
    original_override = {"provider": "test", "model": "original"}
    runner._session_state(key).conversation.model_override = original_override
    runner._evict_cached_agent = MagicMock()

    async def _agent(_event, _source, session_key, _generation):
        active_override = runner._session_state(
            session_key
        ).conversation.model_override
        assert active_override["provider"] == "moa"
        return "accepted"

    runner._handle_message_with_agent = _agent
    result = await runner._handle_message(
        MessageEvent(text="/moa compare", source=source)
    )

    assert result == "accepted"
    assert runner._session_state(key).conversation.model_override is original_override
    assert runner._evict_cached_agent.call_count == 2
