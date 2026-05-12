"""Tests for /restart notification — the gateway notifies the requester on comeback."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ── restart marker helpers ───────────────────────────────────────────────


def test_restart_notification_pending_false_without_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    assert gateway_run._restart_notification_pending() is False


def test_restart_notification_pending_true_with_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / ".restart_notify.json").write_text("{}")

    assert gateway_run._restart_notification_pending() is True


def test_planned_restart_notification_pending_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    marker = tmp_path / ".restart_pending.json"

    assert gateway_run._planned_restart_notification_pending() is False
    marker.write_text("{}")
    assert gateway_run._planned_restart_notification_pending() is True

    gateway_run._clear_planned_restart_notification()

    assert gateway_run._planned_restart_notification_pending() is False


async def _start_gateway_for_lifecycle_test(
    tmp_path,
    monkeypatch,
    *,
    startup_notification: bool,
    planned_restart_pending: bool,
    chat_restart_pending: bool,
):
    """Exercise GatewayRunner.start() while isolating external platform/runtime I/O."""
    runner, adapter = make_restart_runner()
    runner.config.sessions_dir = tmp_path / "sessions"
    runner.config.gateway_startup_notification = startup_notification
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-123",
        name="Home",
        thread_id="topic-456",
    )
    runner._running = False
    runner._exit_with_failure = False
    runner._exit_cleanly = False
    runner._failed_platforms = {}
    setattr(runner, "_honcho_managers", {})
    setattr(runner, "_honcho_configs", {})
    setattr(runner, "_create_adapter", MagicMock(return_value=adapter))
    setattr(
        runner,
        "_connect_adapter_with_timeout",
        AsyncMock(return_value=True),
    )
    setattr(runner, "_update_platform_runtime_status", MagicMock())
    setattr(runner, "_sync_voice_mode_state_to_adapter", MagicMock())
    setattr(
        runner,
        "_start_secondary_profile_adapters",
        AsyncMock(return_value=0),
    )
    setattr(runner, "_suspend_stuck_loop_sessions", MagicMock(return_value=0))
    setattr(runner, "_send_update_notification", AsyncMock(return_value=True))
    setattr(runner, "_schedule_resume_pending_sessions", MagicMock())
    setattr(runner, "_finish_startup_restore", AsyncMock())
    setattr(runner, "_wire_teams_pipeline_runtime", MagicMock())
    runner.hooks = MagicMock()
    runner.hooks.loaded_hooks = []
    runner.hooks.emit = AsyncMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    if planned_restart_pending:
        (tmp_path / ".restart_pending.json").write_text(
            "{}",
            encoding="utf-8",
        )
    if chat_restart_pending:
        (tmp_path / ".restart_notify.json").write_text(
            json.dumps(
                {
                    "platform": "telegram",
                    "chat_id": "origin-999",
                    "chat_type": "group",
                    "thread_id": "thread-888",
                    "message_id": "message-777",
                }
            ),
            encoding="utf-8",
        )

    # Avoid touching the real session DB/process checkpoint and avoid spawning
    # long-lived watcher tasks. Closing each coroutine keeps asyncio warning-free.
    (tmp_path / ".clean_shutdown").write_text("", encoding="utf-8")

    def fake_create_task(coro):
        coro.close()
        return MagicMock()

    with patch("gateway.status.write_runtime_status"):
        with patch("gateway.shutdown_forensics.check_systemd_timing_alignment", return_value=None):
            with patch("hermes_cli.plugins.discover_plugins"):
                with patch(
                    "gateway.platform_registry.platform_registry.plugin_entries",
                    return_value=[],
                ):
                    with patch("hermes_cli.config.load_config", return_value={}):
                        with patch("agent.shell_hooks.register_from_config"):
                            with patch(
                                "tools.process_registry.process_registry.recover_from_checkpoint",
                                return_value=0,
                            ):
                                with patch(
                                    "gateway.channel_directory.build_channel_directory",
                                    new=AsyncMock(return_value={"platforms": {}}),
                                ):
                                    with patch(
                                        "gateway.run.asyncio.create_task",
                                        side_effect=fake_create_task,
                                    ):
                                        with patch(
                                            "gateway.run.asyncio.sleep",
                                            new=AsyncMock(),
                                        ):
                                            assert await runner.start() is True

    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "startup_notification",
        "planned_restart_pending",
        "chat_restart_pending",
        "expected_home_calls",
    ),
    [
        pytest.param(False, False, False, 0, id="ordinary-start-default-off"),
        pytest.param(True, False, False, 1, id="ordinary-start-opted-in"),
        pytest.param(False, True, False, 1, id="planned-restart"),
        pytest.param(True, False, True, 0, id="chat-restart-target-only"),
        pytest.param(
            True,
            True,
            True,
            0,
            id="chat-restart-wins-over-stale-planned-marker",
        ),
    ],
)
async def test_start_lifecycle_routes_online_notifications_without_duplicates(
    tmp_path,
    monkeypatch,
    startup_notification,
    planned_restart_pending,
    chat_restart_pending,
    expected_home_calls,
):
    _runner, adapter = await _start_gateway_for_lifecycle_test(
        tmp_path,
        monkeypatch,
        startup_notification=startup_notification,
        planned_restart_pending=planned_restart_pending,
        chat_restart_pending=chat_restart_pending,
    )

    sent_calls = getattr(adapter, "sent_calls")
    if expected_home_calls:
        assert len(sent_calls) == 1
        chat_id, content, metadata = sent_calls[0]
        assert chat_id == "home-123"
        assert content == "♻️ Gateway online — Hermes is back and ready."
        assert metadata["thread_id"] == "topic-456"
    elif chat_restart_pending:
        # A chat-originated /restart gets exactly one targeted reply and no
        # duplicate online ping in the configured home channel.
        assert len(sent_calls) == 1
        chat_id, content, metadata = sent_calls[0]
        assert chat_id == "origin-999"
        assert "Gateway restarted successfully" in content
        assert metadata["thread_id"] == "thread-888"
    else:
        assert sent_calls == []

    assert not (tmp_path / ".restart_notify.json").exists()
    assert not (tmp_path / ".restart_pending.json").exists()


def test_home_channel_startup_notification_skipped_on_cold_start_by_default():
    runner, _adapter = make_restart_runner()

    assert runner._should_send_home_channel_startup_notifications(
        planned_restart_notification_pending=False,
        restart_notification_pending=False,
    ) is False


def test_home_channel_startup_notification_runs_on_planned_restart_marker():
    runner, _adapter = make_restart_runner()

    assert runner._should_send_home_channel_startup_notifications(
        planned_restart_notification_pending=True,
        restart_notification_pending=False,
    ) is True


def test_home_channel_startup_notification_keeps_chat_restart_quiet_even_when_opted_in():
    runner, _adapter = make_restart_runner()
    runner.config.gateway_startup_notification = True

    assert runner._should_send_home_channel_startup_notifications(
        planned_restart_notification_pending=False,
        restart_notification_pending=True,
    ) is False


def test_chat_restart_marker_takes_priority_over_planned_restart_marker():
    runner, _adapter = make_restart_runner()
    runner.config.gateway_startup_notification = True

    assert runner._should_send_home_channel_startup_notifications(
        planned_restart_notification_pending=True,
        restart_notification_pending=True,
    ) is False


def test_home_channel_startup_notification_runs_on_opt_in_cold_start():
    runner, _adapter = make_restart_runner()
    runner.config.gateway_startup_notification = True

    assert runner._should_send_home_channel_startup_notifications(
        planned_restart_notification_pending=False,
        restart_notification_pending=False,
    ) is True


# ── _handle_restart_command writes .restart_notify.json ──────────────────


@pytest.mark.asyncio
async def test_restart_command_writes_notify_file(tmp_path, monkeypatch):
    """When /restart fires, the requester's routing info is persisted to disk."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="42")
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )

    result = await runner._handle_restart_command(event)
    assert "Restarting" in result

    notify_path = tmp_path / ".restart_notify.json"
    assert notify_path.exists()
    data = json.loads(notify_path.read_text())
    assert data["platform"] == "telegram"
    assert data["chat_id"] == "42"
    assert data["chat_type"] == "dm"
    assert data["message_id"] == "m1"
    assert "thread_id" not in data  # no thread → omitted


@pytest.mark.asyncio
async def test_restart_command_uses_service_restart_under_systemd(tmp_path, monkeypatch):
    """Under systemd (INVOCATION_ID set), /restart uses via_service=True."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("INVOCATION_ID", "abc123")

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="42")
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )

    await runner._handle_restart_command(event)
    runner.request_restart.assert_called_once_with(detached=False, via_service=True)


@pytest.mark.asyncio
async def test_restart_command_uses_detached_without_systemd(tmp_path, monkeypatch):
    """Without systemd or a container runtime, /restart uses the detached subprocess approach."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    import gateway.slash_commands as slash_commands

    real_exists = slash_commands.os.path.exists

    def fake_exists(path):
        if path in {"/.dockerenv", "/run/.containerenv"}:
            return False
        return real_exists(path)

    monkeypatch.setattr(slash_commands.os.path, "exists", fake_exists)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="42")
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )

    await runner._handle_restart_command(event)
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


@pytest.mark.asyncio
async def test_restart_command_preserves_thread_id(tmp_path, monkeypatch):
    """Thread ID is saved when the requester is in a threaded chat."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="99", thread_id="777")

    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m2",
    )

    await runner._handle_restart_command(event)

    data = json.loads((tmp_path / ".restart_notify.json").read_text())
    assert data["chat_type"] == "dm"
    assert data["thread_id"] == "777"
    assert data["message_id"] == "m2"


@pytest.mark.asyncio
async def test_restart_command_uses_atomic_json_writes_for_marker_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    calls = []

    def _fake_atomic_json_write(path, payload, **kwargs):
        calls.append((Path(path).name, payload, kwargs))

    # _handle_restart_command lives in gateway/slash_commands.py (extracted from
    # run.py); it uses that module's top-level atomic_json_write import.
    import gateway.slash_commands as gateway_slash
    monkeypatch.setattr(gateway_slash, "atomic_json_write", _fake_atomic_json_write)
    monkeypatch.setattr(gateway_run, "atomic_json_write", _fake_atomic_json_write)

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)

    source = make_restart_source(chat_id="42")
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )

    await runner._handle_restart_command(event)

    names = [name for name, _payload, _kwargs in calls]
    assert names == [".restart_notify.json", ".restart_last_processed.json"]
    assert calls[0][1]["chat_id"] == "42"
    assert calls[1][1]["platform"] == "telegram"


@pytest.mark.asyncio
async def test_sethome_updates_running_config_for_same_process_restart(tmp_path, monkeypatch):
    """/sethome persists to env and updates in-memory config before restart."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    saved = {}

    def _fake_save_env_value(key, value):
        saved[key] = value

    monkeypatch.setattr("hermes_cli.config.save_env_value", _fake_save_env_value)

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="home-42")
    source.chat_name = "Ops Home"
    event = MessageEvent(
        text="/sethome",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m-home",
    )

    result = await runner._handle_set_home_command(event)

    home = runner.config.get_home_channel(Platform.TELEGRAM)
    assert "Home channel set" in result
    assert saved["TELEGRAM_HOME_CHANNEL"] == "home-42"
    assert home is not None
    assert home.chat_id == "home-42"
    assert home.name == "Ops Home"


@pytest.mark.asyncio
async def test_sethome_preserves_thread_target_for_same_process_restart(tmp_path, monkeypatch):
    """/sethome from a topic/thread stores the thread-aware home target."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    saved = {}

    def _fake_save_env_value(key, value):
        saved[key] = value

    monkeypatch.setattr("hermes_cli.config.save_env_value", _fake_save_env_value)

    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="parent-42", thread_id="topic-7")
    source.chat_name = "Ops Topic"
    event = MessageEvent(
        text="/sethome",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m-home-thread",
    )

    result = await runner._handle_set_home_command(event)

    home = runner.config.get_home_channel(Platform.TELEGRAM)
    assert "Home channel set" in result
    assert saved["TELEGRAM_HOME_CHANNEL"] == "parent-42"
    assert saved["TELEGRAM_HOME_CHANNEL_THREAD_ID"] == "topic-7"
    assert home is not None
    assert home.chat_id == "parent-42"
    assert home.thread_id == "topic-7"


# ── home-channel startup notifications ─────────────────────────────────────


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_to_configured_home(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock()

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "home-42", None)}
    adapter.send.assert_called_once_with(
        "home-42",
        "♻️ Gateway online — Hermes is back and ready.",
    )


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_preserves_thread_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="parent-42",
        name="Ops Topic",
        thread_id="777",
    )
    # Declare the DM-topic lookup on the adapter CLASS, not the instance.
    # _is_telegram_dm_topic_target resolves _get_dm_topic_info via type(adapter)
    # so a MagicMock auto-attribute (instance-level) is intentionally ignored;
    # a real adapter exposes the method on its class. Mirrors the fake-adapter
    # pattern in test_telegram_topic_mode.py.
    class _DmTopicAdapter(type(adapter)):
        def _get_dm_topic_info(self, chat_id, thread_id):
            return {"name": "Ops Topic"}

    adapter.__class__ = _DmTopicAdapter
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "parent-42", "777")}
    adapter.send.assert_called_once_with(
        "parent-42",
        "♻️ Gateway online — Hermes is back and ready.",
        metadata={
            "thread_id": "777",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "777",
        },
    )


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_skips_restart_target(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="42",
        name="Ops Home",
    )
    adapter.send = AsyncMock()

    delivered = await runner._send_home_channel_startup_notifications(
        skip_targets={("telegram", "42", None)}
    )

    assert delivered == set()
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_does_not_skip_different_thread(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    delivered = await runner._send_home_channel_startup_notifications(
        skip_targets={("telegram", "42", "topic-7")}
    )

    assert delivered == {("telegram", "42", None)}
    adapter.send.assert_called_once()


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_ignores_false_send_result(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="network down"))

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == set()
    adapter.send.assert_called_once()


# ── _send_restart_notification ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_restart_notification_delivers_and_cleans_up(tmp_path, monkeypatch):
    """On startup, the notification is sent and the file is removed."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()

    delivered_target = await runner._send_restart_notification()

    assert delivered_target == ("telegram", "42", None)
    adapter.send.assert_called_once()
    call_args = adapter.send.call_args
    assert call_args[0][0] == "42"  # chat_id
    assert "restarted" in call_args[0][1].lower()
    assert call_args[1].get("metadata") is None  # no thread
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_with_thread(tmp_path, monkeypatch):
    """Thread ID is passed as metadata so the message lands in the right topic."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "99",
        "chat_type": "dm",
        "thread_id": "777",
        "message_id": "m2",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()

    delivered_target = await runner._send_restart_notification()

    assert delivered_target == ("telegram", "99", "777")
    call_args = adapter.send.call_args
    assert call_args[1]["metadata"] == {
        "thread_id": "777",
        "telegram_dm_topic_reply_fallback": True,
        "direct_messages_topic_id": "777",
        "telegram_reply_to_message_id": "m2",
    }
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_noop_when_no_file(tmp_path, monkeypatch):
    """Nothing happens if there's no pending restart notification."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()

    await runner._send_restart_notification()

    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_restart_notification_skips_when_adapter_missing(tmp_path, monkeypatch):
    """If the requester's platform isn't connected, clean up without crashing."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "discord",  # runner only has telegram adapter
        "chat_id": "42",
    }))

    runner, _adapter = make_restart_runner()

    await runner._send_restart_notification()

    # File cleaned up even though we couldn't send
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_cleans_up_on_send_failure(
    tmp_path, monkeypatch
):
    """If the adapter.send() raises, the file is still cleaned up."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(side_effect=RuntimeError("network down"))

    delivered_target = await runner._send_restart_notification()

    # File cleaned up even though send raised.
    assert delivered_target is None
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_logs_warning_on_sendresult_failure(
    tmp_path, monkeypatch, caplog
):
    """Adapter that returns SendResult(success=False) must log a WARNING, not INFO.

    Regression guard: adapter.send() catches provider errors (e.g. Telegram
    "Chat not found") and returns SendResult(success=False) rather than
    raising. The caller previously ignored the return value and always
    logged "Sent restart notification to ..." at INFO — masking real
    delivery failures behind a fake success line.
    """
    from gateway.platforms.base import SendResult

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(
        return_value=SendResult(success=False, error="Chat not found"),
    )

    with caplog.at_level("DEBUG", logger="gateway.run"):
        delivered_target = await runner._send_restart_notification()

    success_lines = [
        r for r in caplog.records
        if r.levelname == "INFO" and "Sent restart notification" in r.getMessage()
    ]
    warning_lines = [
        r for r in caplog.records
        if r.levelname == "WARNING"
        and "was not delivered" in r.getMessage()
        and "Chat not found" in r.getMessage()
    ]
    assert delivered_target is None
    assert not success_lines, (
        "Expected no INFO 'Sent restart notification' line when send failed, "
        f"got: {[r.getMessage() for r in success_lines]}"
    )
    assert warning_lines, (
        "Expected a WARNING line mentioning the failure; "
        f"got records: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    # Still cleans up.
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_skipped_when_flag_disabled(
    tmp_path, monkeypatch
):
    """Per-platform opt-out: gateway_restart_notification=False mutes the home-channel ping."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    adapter.send = AsyncMock()

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == set()
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_home_channel_startup_notification_default_flag_true(
    tmp_path, monkeypatch
):
    """Default behavior is unchanged: missing flag means notifications still fire."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, adapter = make_restart_runner()
    # Sanity-check the dataclass default — guards against future refactors
    # silently flipping the default to False.
    assert runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification is True

    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("telegram", "home-42", None)}
    adapter.send.assert_called_once()


@pytest.mark.asyncio
async def test_send_restart_notification_skipped_when_flag_disabled(
    tmp_path, monkeypatch
):
    """The /restart originator's notification also honors the per-platform flag.

    Slack used by end users → flag off → no "Gateway restarted" message even
    when an end user accidentally triggers /restart. The marker file is still
    cleaned up so the notification doesn't leak into the next boot.
    """
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    adapter.send = AsyncMock()

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    adapter.send.assert_not_called()
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_logs_info_on_sendresult_success(
    tmp_path, monkeypatch, caplog
):
    """Adapter returning SendResult(success=True) keeps the INFO log line."""
    from gateway.platforms.base import SendResult

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(json.dumps({
        "platform": "telegram",
        "chat_id": "42",
    }))

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m-1"))

    with caplog.at_level("DEBUG", logger="gateway.run"):
        delivered_target = await runner._send_restart_notification()

    success_lines = [
        r for r in caplog.records
        if r.levelname == "INFO" and "Sent restart notification" in r.getMessage()
    ]
    assert delivered_target == ("telegram", "42", None)
    assert success_lines, (
        "Expected INFO 'Sent restart notification' when send succeeded; "
        f"got records: {[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_shutdown_notifications_use_cached_live_thread_source_when_origin_missing():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="parent-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)

    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=None)
    runner._cache_session_source(session_key, source)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="shutdown"))

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_awaited_once_with(
        "parent-42",
        "⚠️ Gateway shutting down — Your current task will be interrupted.",
        metadata={"thread_id": "topic-7"},
    )


@pytest.mark.asyncio
async def test_restart_shutdown_notification_anchors_telegram_dm_topic():
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    source = make_restart_source(chat_id="123456", thread_id="20197")
    source.message_id = "462"
    session_key = build_session_key(source)

    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=source)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="shutdown"))

    await runner._notify_active_sessions_of_shutdown()

    call = adapter.send.await_args
    assert call.args[0] == "123456"
    assert "Gateway restarting" in call.args[1]
    assert call.kwargs["metadata"] == {
        "thread_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "direct_messages_topic_id": "20197",
        "telegram_reply_to_message_id": "462",
    }
