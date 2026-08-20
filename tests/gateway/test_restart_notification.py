"""Tests for /restart notification — the gateway notifies the requester on comeback."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import (
    make_restart_runner,
    make_restart_source,
)


# ── restart marker helpers ───────────────────────────────────────────────


def test_planned_restart_notification_pending_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    marker = tmp_path / ".restart_pending.json"

    assert gateway_run._planned_restart_notification_pending() is False
    marker.write_text("{}")
    assert gateway_run._planned_restart_notification_pending() is True

    gateway_run._clear_planned_restart_notification()

    assert gateway_run._planned_restart_notification_pending() is False


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
    monkeypatch.setattr("gateway.slash_commands.persist_home_channel", lambda home, **kwargs: None)

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
    monkeypatch.setattr("gateway.slash_commands.persist_home_channel", lambda home, **kwargs: None)

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
async def test_relay_fronted_logical_home_gets_startup_notification(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner, _native = make_restart_runner()
    relay = MagicMock()
    relay.fronts_platform.side_effect = lambda platform: platform == Platform.SLACK
    relay.send_for_platform = AsyncMock(return_value=SendResult(success=True, message_id="home"))
    runner.adapters = {Platform.RELAY: relay}
    runner.config.platforms = {
        Platform.RELAY: PlatformConfig(enabled=True),
        Platform.SLACK: PlatformConfig(
            enabled=False,
            home_channel=HomeChannel(
                platform=Platform.SLACK,
                chat_id="D123",
                name="Owner DM",
                user_id="U123",
                scope_id="T123",
            ),
        ),
    }

    delivered = await runner._send_home_channel_startup_notifications()

    assert delivered == {("slack", "D123", None)}
    relay.send_for_platform.assert_awaited_once()
    assert relay.send_for_platform.await_args.args[:3] == (
        Platform.SLACK,
        "D123",
        "♻️ Gateway online — Hermes is back and ready.",
    )
    assert relay.send_for_platform.await_args.kwargs["metadata"]["user_id"] == "U123"
    assert relay.send_for_platform.await_args.kwargs["metadata"]["scope_id"] == "T123"


# ── _send_restart_notification ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_relay_restart_notification_uses_logical_platform_and_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps(
            {
                "platform": "slack",
                "chat_id": "D123",
                "chat_type": "dm",
                "user_id": "U123",
                "scope_id": "T123",
                "delivered_via_upstream_relay": True,
            }
        )
    )

    runner, _native = make_restart_runner()
    relay = MagicMock()
    relay.fronts_platform.side_effect = lambda platform: platform == Platform.SLACK
    relay.send_for_platform = AsyncMock(
        return_value=SendResult(success=True, message_id="restart")
    )
    runner.adapters = {Platform.RELAY: relay}
    runner.config.platforms = {
        Platform.RELAY: PlatformConfig(enabled=True),
        Platform.SLACK: PlatformConfig(enabled=False),
    }

    delivered_target = await runner._send_restart_notification()

    assert delivered_target == ("slack", "D123", None)
    relay.send_for_platform.assert_awaited_once()
    assert relay.send_for_platform.await_args.args[0:2] == (Platform.SLACK, "D123")
    metadata = relay.send_for_platform.await_args.kwargs["metadata"]
    assert metadata["user_id"] == "U123"
    assert metadata["scope_id"] == "T123"
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
async def test_shutdown_notifications_are_fully_muted_when_flag_disabled():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="active-42", chat_type="group", thread_id="topic-7")
    session_key = build_session_key(source)

    runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    runner._running_agents[session_key] = object()
    runner.session_store._entries[session_key] = MagicMock(origin=source)
    adapter.send = AsyncMock()

    await runner._notify_active_sessions_of_shutdown()

    adapter.send.assert_not_awaited()


# ── unexpected-restart startup notification ────────────────────────────────


def _simulate_startup_state_snapshot(monkeypatch, tmp_path):
    """Reproduce the state-snapshot timing from ``GatewayRunner.start()``.

    ``start()`` snapshots ``prev_gateway_state`` *before* overwriting the
    status file with ``"starting"`` / ``"running"``.  This helper mirrors
    that sequence so tests can verify the decision is based on the
    *previous* gateway's state, not the current startup's writes.
    """
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    from gateway.status import read_runtime_status, write_runtime_status

    # 1. Snapshot previous state (what start() does at the very top).
    prev_state = read_runtime_status()
    prev_gateway_state = (prev_state or {}).get("gateway_state")

    # 2. Simulate startup overwriting the status file (start() writes
    #    "starting" at line ~8117, then "running" at line ~8665).
    write_runtime_status(gateway_state="starting", exit_reason=None)
    write_runtime_status(gateway_state="running", exit_reason=None)

    return prev_gateway_state


@pytest.mark.asyncio
async def test_unexpected_restart_sends_home_channel_notification(tmp_path, monkeypatch):
    """When the previous gateway died while running (state='running'),
    the next boot treats it as an unexpected restart and notifies home channels
    even without a .restart_pending.json marker.

    This test reproduces the start() lifecycle: the previous state is
    snapshotted *before* the startup writes "starting"/"running", so the
    decision uses the stale "running" from the dead gateway, not the
    current boot's "running".
    """
    from gateway.status import write_runtime_status

    # Pre-seed: previous gateway died while "running" (SIGTERM, OOM, etc.)
    write_runtime_status(gateway_state="running")

    # Run the snapshot sequence - mimics what start() does.
    prev_gateway_state = _simulate_startup_state_snapshot(monkeypatch, tmp_path)

    # The snapshot captured "running" from the *previous* gateway, even
    # though the status file now says "running" from *this* boot.
    assert prev_gateway_state == "running"

    # No planned-restart marker, no chat-restart marker -> unexpected restart.
    assert gateway_run._planned_restart_notification_pending() is False
    assert gateway_run._restart_notification_pending() is False

    # The decision branch in start() would fire the notifier:
    if prev_gateway_state in ("running", "draining"):
        runner, adapter = make_restart_runner()
        runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id="home-99",
            name="Ops Home",
        )
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="home"))
        delivered = await runner._send_home_channel_startup_notifications()
        assert delivered == {("telegram", "home-99", None)}
        adapter.send.assert_called_once()
        assert "Gateway online" in adapter.send.call_args.args[1]


@pytest.mark.asyncio
async def test_clean_stop_does_not_trigger_startup_notification(tmp_path, monkeypatch):
    """When the previous gateway stopped cleanly (state='stopped'), the next
    boot does NOT send a startup notification - the operator intentionally
    stopped it and doesn't need a "back online" ping on manual restart.

    Crucially, even though start() overwrites the status file with
    "starting"/"running" during boot, the snapshot was taken *before* those
    writes, so it correctly captures "stopped".
    """
    from gateway.status import write_runtime_status

    # Pre-seed: previous gateway stopped cleanly.
    write_runtime_status(gateway_state="stopped")

    # Run the snapshot sequence - mimics what start() does.
    prev_gateway_state = _simulate_startup_state_snapshot(monkeypatch, tmp_path)

    # The snapshot captured "stopped" from the *previous* gateway.
    assert prev_gateway_state == "stopped"

    # No planned-restart marker, no chat-restart marker.
    assert gateway_run._planned_restart_notification_pending() is False
    assert gateway_run._restart_notification_pending() is False

    # "stopped" is NOT in the trigger set -> no notification should fire.
    assert prev_gateway_state not in ("running", "draining")


def test_first_boot_has_no_runtime_state(tmp_path, monkeypatch):
    """On a first-ever boot, gateway_state.json doesn't exist at all.
    The snapshot correctly captures None, which is not in the trigger set,
    so no notification fires on fresh install.

    This also verifies the timing fix: even after start() writes
    "starting"/"running", the snapshot was taken before those writes and
    correctly reflects the absence of a prior state file.
    """
    # No pre-seed: first-ever boot, no state file exists.

    # Run the snapshot sequence - mimics what start() does.
    prev_gateway_state = _simulate_startup_state_snapshot(monkeypatch, tmp_path)

    # No prior state file -> snapshot is None.
    assert prev_gateway_state is None
    # None is NOT in the trigger set -> no notification on first boot.
    assert prev_gateway_state not in ("running", "draining")
