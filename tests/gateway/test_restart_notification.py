"""Tests for /restart notification — the gateway notifies the requester on comeback."""

import asyncio
import json
import threading
import time
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
    assert len(data["request_id"]) == 32
    assert "thread_id" not in data  # no thread → omitted


@pytest.mark.asyncio
async def test_restart_command_gives_identical_route_a_unique_request_id(
    tmp_path, monkeypatch
):
    """Consecutive markers from the same route must remain distinguishable."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="42"),
    )

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    await runner._handle_restart_command(event)
    first = json.loads((tmp_path / ".restart_notify.json").read_text(encoding="utf-8"))

    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    await runner._handle_restart_command(event)
    second = json.loads((tmp_path / ".restart_notify.json").read_text(encoding="utf-8"))

    assert first["request_id"] != second["request_id"]
    assert {k: v for k, v in first.items() if k != "request_id"} == {
        k: v for k, v in second.items() if k != "request_id"
    }


@pytest.mark.asyncio
async def test_restart_command_serializes_async_marker_writes(tmp_path, monkeypatch):
    """Concurrent restart commands cannot reorder their worker-thread writes."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    import gateway.slash_commands as gateway_slash

    notify_write_started = threading.Event()
    allow_notify_write = threading.Event()
    notify_payloads = []

    def _blocking_atomic_json_write(path, payload, **_kwargs):
        path = Path(path)
        if path.name == ".restart_notify.json":
            notify_payloads.append(dict(payload))
            notify_write_started.set()
            assert allow_notify_write.wait(timeout=2)
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        gateway_slash,
        "atomic_json_write",
        _blocking_atomic_json_write,
    )

    runner, _adapter = make_restart_runner()

    def _request_restart(**_kwargs):
        runner._restart_requested = True
        return True

    runner.request_restart = MagicMock(side_effect=_request_restart)
    first_event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="first"),
        message_id="m1",
    )
    second_event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="second"),
        message_id="m2",
    )

    first_task = asyncio.create_task(runner._handle_restart_command(first_event))
    assert await asyncio.to_thread(notify_write_started.wait, 2)
    second_task = asyncio.create_task(runner._handle_restart_command(second_event))
    await asyncio.sleep(0)

    try:
        assert not second_task.done()
    finally:
        allow_notify_write.set()

    await asyncio.gather(first_task, second_task)

    runner.request_restart.assert_called_once()
    assert len(notify_payloads) == 1
    assert notify_payloads[0]["chat_id"] == "first"
    marker = json.loads(
        (tmp_path / ".restart_notify.json").read_text(encoding="utf-8")
    )
    assert marker["chat_id"] == "first"


@pytest.mark.asyncio
async def test_restart_command_cancellation_does_not_orphan_marker_writer(
    tmp_path, monkeypatch
):
    """Cancellation keeps the command lock until its worker write completes."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    import gateway.slash_commands as gateway_slash

    first_write_started = threading.Event()
    second_write_started = threading.Event()
    allow_first_write = threading.Event()

    def _ordered_atomic_json_write(path, payload, **_kwargs):
        path = Path(path)
        if path.name == ".restart_notify.json":
            if payload["chat_id"] == "first":
                first_write_started.set()
                assert allow_first_write.wait(timeout=2)
            else:
                second_write_started.set()
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(gateway_slash, "atomic_json_write", _ordered_atomic_json_write)

    runner, _adapter = make_restart_runner()

    def _request_restart(**_kwargs):
        runner._restart_requested = True
        return True

    runner.request_restart = MagicMock(side_effect=_request_restart)
    first_event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="first"),
        message_id="m1",
    )
    second_event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="second"),
        message_id="m2",
    )

    first_task = asyncio.create_task(runner._handle_restart_command(first_event))
    assert await asyncio.to_thread(first_write_started.wait, 2)
    first_task.cancel()
    await asyncio.sleep(0)
    first_task.cancel()
    second_task = asyncio.create_task(runner._handle_restart_command(second_event))
    second_started_before_release = await asyncio.to_thread(
        second_write_started.wait, 0.2
    )

    allow_first_write.set()
    results = await asyncio.gather(first_task, second_task, return_exceptions=True)

    assert not second_started_before_release
    assert isinstance(results[0], asyncio.CancelledError)
    runner.request_restart.assert_called_once()
    marker = json.loads(
        (tmp_path / ".restart_notify.json").read_text(encoding="utf-8")
    )
    assert marker["chat_id"] == "second"


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
async def test_send_restart_notification_retries_when_adapter_appears(
    tmp_path, monkeypatch
):
    """A platform reconnect during startup must not lose the notification."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    send = AsyncMock(return_value=SendResult(success=True, message_id="m-1"))
    adapter.send = send
    runner.adapters = {}

    async def _restore_adapter(_delay):
        runner.adapters[Platform.TELEGRAM] = adapter

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _restore_adapter)

    delivered_target = await runner._send_restart_notification()

    assert delivered_target == ("telegram", "42", None)
    send.assert_awaited_once()
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_preserves_captured_marker_if_first_read_fails(
    tmp_path, monkeypatch
):
    """A transient first-read failure preserves the captured unsent obligation."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps(
            {
                "platform": "telegram",
                "chat_id": "42",
                "request_id": "captured",
            }
        ),
        encoding="utf-8",
    )
    claimed_payload = notify_path.read_text(encoding="utf-8")

    runner, adapter = make_restart_runner()
    send = AsyncMock()
    adapter.send = send
    real_read_text = Path.read_text
    failed = False

    def _fail_first_marker_read(path, *args, **kwargs):
        nonlocal failed
        if path == notify_path and not failed:
            failed = True
            raise OSError("temporary marker read failure")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fail_first_marker_read)

    delivered_target = await runner._send_restart_notification(
        claimed_marker_payload=claimed_payload
    )

    assert delivered_target is None
    send.assert_not_awaited()
    assert real_read_text(notify_path, encoding="utf-8") == claimed_payload


@pytest.mark.parametrize(
    "marker_payload",
    [
        "{not-json",
        "[]",
        json.dumps({"platform": "not-a-platform", "chat_id": "42"}),
        json.dumps({"platform": "telegram"}),
        json.dumps({"platform": "telegram", "chat_id": "   "}),
        json.dumps({"platform": "telegram", "chat_id": {"not": "a route"}}),
        json.dumps({"platform": "telegram", "chat_id": ["42"]}),
        json.dumps({"platform": "telegram", "chat_id": 42}),
        json.dumps({"platform": ["telegram"], "chat_id": "42"}),
        json.dumps(
            {
                "platform": "telegram",
                "chat_id": "42",
                "thread_id": {"not": "a thread"},
            }
        ),
    ],
    ids=[
        "invalid-json",
        "non-object",
        "unknown-platform",
        "missing-route",
        "blank-chat-id",
        "object-chat-id",
        "list-chat-id",
        "integer-chat-id",
        "list-platform",
        "object-thread-id",
    ],
)
@pytest.mark.asyncio
async def test_send_restart_notification_preserves_malformed_marker(
    tmp_path, monkeypatch, marker_payload
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(marker_payload, encoding="utf-8")

    runner, adapter = make_restart_runner()
    send = AsyncMock()
    adapter.send = send

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_not_awaited()
    assert notify_path.read_text(encoding="utf-8") == marker_payload


@pytest.mark.asyncio
async def test_send_restart_notification_preserves_unreadable_marker(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    notify_path = tmp_path / ".restart_notify.json"
    marker_bytes = b"\xff\xfe"
    notify_path.write_bytes(marker_bytes)

    runner, adapter = make_restart_runner()
    send = AsyncMock()
    adapter.send = send

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_not_awaited()
    assert notify_path.read_bytes() == marker_bytes


@pytest.mark.asyncio
async def test_send_restart_notification_retries_retryable_refusal(
    tmp_path, monkeypatch
):
    """A safe pre-send refusal is retried instead of consuming the marker."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    send = AsyncMock(
        side_effect=[
            SendResult(
                success=False,
                error="send_path_degraded",
                retryable=True,
            ),
            SendResult(success=True, message_id="m-1"),
        ]
    )
    adapter.send = send
    sleep = AsyncMock()
    monkeypatch.setattr(gateway_run.asyncio, "sleep", sleep)

    delivered_target = await runner._send_restart_notification()

    assert delivered_target == ("telegram", "42", None)
    assert send.await_count == 2
    sleep.assert_awaited_once_with(1.0)
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_stops_when_marker_replaced_during_backoff(
    tmp_path, monkeypatch
):
    """A newer restart marker supersedes an older worker before its retry."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps(
            {
                "platform": "telegram",
                "chat_id": "42",
                "request_id": "old",
            }
        ),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    send = AsyncMock(
        side_effect=[
            SendResult(
                success=False,
                error="send_path_degraded",
                retryable=True,
            ),
            SendResult(success=True, message_id="must-not-send"),
        ]
    )
    adapter.send = send

    async def _replace_marker(_delay):
        notify_path.write_text(
            json.dumps(
                {
                    "platform": "telegram",
                    "chat_id": "42",
                    "request_id": "new",
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _replace_marker)

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_awaited_once()
    assert json.loads(notify_path.read_text(encoding="utf-8"))["request_id"] == "new"


@pytest.mark.asyncio
async def test_send_restart_notification_stops_when_async_replacement_is_announced(
    tmp_path, monkeypatch
):
    """A queued worker-thread write supersedes old bytes before its retry."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    old_payload = json.dumps(
        {"platform": "telegram", "chat_id": "42", "request_id": "old"}
    )
    notify_path.write_text(old_payload, encoding="utf-8")

    runner, adapter = make_restart_runner()
    calls = 0

    async def _announce_replacement_before_write(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            runner._restart_notification_request_id = "new"
            return SendResult(
                success=False,
                error="send_path_degraded",
                retryable=True,
            )
        return SendResult(success=True, message_id="must-not-send")

    send = AsyncMock(side_effect=_announce_replacement_before_write)
    adapter.send = send
    monkeypatch.setattr(gateway_run.asyncio, "sleep", AsyncMock())

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_awaited_once()
    assert notify_path.read_text(encoding="utf-8") == old_payload


@pytest.mark.asyncio
async def test_send_restart_notification_revalidates_marker_in_dispatch_task(
    tmp_path, monkeypatch
):
    """A replacement written after scheduling must supersede the stale send."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps(
            {
                "platform": "telegram",
                "chat_id": "42",
                "request_id": "old",
            }
        ),
        encoding="utf-8",
    )
    replacement_payload = json.dumps(
        {
            "platform": "telegram",
            "chat_id": "42",
            "request_id": "new",
        }
    )

    runner, adapter = make_restart_runner()
    send = AsyncMock(return_value=SendResult(success=True, message_id="must-not-send"))
    adapter.send = send
    real_ensure_future = gateway_run.asyncio.ensure_future

    def _schedule_after_replacement(coro):
        task = real_ensure_future(coro)
        notify_path.write_text(replacement_payload, encoding="utf-8")
        return task

    monkeypatch.setattr(gateway_run.asyncio, "ensure_future", _schedule_after_replacement)

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_not_awaited()
    assert notify_path.read_text(encoding="utf-8") == replacement_payload


@pytest.mark.asyncio
async def test_send_restart_notification_honors_retry_after(
    tmp_path, monkeypatch
):
    """Provider-requested retry delays take precedence over local backoff."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    send = AsyncMock(
        side_effect=[
            SendResult(
                success=False,
                error="rate limited",
                retryable=True,
                retry_after=5.0,
            ),
            SendResult(success=True, message_id="m-1"),
        ]
    )
    adapter.send = send
    sleep = AsyncMock()
    monkeypatch.setattr(gateway_run.asyncio, "sleep", sleep)

    await runner._send_restart_notification()

    sleep.assert_awaited_once_with(5.0)
    assert send.await_count == 2
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_retry_budget_is_bounded(
    tmp_path, monkeypatch
):
    """Persistent retryable failures consume the bounded budget, then clean up."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_RESTART_NOTIFICATION_RETRY_TIMEOUT_SECS",
        3.0,
    )
    monkeypatch.setattr(
        gateway_run,
        "_RESTART_NOTIFICATION_RETRY_MAX_DELAY_SECS",
        2.0,
    )

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    send = AsyncMock(
        return_value=SendResult(
            success=False,
            error="send_path_degraded",
            retryable=True,
        )
    )
    adapter.send = send
    clock = {"now": 0.0}
    sleeps = []

    async def _advance_clock(delay):
        sleeps.append(delay)
        clock["now"] += delay

    monkeypatch.setattr(
        gateway_run,
        "time",
        MagicMock(monotonic=lambda: clock["now"]),
    )
    monkeypatch.setattr(gateway_run.asyncio, "sleep", _advance_clock)

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    assert sleeps == [1.0, 2.0]
    # The second sleep reaches the deadline. Do not launch one more send after
    # the retry budget is already exhausted.
    assert send.await_count == 2
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_rechecks_deadline_inside_dispatch_task(
    tmp_path, monkeypatch
):
    """Event-loop congestion cannot start the provider after total expiry."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_RESTART_NOTIFICATION_RETRY_TIMEOUT_SECS",
        0.01,
    )
    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    entered = []

    async def _record_send(*_args, **_kwargs):
        entered.append(time.monotonic())
        return SendResult(success=True, message_id="late-dispatch")

    adapter.send = AsyncMock(side_effect=_record_send)
    real_resolve = gateway_run.resolve_delivery_transport

    def _congest_before_child_dispatch(platform, config, adapters):
        transport = real_resolve(platform, config, adapters)
        asyncio.get_running_loop().call_soon(time.sleep, 0.03)
        return transport

    monkeypatch.setattr(
        gateway_run,
        "resolve_delivery_transport",
        _congest_before_child_dispatch,
    )

    delivered_target = await runner._send_restart_notification()

    assert entered == []
    assert delivered_target is None
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_rechecks_deadline_after_marker_read(
    tmp_path, monkeypatch
):
    """A marker read that crosses expiry cannot begin a provider call."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_RESTART_NOTIFICATION_RETRY_TIMEOUT_SECS",
        1.0,
    )

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    fake_now = 0.0
    monkeypatch.setattr(gateway_run.time, "monotonic", lambda: fake_now)
    real_read_text = Path.read_text
    marker_reads = 0

    def _expire_during_dispatch_marker_read(path, *args, **kwargs):
        nonlocal fake_now, marker_reads
        value = real_read_text(path, *args, **kwargs)
        if path == notify_path:
            marker_reads += 1
            if marker_reads == 3:
                fake_now = 2.0
        return value

    monkeypatch.setattr(Path, "read_text", _expire_during_dispatch_marker_read)

    runner, adapter = make_restart_runner()
    send = AsyncMock(return_value=SendResult(success=True, message_id="too-late"))
    adapter.send = send

    delivered_target = await runner._send_restart_notification()

    assert marker_reads >= 3
    send.assert_not_awaited()
    assert delivered_target is None
    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_hard_deadline_survives_cancel_suppression(
    tmp_path, monkeypatch
):
    """A provider that swallows cancellation cannot hold the worker past expiry."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_RESTART_NOTIFICATION_RETRY_TIMEOUT_SECS",
        0.02,
    )

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    send_started = asyncio.Event()
    send_cancelled = asyncio.Event()
    release_send = asyncio.Event()
    send_settled = asyncio.Event()

    async def _cancellation_resistant_send(*_args, **_kwargs):
        send_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            send_cancelled.set()
            await release_send.wait()
            send_settled.set()
            return SendResult(success=True, message_id="late")

    adapter.send = AsyncMock(side_effect=_cancellation_resistant_send)
    worker = asyncio.create_task(runner._send_restart_notification())

    try:
        await asyncio.wait_for(send_started.wait(), timeout=1.0)
        done, _pending = await asyncio.wait({worker}, timeout=0.5)

        assert worker in done
        assert await worker is None
        assert send_cancelled.is_set()
        assert not notify_path.exists()
    finally:
        release_send.set()
        await asyncio.wait_for(send_settled.wait(), timeout=1.0)
        if not worker.done():
            await asyncio.wait_for(worker, timeout=1.0)


@pytest.mark.asyncio
async def test_send_restart_notification_cancellation_preserves_marker(
    tmp_path, monkeypatch
):
    """Shutdown during safe backoff leaves the marker for the next process."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(
        return_value=SendResult(
            success=False,
            error="send_path_degraded",
            retryable=True,
        )
    )
    sleeping = asyncio.Event()

    async def _block_in_backoff(_delay):
        sleeping.set()
        await asyncio.Future()

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _block_in_backoff)

    task = asyncio.create_task(runner._send_restart_notification())
    await asyncio.wait_for(sleeping.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_cancel_before_provider_preserves_marker(
    tmp_path, monkeypatch
):
    """Cancellation before the child task runs is a known-safe no-send outcome."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    marker_payload = json.dumps({"platform": "telegram", "chat_id": "42"})
    notify_path.write_text(marker_payload, encoding="utf-8")

    runner, adapter = make_restart_runner()
    send = AsyncMock(return_value=SendResult(success=True, message_id="never"))
    adapter.send = send

    async def _cancel_before_child_dispatch(_tasks, *, timeout):
        del timeout
        raise asyncio.CancelledError

    monkeypatch.setattr(gateway_run.asyncio, "wait", _cancel_before_child_dispatch)

    with pytest.raises(asyncio.CancelledError):
        await runner._send_restart_notification()

    send.assert_not_awaited()
    assert notify_path.read_text(encoding="utf-8") == marker_payload


@pytest.mark.asyncio
async def test_send_restart_notification_cancel_in_provider_consumes_marker(
    tmp_path, monkeypatch
):
    """Cancellation after provider entry remains ambiguous and consumes the marker."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    entered = asyncio.Event()

    async def _block_in_provider(*_args, **_kwargs):
        entered.set()
        await asyncio.Future()

    adapter.send = AsyncMock(side_effect=_block_in_provider)
    task = asyncio.create_task(runner._send_restart_notification())
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_shutdown_before_child_dispatch_preserves_marker(
    tmp_path, monkeypatch
):
    """Shutdown beginning after the parent check still prevents provider entry."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    marker_payload = json.dumps({"platform": "telegram", "chat_id": "42"})
    notify_path.write_text(marker_payload, encoding="utf-8")

    runner, adapter = make_restart_runner()
    send = AsyncMock(return_value=SendResult(success=True, message_id="never"))
    adapter.send = send
    real_ensure_future = asyncio.ensure_future

    def _begin_shutdown_after_scheduling(coro):
        task = real_ensure_future(coro)
        runner._running = False
        return task

    monkeypatch.setattr(
        gateway_run.asyncio,
        "ensure_future",
        _begin_shutdown_after_scheduling,
    )

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_not_awaited()
    assert notify_path.read_text(encoding="utf-8") == marker_payload


@pytest.mark.asyncio
async def test_send_restart_notification_cancel_after_retryable_result_preserves_marker(
    tmp_path, monkeypatch
):
    """A completed retryable refusal stays a known-safe no-send outcome."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    marker_payload = json.dumps({"platform": "telegram", "chat_id": "42"})
    notify_path.write_text(marker_payload, encoding="utf-8")

    runner, adapter = make_restart_runner()
    send = AsyncMock(
        return_value=SendResult(
            success=False,
            error="cold close",
            retryable=True,
        )
    )
    adapter.send = send

    async def _cancel_after_child_completed(tasks, *, timeout):
        del timeout
        (send_task,) = tasks
        await send_task
        assert send_task.done()
        raise asyncio.CancelledError

    monkeypatch.setattr(gateway_run.asyncio, "wait", _cancel_after_child_completed)

    with pytest.raises(asyncio.CancelledError):
        await runner._send_restart_notification()

    send.assert_awaited_once()
    assert notify_path.read_text(encoding="utf-8") == marker_payload


@pytest.mark.asyncio
async def test_send_restart_notification_shutdown_preserves_marker(
    tmp_path, monkeypatch
):
    """Once teardown starts, leave delivery to the replacement process."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    runner._running = False
    send = AsyncMock()
    adapter.send = send

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_not_awaited()
    assert notify_path.exists()


@pytest.mark.asyncio
async def test_send_restart_notification_preserves_replacement_marker(
    tmp_path, monkeypatch
):
    """A second /restart marker must not be consumed by the older delivery."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps(
            {
                "platform": "telegram",
                "chat_id": "42",
                "request_id": "old",
            }
        ),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()

    async def _send_while_marker_is_replaced(*_args, **_kwargs):
        notify_path.write_text(
            json.dumps(
                {
                    "platform": "telegram",
                    "chat_id": "42",
                    "request_id": "new",
                }
            ),
            encoding="utf-8",
        )
        return SendResult(success=True, message_id="sent")

    adapter.send = AsyncMock(side_effect=_send_while_marker_is_replaced)

    delivered_target = await runner._send_restart_notification()

    assert delivered_target == ("telegram", "42", None)
    assert json.loads(notify_path.read_text(encoding="utf-8"))["request_id"] == "new"


@pytest.mark.asyncio
async def test_send_restart_notification_does_not_unlink_async_replacement(
    tmp_path, monkeypatch
):
    """A worker-thread replacement between final read/unlink must survive."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    old_payload = json.dumps(
        {"platform": "telegram", "chat_id": "42", "request_id": "old"}
    )
    new_payload = json.dumps(
        {"platform": "telegram", "chat_id": "42", "request_id": "new"}
    )
    notify_path.write_text(old_payload, encoding="utf-8")

    runner, adapter = make_restart_runner()
    allow_replacement = threading.Event()
    replacement_done = threading.Event()

    def _replace_marker():
        allow_replacement.wait(timeout=2)
        notify_path.write_text(new_payload, encoding="utf-8")
        replacement_done.set()

    writer = threading.Thread(target=_replace_marker)
    writer.start()

    real_read_text = Path.read_text

    def _read_while_replacement_completes(path, *args, **kwargs):
        if (
            path == notify_path
            and getattr(runner, "_restart_notification_request_id", None) == "new"
            and not allow_replacement.is_set()
        ):
            allow_replacement.set()
            assert replacement_done.wait(timeout=2)
            return old_payload
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _read_while_replacement_completes)

    async def _send_while_replacement_is_queued(*_args, **_kwargs):
        # Mirrors _handle_restart_command: publish the generation before its
        # atomic write is dispatched to a worker thread.
        runner._restart_notification_request_id = "new"
        return SendResult(success=True, message_id="sent")

    adapter.send = AsyncMock(side_effect=_send_while_replacement_is_queued)

    try:
        delivered_target = await runner._send_restart_notification()
    finally:
        allow_replacement.set()
        writer.join(timeout=2)

    assert delivered_target == ("telegram", "42", None)
    assert replacement_done.is_set()
    assert notify_path.read_text(encoding="utf-8") == new_payload


@pytest.mark.asyncio
async def test_send_restart_notification_cleans_up_on_send_failure(
    tmp_path, monkeypatch
):
    """An ambiguous send exception consumes the marker to avoid duplicates."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    notify_path = tmp_path / ".restart_notify.json"
    notify_path.write_text(
        json.dumps({"platform": "telegram", "chat_id": "42"}),
        encoding="utf-8",
    )

    runner, adapter = make_restart_runner()
    send = AsyncMock(side_effect=RuntimeError("network down"))
    adapter.send = send

    delivered_target = await runner._send_restart_notification()

    assert delivered_target is None
    send.assert_awaited_once()
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


