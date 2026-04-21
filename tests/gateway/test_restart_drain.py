import asyncio
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.platforms.base import MessageEvent, MessageType
from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
async def test_restart_command_while_busy_requests_drain_without_interrupt(monkeypatch):
    # Ensure INVOCATION_ID is NOT set — systemd sets this in service mode,
    # which changes the restart call signature.
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
    )
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    result = await runner._handle_message(event)

    assert result == "⏳ Draining 1 active agent(s) before restart..."
    running_agent.interrupt.assert_not_called()
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


@pytest.mark.asyncio
async def test_drain_queue_mode_queues_follow_up_without_interrupt():
    runner, adapter = make_restart_runner()
    runner._draining = True
    runner._restart_requested = True
    runner._busy_input_mode = "queue"

    event = MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m2",
    )
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert session_key in adapter._pending_messages
    assert adapter._pending_messages[session_key].text == "follow up"
    assert not adapter._active_sessions[session_key].is_set()
    assert any("queued for the next turn" in message for message in adapter.sent)


@pytest.mark.asyncio
async def test_draining_rejects_new_session_messages():
    runner, _adapter = make_restart_runner()
    runner._draining = True
    runner._restart_requested = True

    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=make_restart_source("fresh"),
        message_id="m3",
    )

    result = await runner._handle_message(event)

    assert result == "⏳ Gateway is restarting and is not accepting new work right now."


def test_load_busy_input_mode_prefers_env_then_config_then_default(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INPUT_MODE", raising=False)

    assert gateway_run.GatewayRunner._load_busy_input_mode() == "interrupt"

    (tmp_path / "config.yaml").write_text(
        "display:\n  busy_input_mode: queue\n", encoding="utf-8"
    )
    assert gateway_run.GatewayRunner._load_busy_input_mode() == "queue"

    monkeypatch.setenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "interrupt")
    assert gateway_run.GatewayRunner._load_busy_input_mode() == "interrupt"


def test_load_restart_drain_timeout_prefers_env_then_config_then_default(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT", raising=False)

    assert (
        gateway_run.GatewayRunner._load_restart_drain_timeout()
        == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    )

    (tmp_path / "config.yaml").write_text(
        "agent:\n  restart_drain_timeout: 12\n", encoding="utf-8"
    )
    assert gateway_run.GatewayRunner._load_restart_drain_timeout() == 12.0

    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "7")
    assert gateway_run.GatewayRunner._load_restart_drain_timeout() == 7.0

    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "invalid")
    assert (
        gateway_run.GatewayRunner._load_restart_drain_timeout()
        == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    )
    assert "Invalid restart_drain_timeout" in caplog.text


@pytest.mark.asyncio
async def test_request_restart_is_idempotent():
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()

    assert runner.request_restart(detached=True, via_service=False) is True
    first_task = next(iter(runner._background_tasks))
    assert runner.request_restart(detached=True, via_service=False) is False

    await first_task

    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=True, service_restart=False
    )


@pytest.mark.asyncio
async def test_launch_detached_restart_command_uses_setsid(monkeypatch):
    runner, _adapter = make_restart_runner()
    popen_calls = []

    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["/usr/bin/hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 321)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/setsid" if cmd == "setsid" else None)

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    await runner._launch_detached_restart_command()

    assert len(popen_calls) == 1
    cmd, kwargs = popen_calls[0]
    assert cmd[:2] == ["/usr/bin/setsid", "bash"]
    assert "gateway restart" in cmd[-1]
    assert "kill -0 321" in cmd[-1]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL


# ── Shutdown notification tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_notification_sent_to_active_sessions():
    """Active sessions receive a notification when the gateway starts shutting down."""
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="999", chat_type="dm")
    session_key = f"agent:main:telegram:dm:999"
    runner._running_agents[session_key] = MagicMock()

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 1
    assert "shutting down" in adapter.sent[0]
    assert "interrupted" in adapter.sent[0]


@pytest.mark.asyncio
async def test_shutdown_notification_says_restarting_when_restart_requested():
    """When _restart_requested is True, the message says 'restarting' and mentions /retry."""
    runner, adapter = make_restart_runner()
    runner._restart_requested = True
    session_key = "agent:main:telegram:dm:999"
    runner._running_agents[session_key] = MagicMock()

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 1
    assert "restarting" in adapter.sent[0]
    assert "resume" in adapter.sent[0]


@pytest.mark.asyncio
async def test_shutdown_notification_deduplicates_per_chat():
    """Multiple sessions in the same chat only get one notification."""
    runner, adapter = make_restart_runner()
    # Two sessions (different users) in the same chat
    runner._running_agents["agent:main:telegram:group:chat1:u1"] = MagicMock()
    runner._running_agents["agent:main:telegram:group:chat1:u2"] = MagicMock()

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_shutdown_notification_skipped_when_no_active_agents():
    """No notification is sent when there are no active agents."""
    runner, adapter = make_restart_runner()

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_shutdown_notification_ignores_pending_sentinels():
    """Pending sentinels (not-yet-started agents) don't trigger notifications."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    runner, adapter = make_restart_runner()
    runner._running_agents["agent:main:telegram:dm:999"] = _AGENT_PENDING_SENTINEL

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_shutdown_notification_send_failure_does_not_block():
    """If sending a notification fails, the method still completes."""
    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock(side_effect=Exception("network error"))
    session_key = "agent:main:telegram:dm:999"
    runner._running_agents[session_key] = MagicMock()

    # Should not raise
    await runner._notify_active_sessions_of_shutdown()


def test_persist_restart_pending_events_writes_serialized_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()

    source = make_restart_source(chat_id="777", chat_type="thread")
    event = MessageEvent(
        text="resume me after restart",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-restart",
        channel_prompt="channel prompt",
    )
    session_key = build_session_key(source)
    adapter._pending_messages[session_key] = event

    written = runner._persist_restart_pending_events()

    assert written == 1
    checkpoint = tmp_path / ".restart_pending_events.json"
    assert checkpoint.exists()
    payload = checkpoint.read_text(encoding="utf-8")
    assert "resume me after restart" in payload
    assert "msg-restart" in payload


def test_restore_restart_pending_events_rehydrates_adapter_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    checkpoint = tmp_path / ".restart_pending_events.json"
    checkpoint.write_text(
        """
[
  {
    "session_key": "agent:main:telegram:thread:777:u1:777",
    "event": {
      "text": "resume me after restart",
      "message_type": "text",
      "message_id": "msg-restart",
      "channel_prompt": "channel prompt",
      "media_urls": [],
      "media_types": [],
      "reply_to_message_id": null,
      "reply_to_text": null,
      "internal": false,
      "source": {
        "platform": "telegram",
        "chat_id": "777",
        "chat_name": null,
        "chat_type": "thread",
        "user_id": "u1",
        "user_name": null,
        "thread_id": null,
        "chat_topic": null
      }
    }
  }
]
        """.strip(),
        encoding="utf-8",
    )

    restored = runner._restore_restart_pending_events()

    assert restored == 1
    session_key = "agent:main:telegram:thread:777:u1:777"
    assert session_key in adapter._pending_messages
    restored_event = adapter._pending_messages[session_key]
    assert restored_event.text == "resume me after restart"
    assert restored_event.message_id == "msg-restart"
    assert restored_event.channel_prompt == "channel prompt"
    assert not checkpoint.exists()


def test_preserve_pending_followup_during_restart_drain_requeues_event():
    runner, adapter = make_restart_runner()
    runner._draining = True
    runner._restart_requested = True
    runner._busy_input_mode = "queue"

    event = MessageEvent(
        text="original request",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m-drain",
        channel_prompt="prompt",
    )
    follow_up = MessageEvent(
        text="follow-up survives restart",
        message_type=MessageType.TEXT,
        source=event.source,
        message_id="m-follow",
        channel_prompt="prompt",
    )
    session_key = build_session_key(event.source)

    pending_event, pending_text = runner._preserve_pending_followup_during_drain(
        adapter=adapter,
        session_key=session_key,
        event=event,
        pending_event=follow_up,
        pending_text=follow_up.text,
    )

    assert pending_event is None
    assert pending_text is None
    assert session_key in adapter._pending_messages
    assert adapter._pending_messages[session_key].text == "follow-up survives restart"


def test_preserve_pending_followup_during_restart_drain_still_discards_without_queue_mode():
    runner, adapter = make_restart_runner()
    runner._draining = True
    runner._restart_requested = True
    runner._busy_input_mode = "interrupt"

    event = MessageEvent(
        text="original request",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m-drain",
    )
    follow_up = MessageEvent(
        text="drop me",
        message_type=MessageType.TEXT,
        source=event.source,
        message_id="m-follow",
    )
    session_key = build_session_key(event.source)

    pending_event, pending_text = runner._preserve_pending_followup_during_drain(
        adapter=adapter,
        session_key=session_key,
        event=event,
        pending_event=follow_up,
        pending_text=follow_up.text,
    )

    assert pending_event is None
    assert pending_text is None
    assert session_key not in adapter._pending_messages
