import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import i18n
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _message(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="msg-i18n",
    )


@pytest.fixture(autouse=True)
def _zh_language(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


@pytest.fixture(autouse=True)
def _suppress_busy_onboarding(monkeypatch):
    import agent.onboarding as onboarding

    monkeypatch.setattr(onboarding, "is_seen", lambda *_args, **_kwargs: True)


@pytest.mark.asyncio
async def test_busy_queue_ack_uses_gateway_locale():
    runner, adapter = make_restart_runner()
    event = _message("排队这条")
    session_key = build_session_key(event.source)
    runner._busy_input_mode = "queue"
    runner._busy_ack_ts = {}
    runner._running_agents[session_key] = MagicMock()
    runner._running_agents_ts[session_key] = time.time() - 120
    adapter._send_with_retry = AsyncMock()

    with patch("gateway.run.merge_pending_message_event"):
        handled = await runner._handle_active_session_busy_message(event, session_key)

    assert handled is True
    content = adapter._send_with_retry.call_args.kwargs["content"]
    assert "已排队到下一轮" in content
    assert "Queued for the next turn" not in content


def test_busy_redirect_messages_use_gateway_locale():
    from agent.onboarding import busy_input_hint_gateway

    ack = i18n.t("gateway.busy.redirect_ack", status="")
    hint = busy_input_hint_gateway("redirect")

    assert "已按你的修正重定向当前运行" in ack
    assert "Redirected current run" not in ack
    assert "已使用你的消息重定向当前运行" in hint
    assert "First-time tip" not in hint


@pytest.mark.asyncio
async def test_busy_first_touch_hint_uses_gateway_locale(monkeypatch):
    import agent.onboarding as onboarding

    monkeypatch.setattr(onboarding, "is_seen", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(onboarding, "mark_seen", lambda *_args, **_kwargs: True)

    runner, adapter = make_restart_runner()
    event = _message("排队这条")
    session_key = build_session_key(event.source)
    runner._busy_input_mode = "queue"
    runner._busy_ack_ts = {}
    runner._running_agents[session_key] = MagicMock()
    runner._running_agents_ts[session_key] = time.time() - 120
    adapter._send_with_retry = AsyncMock()

    with patch("gateway.run.merge_pending_message_event"):
        handled = await runner._handle_active_session_busy_message(event, session_key)

    assert handled is True
    content = adapter._send_with_retry.call_args.kwargs["content"]
    assert "首次提示" in content
    assert "此提示不会再次显示" in content
    assert "First-time tip" not in content


@pytest.mark.asyncio
async def test_busy_slash_command_reject_uses_gateway_locale():
    runner, _adapter = make_restart_runner()
    event = _message("/model gpt-5")
    session_key = build_session_key(event.source)
    runner._running_agents[session_key] = MagicMock()

    result = await runner._handle_message(event)

    assert "代理正在运行" in result
    assert "/stop" in result
    assert "Agent is running" not in result


@pytest.mark.asyncio
async def test_queue_and_steer_active_replies_use_gateway_locale():
    runner, adapter = make_restart_runner()
    session_key = build_session_key(make_restart_source())
    running_agent = MagicMock()
    running_agent.steer.return_value = True
    runner._running_agents[session_key] = running_agent

    queue_result = await runner._handle_message(_message("/queue 后面再处理"))
    assert "已排队到下一轮" in queue_result
    assert "Queued for the next turn" not in queue_result

    steer_result = await runner._handle_message(_message("/steer 改成更短"))
    assert "已加入当前运行" in steer_result
    assert "Steer queued" not in steer_result


@pytest.mark.asyncio
async def test_shutdown_notification_uses_gateway_locale():
    runner, adapter = make_restart_runner()
    source = make_restart_source(chat_id="789")
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._restart_requested = True
    runner._cache_session_source(session_key, source)

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.sent
    message = adapter.sent[-1]
    assert "网关正在重启" in message
    assert "当前任务将被中断" in message
    assert "Gateway restarting" not in message


@pytest.mark.asyncio
async def test_active_session_drain_messages_use_gateway_locale():
    runner, adapter = make_restart_runner()
    event = _message("重启后再处理")
    session_key = build_session_key(event.source)
    runner._draining = True
    runner._restart_requested = True
    adapter._send_with_retry = AsyncMock()

    runner._queue_during_drain_enabled = lambda: False
    handled = await runner._handle_active_session_busy_message(event, session_key)
    assert handled is True
    rejected = adapter._send_with_retry.call_args.kwargs["content"]
    assert "网关正在重启" in rejected
    assert "暂不接受另一轮任务" in rejected

    adapter._send_with_retry.reset_mock()
    runner._queue_during_drain_enabled = lambda: True
    handled = await runner._handle_active_session_busy_message(event, session_key)
    assert handled is True
    queued = adapter._send_with_retry.call_args.kwargs["content"]
    assert "网关正在重启" in queued
    assert "已排队" in queued


@pytest.mark.asyncio
async def test_restart_and_startup_notifications_use_gateway_locale(tmp_path, monkeypatch):
    import json
    import gateway.run as gateway_run
    from gateway.config import HomeChannel, Platform

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()

    (tmp_path / ".restart_notify.json").write_text(
        json.dumps({"platform": "telegram", "chat_id": "123456"}),
        encoding="utf-8",
    )
    await runner._send_restart_notification()
    assert "网关已成功重启" in adapter.sent[-1]
    assert "Gateway restarted successfully" not in adapter.sent[-1]

    adapter.sent.clear()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Home",
    )
    await runner._send_home_channel_startup_notifications()
    assert "网关已上线" in adapter.sent[-1]
    assert "Gateway online" not in adapter.sent[-1]
