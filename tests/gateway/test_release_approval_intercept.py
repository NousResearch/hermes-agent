"""Gateway integration contracts for the exact release-approval reply."""
from __future__ import annotations

import asyncio
from types import MethodType

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import SessionSource
from hermes_cli.kanban_release_approval import ApprovalResult


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="armin-1",
        thread_id="thread-7",
    )


def test_enabled_exact_reply_is_consumed_with_authenticated_route_context(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "hermes_cli.config_effective.load_user_config_effective",
        lambda: {"kanban": {"release_approval": {
            "enabled": True,
            "promotion_argv": ["/opt/bin/promote wrapper", "--fixed"],
        }}},
    )

    def fake_process(**kwargs):
        captured.update(kwargs)
        return ApprovalResult(
            ok=True,
            classification="success",
            release_id="rel-1",
            dev_result="success",
            test_result="success",
            production_result="success",
            active_release_id="rel-1",
            previous_release_id="rel-0",
            rollback_available=True,
        )

    monkeypatch.setattr(
        "hermes_cli.kanban_release_approval.process_current_board_approval",
        fake_process,
    )
    event = MessageEvent(
        text="freigegeben",
        source=_source(),
        reply_to_message_id="visible-message-9",
    )

    handled, reply = asyncio.run(
        GatewayInboundMixin()._hm_release_approval_intercept(event, event.source)
    )

    assert handled is True
    assert reply is not None
    assert "Release-ID: rel-1" in reply
    assert "Vorgänger: rel-0" in reply
    assert "Rollback verfügbar: ja" in reply
    assert captured["promotion_argv"] == ["/opt/bin/promote wrapper", "--fixed"]
    context = captured["context"]
    assert (context.platform, context.chat_id, context.thread_id) == (
        "telegram", "chat-1", "thread-7",
    )
    assert (context.actor_id, context.reply_to_message_id) == (
        "armin-1", "visible-message-9",
    )


def test_non_exact_or_disabled_reply_is_not_consumed(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config_effective.load_user_config_effective",
        lambda: {"kanban": {"release_approval": {"enabled": False}}},
    )
    source = _source()
    mixin = GatewayInboundMixin()

    assert asyncio.run(mixin._hm_release_approval_intercept(
        MessageEvent(text=" freigegeben", source=source), source,
    )) == (False, None)
    assert asyncio.run(mixin._hm_release_approval_intercept(
        MessageEvent(text="freigegeben", source=source), source,
    )) == (False, None)


class _BusyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="reply-1")

    async def get_chat_info(self, chat_id):
        return {}


def test_exact_reply_bypasses_busy_base_guard_and_runner_consumes_after_auth(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config_effective.load_user_config_effective",
        lambda: {"kanban": {"release_approval": {
            "enabled": True,
            "promotion_argv": ["/opt/bin/promote"],
        }}},
    )
    processed = []

    def fake_process(**kwargs):
        processed.append(kwargs)
        return ApprovalResult(
            ok=True,
            classification="success",
            release_id="rel-busy",
            dev_result="success",
            test_result="success",
            production_result="success",
            active_release_id="rel-busy",
            previous_release_id="rel-old",
            rollback_available=True,
        )

    monkeypatch.setattr(
        "hermes_cli.kanban_release_approval.process_current_board_approval", fake_process,
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}
    )
    auth_calls = []

    async def admit(_self, event):
        auth_calls.append(event.text)
        return event, event.source, False

    runner._hm_admit_event = MethodType(admit, runner)
    adapter = _BusyAdapter()
    adapter.gateway_runner = runner  # type: ignore[assignment]
    adapter._message_handler = runner._handle_message
    source = _source()
    event = MessageEvent(
        text="freigegeben",
        message_type=MessageType.TEXT,
        source=source,
        reply_to_message_id="visible-message-9",
        allow_gateway_control=True,
    )
    session_key = adapter._event_session_key(event)
    adapter._active_sessions[session_key] = asyncio.Event()

    asyncio.run(adapter.handle_message(event))

    assert auth_calls == ["freigegeben"]
    assert len(processed) == 1
    assert adapter._pending_messages == {}
    assert adapter.sent and "Release-ID: rel-busy" in adapter.sent[0]
