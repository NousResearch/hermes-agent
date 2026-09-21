from types import SimpleNamespace

import pytest

from gateway.credential_capture import (
    prepare_credential_capture,
    save_authorized_credential_capture,
)
from gateway.platforms.event import MessageEvent
from gateway.run_inbound import GatewayInboundMixin
from gateway.config import Platform
from gateway.session import SessionSource


class _Broker:
    def __init__(self):
        self.calls = []

    def save_login(self, **kwargs):
        self.calls.append(kwargs)
        meta = SimpleNamespace(id="bws:opaque-id", origin="https://example.com")
        return SimpleNamespace(meta=meta, action="created")


def test_explicit_chinese_capture_is_sanitized_before_save():
    identifier = "private-user@example.com"
    password = "unique-password-do-not-log"
    event = MessageEvent(
        "保存账号密码\n网站: https://example.com/login\n"
        f"邮箱: {identifier}\n密码: {password}\n标签: {password}\n继续: 登录并查看订单",
        raw_message={"content": f"account={identifier} password={password}"},
        metadata={"original_text": f"{identifier}:{password}"},
    )

    assert prepare_credential_capture(event)
    assert identifier not in event.text and password not in event.text
    assert identifier not in repr(event) and password not in repr(event)
    assert identifier not in repr(event.raw_message) and password not in repr(event.raw_message)
    assert identifier not in repr(event.metadata) and password not in repr(event.metadata)

    broker = _Broker()
    result = save_authorized_credential_capture(event, broker)
    assert result.handled and result.reply is None
    assert broker.calls == [{
        "label": "https://example.com/login",
        "origin": "https://example.com/login",
        "identifier_type": "email",
        "identifier": identifier,
        "password": password,
    }]
    assert event.text == "[登录凭据已保存到 bws:opaque-id，绑定 https://example.com]\n继续执行：登录并查看订单"
    assert event._credential_capture is None


@pytest.mark.asyncio
async def test_startup_restore_queue_preserves_capture_for_replay():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="owner")

    class Runner(GatewayInboundMixin):
        config = SimpleNamespace(multiplex_profiles=False)
        _startup_restore_in_progress = True

        def _queue_startup_restore_event(self, event):
            self.queued = event

    runner = Runner()
    event = MessageEvent(
        "保存账号密码\n网站: https://example.com\n账号: queued-user\n密码: queued-password",
        source=source,
    )
    assert await runner._handle_message(event) is None
    assert runner.queued is event
    assert event._credential_capture is not None
    assert event._credential_capture_deferred is True


def test_plain_chat_is_not_treated_as_authorized_capture():
    event = MessageEvent("这个账号的密码好像过期了")
    assert not prepare_credential_capture(event)
    assert event.text == "这个账号的密码好像过期了"


def test_polite_intent_and_instruction_repeat_are_sanitized():
    event = MessageEvent(
        "请保存账号密码\n网站: https://example.com\n账号: private-user\n"
        "密码: private-password\n继续: 用 private-user 和 private-password 登录"
    )
    assert prepare_credential_capture(event)
    assert "private-user" not in event.text
    assert "private-password" not in event.text


def test_incomplete_capture_returns_format_without_writing():
    event = MessageEvent("保存登录凭据\n网站: https://example.com\n账号: me")
    assert prepare_credential_capture(event)
    broker = _Broker()
    result = save_authorized_credential_capture(event, broker)
    assert result.handled and "缺少字段：密码" in result.reply
    assert broker.calls == []
    assert event._credential_capture is None


def test_save_error_scrubs_identifier_and_password():
    identifier = "secret-account"
    password = "secret-password"
    event = MessageEvent(
        f"/credential save\norigin: https://example.com\nusername: {identifier}\npassword: {password}"
    )
    assert prepare_credential_capture(event)

    class FailingBroker:
        def save_login(self, **kwargs):
            raise RuntimeError(f"failed for {identifier}: {password}")

    result = save_authorized_credential_capture(event, FailingBroker())
    assert result.handled
    assert identifier not in result.reply and password not in result.reply
    assert "[REDACTED]" in result.reply


@pytest.mark.asyncio
async def test_gateway_denial_discards_capture_without_writing():
    class Runner(GatewayInboundMixin):
        async def _hm_admit_event(self, event):
            return None

    event = MessageEvent(
        "保存账号密码\n网站: https://example.com\n账号: denied-user\n密码: denied-password"
    )
    assert await Runner()._handle_message(event) is None
    assert event._credential_capture is None
    assert "denied-user" not in event.text and "denied-password" not in event.text


@pytest.mark.asyncio
async def test_gateway_writes_only_after_admission(monkeypatch):
    broker = _Broker()

    class Runner(GatewayInboundMixin):
        config = SimpleNamespace(multiplex_profiles=False)

        async def _hm_admit_event(self, event):
            return event, SimpleNamespace(), False

    monkeypatch.setattr("agent.credential_broker.get_credential_broker", lambda: broker)
    event = MessageEvent(
        "保存账号密码\n网站: https://example.com\n账号: admitted-user\n密码: admitted-password"
    )
    reply = await Runner()._handle_message(event)
    assert reply == "✅ 登录凭据已保存到 bws:opaque-id，绑定 https://example.com。"
    assert broker.calls[0]["identifier"] == "admitted-user"
    assert event._credential_capture is None
