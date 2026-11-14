"""Capability gate: skip synthesis when the adapter has no native send_voice.

A platform whose adapter inherits BasePlatformAdapter.send_voice can only
deliver the failure notice, never audio — skip synthesis rather than billing
a paid call whose only product is the failure the base path posts. The gate
checks capability (type(adapter).send_voice is the base fallback), not a
platform list, so future adapters inherit the right default. Google Chat and
relay hold real overrides and stay untouched; local keeps its existing guard.
"""

from __future__ import annotations

from unittest.mock import patch

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _BaseOnlyAdapter(BasePlatformAdapter):
    """Adapter with no native voice delivery (sms, dingtalk, irc, ntfy, raft...)."""

    async def connect(self, *, is_reconnect: bool = False) -> bool:  # pragma: no cover
        raise NotImplementedError

    def disconnect(self):  # pragma: no cover
        raise NotImplementedError

    def send(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def get_chat_info(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        return True


class _OverrideAdapter(_BaseOnlyAdapter):
    """Adapter with a real native voice path (telegram, discord, ...)."""

    async def send_voice(self, *args, **kwargs) -> SendResult:
        return SendResult(success=True)


def _runner() -> GatewayRunner:
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
    return runner


def _event(platform: Platform, chat_id: str = "123", message_type: MessageType = MessageType.TEXT) -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(
            platform=platform,
            chat_id=chat_id,
            user_id="u1",
            user_name="User",
        ),
        message_type=message_type,
        message_id="456",
    )


def _with_mode(runner: GatewayRunner, platform: Platform, adapter: BasePlatformAdapter) -> MessageEvent:
    runner._voice_mode[f"{platform.value}:123"] = "all"
    runner.adapters[platform] = adapter
    return _event(platform)


def test_base_only_adapter_gets_no_synthesis_even_with_mode_all():
    runner = _runner()
    event = _with_mode(runner, Platform.SMS, _BaseOnlyAdapter.__new__(_BaseOnlyAdapter))
    assert runner._should_send_voice_reply(event, "hello", []) is False


def test_override_adapter_still_sends_with_mode_all():
    runner = _runner()
    event = _with_mode(runner, Platform.TELEGRAM, _OverrideAdapter.__new__(_OverrideAdapter))
    assert runner._should_send_voice_reply(event, "hello", []) is True


def test_a2a_guard_unchanged():
    runner = _runner()
    event = _event(Platform("a2a"))
    assert runner._should_send_voice_reply(event, "hello", []) is False


def test_email_blocked_by_capability_not_platform_list():
    runner = _runner()
    event = _with_mode(runner, Platform.EMAIL, _BaseOnlyAdapter.__new__(_BaseOnlyAdapter))
    assert runner._should_send_voice_reply(event, "hello", []) is False
