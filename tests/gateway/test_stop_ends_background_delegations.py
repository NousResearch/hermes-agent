"""Invariant: ``/stop`` ends the session's BACKGROUND delegate_task children too — whether the session is
mid-turn (busy fast path) or idle after the dispatching turn already ended — instead of only the in-turn
children, leaving the detached unit to run to completion and wake the chat later. See #114456.

Real ``GatewayRunner`` + real ``BasePlatformAdapter`` subclass; the background unit is a real
``tools.async_delegation`` registry record whose ``interrupt_fn`` is the observable.
"""
from unittest.mock import MagicMock

import pytest

import tools.async_delegation as ad
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="x"), Platform.TELEGRAM)

    @property
    def name(self):
        return "telegram"

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


@pytest.fixture(autouse=True)
def _reset_async_delegation():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _seed_unit(session_key: str, parent_session_id: str = "") -> MagicMock:
    """A live background unit as ``_dispatch_background`` registers it (running, routed by session_key)."""
    fn = MagicMock()
    with ad._records_lock:
        ad._records["deleg_bg1"] = {"delegation_id": "deleg_bg1", "status": "running", "session_key": session_key,
                                    "origin_ui_session_id": "", "parent_session_id": parent_session_id, "interrupt_fn": fn}
    return fn


@pytest.mark.asyncio
@pytest.mark.parametrize("session_state", ["idle", "busy"])
async def test_stop_ends_background_delegations_of_the_session(monkeypatch, session_state):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "u1")
    adapter = _Adapter()
    runner = GatewayRunner(config=GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}))
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1", user_name="tester")
    key = adapter._event_session_key(MessageEvent(text="", message_type=MessageType.TEXT, source=source))
    stop_fn = _seed_unit(key)
    other_fn = MagicMock()
    with ad._records_lock:
        ad._records["deleg_other"] = {"delegation_id": "deleg_other", "status": "running", "session_key": "agent:main:telegram:dm:c2",
                                      "origin_ui_session_id": "", "parent_session_id": "", "interrupt_fn": other_fn}

    if session_state == "busy":
        await runner._interrupt_and_clear_session(key, source, interrupt_reason="stop", invalidation_reason="stop_command")
    else:
        reply = await runner._handle_stop_command(MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source))
        # The chat is told something WAS stopped, not "No active task to stop."
        assert "Stopped" in str(getattr(reply, "text", reply))

    stop_fn.assert_called_once()
    other_fn.assert_not_called()  # another chat's background work is untouched
