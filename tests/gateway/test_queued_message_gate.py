"""Queued (mid-turn) messages face ``pre_gateway_dispatch`` at the drain.

The hook's contract is "once per incoming MessageEvent, before auth and dispatch". A message
that arrives while a session's turn is in flight is diverted into the adapter's pending queue
BEFORE the direct path reaches the hook, and ``_run_agent_drain_pending`` later turns it into
the next user turn without ever running it. A plugin that drops, rewrites or gates inbound
messages (channel policy, addressee detection, content enrichment) therefore never sees the
mid-turn arrivals — they run as ungated follow-up turns.

The drain now runs the same helper the direct path uses, with the same action contract.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


SESSION_KEY = "agent:main:telegram:dm:123"


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="123", chat_id="123",
                         user_name="tester", chat_type="dm")


def _queued(text: str) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source(), message_id="q1")


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = MagicMock()
    runner._draining = False
    runner._peek_session_state = lambda session_key: None  # no /queue overflow chain
    return runner


async def _drain(runner, adapter):
    return await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, _source(), SESSION_KEY)


@pytest.mark.asyncio
async def test_queued_message_is_dropped_when_the_hook_skips(monkeypatch):
    seen = []

    def _hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen.append(kwargs["event"].text)
            return [{"action": "skip", "reason": "policy"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _hook)
    adapter = _StubAdapter()
    adapter._pending_messages[SESSION_KEY] = _queued("arrived mid-turn")

    pending_event, pending = await _drain(_runner(), adapter)

    assert seen == ["arrived mid-turn"]        # the queued event reached the hook
    assert pending_event is None and pending is None


@pytest.mark.asyncio
async def test_queued_message_text_follows_a_rewrite(monkeypatch):
    def _hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "[context] arrived mid-turn"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _hook)
    adapter = _StubAdapter()
    adapter._pending_messages[SESSION_KEY] = _queued("arrived mid-turn")

    pending_event, pending = await _drain(_runner(), adapter)

    assert pending_event is not None and pending_event.text == "[context] arrived mid-turn"
    assert pending == "[context] arrived mid-turn"   # what becomes the next user turn
