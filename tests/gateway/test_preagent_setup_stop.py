"""Invariant (#138358): a turn still in pre-agent setup (session resolve, turn prepare,
``agent:start`` hooks) is ended by /stop, and by its own deadline, instead of holding the
session's slot and the durable FIFO behind it forever.

Real ``GatewayRunner`` + real ``BasePlatformAdapter`` subclass; only the setup step that hangs is
replaced, since any unbounded await there is the bug's trigger.
"""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
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


def _wedged_turn(monkeypatch, timeout: str):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", timeout)
    adapter = _Adapter()
    runner = GatewayRunner(config=GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}))
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1", user_name="tester")
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source, message_id="m1")
    key = adapter._event_session_key(event)
    entered = asyncio.Event()

    async def _hang(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()  # an await nothing ever completes

    runner._hmwa_resolve_session = _hang
    state = runner._session_state(key)
    state.turn.agent = _AGENT_PENDING_SENTINEL  # the claim _handle_message makes before setup
    generation = runner._begin_session_run_generation(key)
    task = asyncio.ensure_future(runner._handle_message_with_agent(event, source, key, generation))
    return runner, source, key, task, entered


@pytest.mark.asyncio
async def test_stop_ends_a_turn_wedged_in_setup(monkeypatch):
    runner, source, key, task, entered = _wedged_turn(monkeypatch, "600")  # deadline far away
    await asyncio.wait_for(entered.wait(), 5)

    await runner._interrupt_and_clear_session(key, source, interrupt_reason="stop", invalidation_reason="stop_command")

    # Past the short grace the turn returns (no second reply: /stop's own answers the user), so
    # whatever awaits it, the session's FIFO drain included, settles and moves on.
    assert await asyncio.wait_for(task, 15) is None
    assert not task.cancelled()
    assert runner._peek_session_state(key).turn.setup_stop is None


@pytest.mark.asyncio
async def test_setup_past_its_deadline_ends_with_a_visible_notice(monkeypatch):
    runner, _source, key, task, entered = _wedged_turn(monkeypatch, "0.2")
    await asyncio.wait_for(entered.wait(), 5)

    reply = await asyncio.wait_for(task, 15)

    assert "Turn setup stalled" in reply and "resend" in reply  # never dropped silently
    assert runner._peek_session_state(key).turn.setup_stop is None
