"""Invariant (#138358): a turn still in pre-agent setup (session resolve, turn prepare,
``agent:start`` hooks) is ended by /stop, and by its own deadline, instead of holding the
session's slot and the durable FIFO behind it forever.

Real ``GatewayRunner`` + real ``BasePlatformAdapter`` subclass; only the setup step that hangs is
replaced, since any unbounded await there is the bug's trigger.
"""
import asyncio
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.platforms.base import BasePlatformAdapter, SendResult
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from hermes_agent.gateway.session import SessionSource


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


def _wedged_turn(monkeypatch, timeout: str, step: str = "_hmwa_resolve_session"):
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

    if step == "_hmwa_resolve_session":
        runner._hmwa_resolve_session = _hang
    else:  # setup runs up to a step just before the agent; only that step hangs
        async def _resolved(event, source):
            return source, SimpleNamespace(session_id="s1", session_key=key), key

        async def _prepared(*_args, **_kwargs):
            return runner._PreparedTurn(
                history=[], context_prompt="", message_text="hello", persist_user_message=None,
                persist_user_timestamp=None, persist_user_display_kind=None), []
        runner._hmwa_resolve_session, runner._hmwa_prepare_turn = _resolved, _prepared
        setattr(runner, step, _hang)
    state = runner._session_state(key)
    state.turn.agent = _AGENT_PENDING_SENTINEL  # the claim _handle_message makes before setup
    generation = runner._begin_session_run_generation(key)
    task = asyncio.ensure_future(runner._handle_message_with_agent(event, source, key, generation))
    return runner, source, key, task, entered


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["_hmwa_resolve_session", "_persist_prompt_pins"])
async def test_stop_ends_a_turn_wedged_in_setup(monkeypatch, step):
    runner, source, key, task, entered = _wedged_turn(monkeypatch, "600", step)  # deadline far away
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


@pytest.mark.asyncio
async def test_a_stalled_setup_is_a_failed_turn_so_goal_never_grades_the_resend_notice(monkeypatch):
    """The stalled notice is the turn's reply, not its answer: with ``/goal`` active the judge must
    not grade "Turn setup stalled … Please resend" and queue a continuation into a setup that may
    hang again. ``_turn_failed`` reads the owner's captured result, as for any turn that never ran."""
    from hermes_agent.gateway.session_results import execution_result
    captured: dict = {}
    token = execution_result.set(captured)  # the owner's capture around _handle_message
    try:
        _runner, _source, _key, task, entered = _wedged_turn(monkeypatch, "0.2")
        await asyncio.wait_for(entered.wait(), 5)
        reply = await asyncio.wait_for(task, 15)
        assert "Turn setup stalled" in reply
        assert GatewayRunner._turn_failed(), captured
    finally:
        execution_result.reset(token)
