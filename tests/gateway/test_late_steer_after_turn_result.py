"""A message sent while a turn winds down gets a turn of its own, never the buffer of a finished agent.

The agent reads its steer buffer for the last time in the turn finalizer. The gateway keeps that
agent in the session's slot for a good while longer (streaming-TTS finalisation, the post-turn
drain, the stream flush) and handed every busy text to ``agent.steer()`` all that time. A text in
that window was acknowledged as steered and read by nobody: it waited on the idle agent for some
later turn, or was gone with the agent (#66704).

The tests drive the gateway's own busy entry points and post-turn drain on a real ``GatewayRunner``
with the agent's real steer buffer (``InterruptControlMixin``).
"""

import threading

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

KEY = "session"
LATE = "use Lisbon instead of Porto"


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


class _Agent(InterruptControlMixin):
    """The real steer buffer and drains; nothing else of an agent is read here."""

    def __init__(self):
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()


class _Turn:
    """A session slot running ``agent``, the way ``_run_agent_track_agent`` leaves it."""

    def __init__(self, monkeypatch):
        self.runner = GatewayRunner(config=GatewayConfig())
        self.adapter = _Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
        monkeypatch.setattr(self.runner, "_delivery_adapter_for", lambda source: self.adapter)
        self.source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user", chat_type="dm")
        self.agent = _Agent()
        self.ctx = TurnContext(source=self.source, session_key=KEY)
        self.ctx.agent_holder[0] = self.agent
        turn = self.runner._session_state(KEY).turn
        turn.agent, turn.ctx = self.agent, self.ctx

    def message(self, text, message_id="m-2"):
        return MessageEvent(text=text, source=self.source, message_id=message_id)

    def finalize(self):
        """The turn finalizer's last read of the steer buffer (``agent/turn_finalizer.py``)."""
        result = {"final_response": "done", "messages": []}
        leftover = self.agent._drain_leftover_steer()
        if leftover:
            result["pending_steer"] = leftover
        return result

    def hand_back(self, result):
        """The worker publishes the result before it returns (``TurnRunner._finish_stream_consumer``)."""
        self.ctx.result_holder[0] = result
        return result

    def queued(self):
        """Every event waiting for a turn of its own, in order."""
        head = self.adapter._pending_messages.get(KEY)
        return ([head] if head is not None else []) + list(self.runner._session_state(KEY).conversation.queued_events)


@pytest.fixture
def turn(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    return _Turn(monkeypatch)


@pytest.mark.asyncio
async def test_a_steer_accepted_after_the_finalizers_last_read_is_the_next_turn(turn):
    """Between the finalizer's drain and the result nothing gates the steer yet: the post-turn
    drain takes it out of the buffer and runs it as the follow-up, like any leftover steer."""
    result = turn.finalize()
    turn.runner._hm_busy_steer(turn.message(LATE), turn.agent, KEY)
    assert turn.agent._pending_steer.endswith(LATE), "accepted: the turn has not handed back yet"
    turn.hand_back(result)

    _, follow_up = await turn.runner._run_agent_drain_pending(result, turn.adapter, turn.source, KEY)

    assert follow_up is not None and follow_up.endswith(LATE)
    assert turn.agent._pending_steer is None, "no later turn of this agent reads it a second time"


@pytest.mark.asyncio
async def test_a_late_steer_follows_the_leftover_the_finalizer_handed_back(turn):
    turn.runner._hm_busy_steer(turn.message("with phone numbers", "m-1"), turn.agent, KEY)
    result = turn.finalize()
    turn.runner._hm_busy_steer(turn.message(LATE), turn.agent, KEY)
    turn.hand_back(result)

    _, follow_up = await turn.runner._run_agent_drain_pending(result, turn.adapter, turn.source, KEY)

    assert follow_up.index("with phone numbers") < follow_up.index(LATE)
    assert turn.agent._pending_steer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["priority", "busy", "command"])
async def test_a_message_after_the_turn_handed_back_is_queued_not_steered(turn, route):
    """Once the result is in, no drain of this turn can read a steer: every busy entry point
    queues the message as the next turn instead."""
    turn.hand_back(turn.finalize())
    event = turn.message(f"/steer {LATE}" if route == "command" else LATE)

    if route == "priority":
        turn.runner._hm_busy_steer(event, turn.agent, KEY)
    elif route == "busy":
        outcome = await turn.runner._resolve_busy_steer_or_redirect(event, KEY, "steer", turn.agent)
        assert (outcome.steered, outcome.effective_mode) == (False, "queue")
        turn.runner._queue_or_replace_pending_event(KEY, event)  # what the busy handler does with that outcome
    else:
        reply = await turn.runner._busy_steer_command(event, KEY, turn.source)
        assert "queued" in reply

    assert turn.agent._pending_steer is None, "nothing enters the buffer of a turn that has ended"
    (queued,) = turn.queued()
    assert queued.text == LATE and queued.message_id == "m-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["priority", "busy", "command"])
async def test_a_message_while_the_turn_still_runs_is_steered_as_before(turn, route):
    event = turn.message(f"/steer {LATE}" if route == "command" else LATE)

    if route == "priority":
        turn.runner._hm_busy_steer(event, turn.agent, KEY)
    elif route == "busy":
        outcome = await turn.runner._resolve_busy_steer_or_redirect(event, KEY, "steer", turn.agent)
        assert outcome.steered is True
    else:
        await turn.runner._busy_steer_command(event, KEY, turn.source)

    assert turn.agent._pending_steer.endswith(LATE)
    assert turn.queued() == []


@pytest.mark.asyncio
async def test_the_follow_up_turn_in_the_same_slot_takes_steers_again(turn):
    """A queued follow-up runs the same agent under a fresh turn context: it is steerable."""
    turn.hand_back(turn.finalize())
    follow_up_ctx = TurnContext(source=turn.source, session_key=KEY)
    follow_up_ctx.agent_holder[0] = turn.agent
    turn.runner._session_state(KEY).turn.ctx = follow_up_ctx

    turn.runner._hm_busy_steer(turn.message(LATE), turn.agent, KEY)

    assert turn.agent._pending_steer.endswith(LATE)
    assert turn.queued() == []
