"""A ``/steer`` sent while a clarify owns the turn releases it, so the steer can drain.

``/steer`` lands between tool-call iterations, i.e. only once the in-flight tool batch
returns. When the only in-flight tool is a blocking ``clarify`` waiting for the same user,
accepting the steer deadlocks the session until ``agent.clarify_timeout`` expires, and the
ack promises an arrival that cannot happen. The prose lane already releases a clarify it
cannot answer (``gateway/run_inbound.py::_hm_clarify_reply``); a recognized slash command is
*not* an answer, so it stays armed, but the steer it carries must still be able to land.
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools import clarify_gateway

SESSION_KEY = "key"
CLARIFY_ID = "clarify-145161"


class _Receiver:
    """The running turn's agent, as ``_busy_steer_command`` sees it."""

    _supports_active_turn_redirect = True

    def __init__(self):
        self.payload = None

    def steer(self, text):
        self.payload = text
        return True


@pytest.fixture(autouse=True)
def _no_leaked_clarify():
    """Drop the entry this module registers, whatever the test's outcome."""
    yield
    clarify_gateway.resolve_gateway_clarify(CLARIFY_ID, clarify_gateway.CANCELLED)
    clarify_gateway.wait_for_response(CLARIFY_ID, 0.01)
    clarify_gateway.clear_session(SESSION_KEY)


def _runner(tmp_path, monkeypatch, receiver=None):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = GatewayRunner(config=GatewayConfig())
    if receiver is not None:
        runner._session_state(SESSION_KEY).turn.agent = receiver
    return runner


def _steer_event():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user")
    return MessageEvent(
        text="/steer use the second option instead", source=source, message_id="message"
    )


@pytest.mark.asyncio
async def test_steer_releases_the_clarify_that_owns_the_turn(tmp_path, monkeypatch):
    """The deadlock: a steer cannot drain while the clarify tool is still waiting."""
    entry = clarify_gateway.register(CLARIFY_ID, SESSION_KEY, "Which option?", ["one", "two"])
    clarify_gateway.mark_awaiting_text(entry.clarify_id)  # tapped "Other (type answer)"
    receiver = _Receiver()
    runner = _runner(tmp_path, monkeypatch, receiver)
    event = _steer_event()

    reply = await runner._busy_steer_command(event, SESSION_KEY, event.source)

    assert receiver.payload, "the steer itself must still be accepted"
    assert reply and "second option" in reply
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.05) == clarify_gateway.CANCELLED


@pytest.mark.asyncio
async def test_steer_releases_a_choice_prompt_too(tmp_path, monkeypatch):
    """A native choice prompt holds the turn the same way as "Other (type answer)"."""
    entry = clarify_gateway.register(CLARIFY_ID, SESSION_KEY, "Which option?", ["one", "two"])
    receiver = _Receiver()
    runner = _runner(tmp_path, monkeypatch, receiver)
    event = _steer_event()

    await runner._busy_steer_command(event, SESSION_KEY, event.source)

    assert receiver.payload
    assert clarify_gateway.wait_for_response(entry.clarify_id, 0.05) == clarify_gateway.CANCELLED


def test_a_clarify_without_a_steer_stays_pending():
    """Control: nothing releases the prompt until a command that needs the turn arrives."""
    entry = clarify_gateway.register(CLARIFY_ID, SESSION_KEY, "Which option?", ["one", "two"])

    assert clarify_gateway.get_pending(entry.clarify_id) is not None


@pytest.mark.asyncio
async def test_steer_without_a_clarify_is_unchanged(tmp_path, monkeypatch):
    """Control: the release is a no-op when no prompt owns the turn."""
    receiver = _Receiver()
    runner = _runner(tmp_path, monkeypatch, receiver)
    event = _steer_event()

    reply = await runner._busy_steer_command(event, SESSION_KEY, event.source)

    assert receiver.payload
    assert reply and "second option" in reply
