"""A turn's answer is quoted/replied to the message it actually answers (#115001).

``busy_input_mode`` lets a second message arrive while a turn is still running, and both flavors
used to hand the FINAL send the anchor of the message that opened the turn:

* ``interrupt`` — the redirect succeeds, the running turn now answers B, but the turn's delivery
  target was bound at turn start (``run_turn.py``'s ``event_message_id``) and read back for the
  final send, so the answer to B went out quoted to A.
* ``queue`` — A's answer leaves through the queued-first-response lane (anchored to A, correct)
  and B's answer leaves through the OUTER final send, which is bracketed by the adapter against
  the event that OPENED the chain — so B's answer was quoted to A.

Both halves are the same defect: outbound delivery ownership was implicit (the outermost event of
a running or recursive turn chain) instead of message-owned. The fix sits at the shared anchor
chokepoint (``gateway/platforms/base.py::_reply_anchor_for_event``), fed by an explicit
``MessageEvent.reply_anchor_override``, so every platform adapter — Moti, Feishu, and the rest —
gets the right target with no per-platform code.

Covered here: both redirect entry points (``_resolve_busy_steer_or_redirect`` in interrupt mode and
the priority ``_hm_busy_interrupt`` path), both anchor SHAPES the shared helper serves (a threaded
platform, and a non-threaded one like Moti where the anchor is the message id), the refused-redirect
fallback, stale-generation protection, multiple redirects, and the queued chain's terminal reply.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult, _reply_anchor_for_event
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource

SESSION_KEY = "agent:main:feishu:dm:oc_1"
A_ID = "101"          # the message that opened the turn
B_ID = "102"          # the message that arrives while the turn runs
C_ID = "103"          # a third message, for the multiple-redirect case
ANSWER = "Tomorrow is Saturday, September 19th."

THREADED = (Platform.FEISHU, "topic-1")   # threaded platform: the anchor may be the replied-to id
PLAIN = (Platform.SIGNAL, None)           # non-threaded (Moti-shaped): the anchor is the message id


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """A bare GatewayRunner must not read or create the developer's real Hermes home/state.db."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    yield


class _Receiver:
    """A running agent that accepts or refuses redirects (``_supports_active_turn_redirect``)."""

    _supports_active_turn_redirect = True

    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.payload = None

    def redirect(self, text: str) -> bool:
        self.payload = text
        return self.accept


def _source(platform, thread_id=None) -> SessionSource:
    return SessionSource(platform=platform, chat_id="oc_1", thread_id=thread_id,
                         user_id="u1", user_name="user", chat_type="dm")


def _event(source, message_id: str, text: str = "text") -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id=message_id)


def _runner() -> GatewayRunner:
    return GatewayRunner(config=GatewayConfig())


def _adapter():
    """A real Telegram adapter (the base send contract) whose transport send is a mock."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="900"))
    return adapter


async def _final_send(adapter, event, text=ANSWER):
    """Drive the normal-lane final send and report the ``reply_to`` the adapter was handed."""
    await adapter._send_final_text(event, SESSION_KEY, text, {}, False, 0, lambda _r: None)
    return adapter._send_with_retry.await_args.kwargs["reply_to"]


async def _redirect(runner, receiver, event, source, route):
    """Fire the busy redirect through one of its two entry points; return True when it landed."""
    if route == "interrupt_mode":
        outcome = await GatewayRunner._resolve_busy_steer_or_redirect(
            runner, event, SESSION_KEY, "interrupt", receiver)
        return outcome.redirected
    await runner._hm_busy_interrupt(event, source, receiver, SESSION_KEY)
    return True


def _ledger_disabled(monkeypatch):
    """Keep the obligation write out of the picture: this pins routing, not the ledger."""
    from gateway import delivery_ledger as dl
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: False)


# ---------------------------------------------------------------------------
# Redirect: the running turn's final answer answers B, so it is quoted to B.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["interrupt_mode", "priority"])
@pytest.mark.parametrize("platform,thread_id", [THREADED, PLAIN])
async def test_a_successful_redirect_moves_the_final_reply_anchor_to_the_redirecting_message(
    route, platform, thread_id, monkeypatch,
):
    _ledger_disabled(monkeypatch)
    runner = _runner()
    source = _source(platform, thread_id)
    state = runner._session_state(SESSION_KEY)
    state.persistent.run_generation = 3           # the running turn's token
    receiver = _Receiver()

    b_event = _event(source, B_ID, "What day is tomorrow?")
    assert _reply_anchor_for_event(b_event) == B_ID
    assert await _redirect(runner, receiver, b_event, source, route) is True
    assert B_ID in (receiver.payload or "")

    # The turn that is still running holds the event that OPENED it for its final send.
    a_event = _event(source, A_ID, "What is the weather like in Shanghai?")
    assert _reply_anchor_for_event(a_event) == A_ID

    runner._apply_turn_delivery_target(a_event, SESSION_KEY, 3, {"final_response": ANSWER})

    assert _reply_anchor_for_event(a_event) == B_ID
    assert a_event.ledger_message_id == B_ID
    assert await _final_send(_adapter(), a_event) == B_ID


@pytest.mark.asyncio
async def test_a_refused_redirect_leaves_the_turn_anchored_to_its_opening_message(monkeypatch):
    """redirect() returning False must not retarget anything: A keeps A (B is queued separately)."""
    _ledger_disabled(monkeypatch)
    runner = _runner()
    source = _source(*PLAIN)
    runner._session_state(SESSION_KEY).persistent.run_generation = 3
    receiver = _Receiver(accept=False)

    b_event = _event(source, B_ID, "What day is tomorrow?")
    assert await _redirect(runner, receiver, b_event, source, "interrupt_mode") is False
    assert runner._peek_session_state(SESSION_KEY).turn.redirect_delivery_target is None

    a_event = _event(source, A_ID)
    runner._apply_turn_delivery_target(a_event, SESSION_KEY, 3, None)

    assert _reply_anchor_for_event(a_event) == A_ID
    assert a_event.ledger_message_id is None
    assert await _final_send(_adapter(), a_event) == A_ID


@pytest.mark.parametrize("invalidate", ["new_generation", "turn_released"])
@pytest.mark.asyncio
async def test_a_stale_redirect_cannot_re_anchor_a_later_turn(invalidate, monkeypatch):
    """``/new``, ``/stop`` or a replacement turn bumps the generation; the old target must be inert."""
    _ledger_disabled(monkeypatch)
    runner = _runner()
    source = _source(*PLAIN)
    state = runner._session_state(SESSION_KEY)
    state.persistent.run_generation = 3
    receiver = _Receiver()

    assert await _redirect(runner, receiver, _event(source, B_ID), source, "interrupt_mode") is True
    if invalidate == "new_generation":
        runner._invalidate_session_run_generation(SESSION_KEY, reason="/stop")
    else:
        state.turn.clear()      # the displaced turn unwinds and clears its turn-scoped state

    a_event = _event(source, A_ID)                  # ... the LATER turn's own event
    runner._apply_turn_delivery_target(a_event, SESSION_KEY, 4, None)

    assert _reply_anchor_for_event(a_event) == A_ID
    assert a_event.ledger_message_id is None
    assert await _final_send(_adapter(), a_event) == A_ID


@pytest.mark.asyncio
async def test_multiple_redirects_report_the_last_one(monkeypatch):
    """A → B → C with successful redirects: the final answer answers C, so it is quoted to C."""
    _ledger_disabled(monkeypatch)
    runner = _runner()
    source = _source(*PLAIN)
    runner._session_state(SESSION_KEY).persistent.run_generation = 3
    receiver = _Receiver()

    for message_id in (B_ID, C_ID):
        assert await _redirect(runner, receiver, _event(source, message_id), source,
                               "interrupt_mode") is True

    a_event = _event(source, A_ID)
    runner._apply_turn_delivery_target(a_event, SESSION_KEY, 3, {"final_response": ANSWER})

    assert _reply_anchor_for_event(a_event) == C_ID
    assert await _final_send(_adapter(), a_event) == C_ID


# ---------------------------------------------------------------------------
# Queue: the chain's terminal reply answers the LAST message, so it quotes that one.
# ---------------------------------------------------------------------------

def _chain_runner() -> GatewayRunner:
    runner = _runner()
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": ANSWER, "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=SESSION_KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="What is your gender?")
    runner._reply_anchor_for_event = MagicMock(side_effect=_reply_anchor_for_event)
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    return runner


async def _run_chain(runner, source, pending_event):
    turn_ctx = SimpleNamespace(
        source=source, session_id="sid", session_key=SESSION_KEY, run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None, context_prompt=None,
        result_holder=[None])
    return await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="What is your gender?",
        pending_event=pending_event, response="Tomorrow is Saturday.",
        result={"interrupted": True, "messages": []}, stream_task=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,thread_id", [THREADED, PLAIN])
async def test_a_queued_chains_terminal_reply_is_quoted_to_its_own_message(
    platform, thread_id, monkeypatch,
):
    """A's answer goes out on the queued lane under A (unchanged); B's answer is the OUTER final
    send, and it must be quoted to B — not to the event that opened the chain."""
    _ledger_disabled(monkeypatch)
    runner = _chain_runner()
    source = _source(platform, thread_id)
    pending_event = SimpleNamespace(source=source, message_id=B_ID, channel_prompt=None,
                                    message_type=None, internal=False, metadata={})

    merged = await _run_chain(runner, source, pending_event)

    # The chain reports the terminal message's anchor next to its ledger id...
    assert merged["queued_terminal_reply_anchor"] == B_ID
    assert merged["queued_terminal_inbound_id"] == B_ID
    # ...the recursive turn ran with it...
    assert runner._run_agent.await_args.kwargs["event_message_id"] == B_ID

    # ...and the outer final send (which carries B's answer) uses it.
    a_event = _event(source, A_ID, "What day is tomorrow?")
    runner._apply_turn_delivery_target(a_event, SESSION_KEY, 1, merged)

    assert _reply_anchor_for_event(a_event) == B_ID
    assert await _final_send(_adapter(), a_event) == B_ID


@pytest.mark.asyncio
async def test_a_deeper_chain_keeps_the_innermost_reply_anchor():
    """Nested follow-ups: the LAST message answered owns the quote target, so a deeper recursion's
    anchor is not overwritten on the way out (the keys are only filled while absent)."""
    runner = _chain_runner()
    source = _source(*PLAIN)
    pending_event = SimpleNamespace(source=source, message_id=B_ID, channel_prompt=None,
                                    message_type=None, internal=False, metadata={})
    runner._run_agent = AsyncMock(return_value={
        "final_response": ANSWER, "messages": [],
        "queued_terminal_inbound_id": C_ID, "queued_terminal_reply_anchor": C_ID})

    merged = await _run_chain(runner, source, pending_event)

    assert merged["queued_terminal_reply_anchor"] == C_ID
    assert merged["queued_terminal_inbound_id"] == C_ID


@pytest.mark.asyncio
async def test_a_turn_without_a_redirect_or_chain_keeps_its_own_anchor(monkeypatch):
    """The ordinary turn is untouched: anchor and ledger stay on the event that opened it."""
    _ledger_disabled(monkeypatch)
    runner = _runner()
    a_event = _event(_source(*PLAIN), A_ID)

    runner._apply_turn_delivery_target(a_event, SESSION_KEY, 1, {"final_response": ANSWER})

    assert _reply_anchor_for_event(a_event) == A_ID
    assert a_event.ledger_message_id is None
    assert await _final_send(_adapter(), a_event) == A_ID


def test_the_override_wins_over_the_per_platform_derivation():
    """``reply_anchor_override`` is honoured ahead of the platform-specific derivations (Feishu in a
    thread anchors on the replied-to message), and a normal event still derives exactly as before."""
    from gateway.platforms.base import _thread_metadata_for_event

    source = _source(Platform.FEISHU, "topic-1")
    event = MessageEvent(text="x", source=source, message_id=B_ID,
                         reply_to_message_id=A_ID, reply_anchor_override=C_ID)
    assert _reply_anchor_for_event(event) == C_ID
    del event.reply_anchor_override
    assert _reply_anchor_for_event(event) == A_ID
    # Thread routing is unchanged either way: the override moves the quote target, not the thread.
    assert _thread_metadata_for_event(event) == {"thread_id": "topic-1"}
