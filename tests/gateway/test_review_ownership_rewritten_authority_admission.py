"""A rewritten message the session authority admits completes its review ownership with the
adapter send that delivers it.

``gateway.run_inbound._handle_message`` runs a ``pre_gateway_dispatch`` ``rewrite`` on a
``dataclasses.replace`` copy of the event the adapter dispatched. Under a session authority
that copy is what ``admit_message`` admits: the drain executes the committed row and
``admit_message`` adopts the drained turn's review-ownership completion onto the object it
waited with -- the copy (``gateway.session_ingress_review.adopt_review_completion``). The
adapter lifecycle that sends the reply reads the completion off the object IT dispatched
(``_take_review_delivery_callback``), so the authority-owned return of ``_handle_message``
must move the completion back onto the dispatched object the way the in-process lane does
(``gateway.run_turn_review.rehome_turn_completion``). Without that the token stays
``live_turn_active`` for the process lifetime (every later automatic review on the session is
refused) and the turn's candidate is neither spawned nor dropped, with no log line.

Real GatewayRunner, adapter, SessionAuthority, drain and ``execute_admission`` over SQLite
(the harness of ``test_background_review_authority_ownership``); the plugin hook is a stand-in
that rewrites every inbound text, and the no-rewrite arm is the control.
"""

import asyncio

import pytest

from agent import review_admission
from gateway.platforms.event import MessageEvent
from tests.gateway.test_background_review_authority_ownership import (
    _install_turn,
    _owner,
    _TimelineAdapter,
)
from tests.gateway.test_shared_busy_receive_loop import _statuses
from tests.waiting import async_wait_until

_COMPLETION = "_gateway_review_delivery_complete"


async def _rewrite(name, **kwargs):
    """A plugin that rewrites every inbound text (``{"action": "rewrite"}``)."""
    if name != "pre_gateway_dispatch":
        return []
    return [{"action": "rewrite", "text": kwargs["event"].text + " (rewritten)"}]


async def _allow(_name, **_kwargs):
    return []


async def _chat(tmp_path, monkeypatch, hook, timeline, turns, **turn_kwargs):
    """The authority harness with ``hook`` as the plugin hook."""
    adapter = _TimelineAdapter(timeline)
    runner, authority = await _owner(tmp_path, monkeypatch, adapter)
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", hook)
    source = adapter.build_source(
        chat_id="rewrite-chat", chat_type="dm", user_id="fixture-user"
    )
    _install_turn(runner, monkeypatch, timeline, turns, **turn_kwargs)
    return adapter, authority, source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hook", "admitted"),
    [(_rewrite, "hello (rewritten)"), (_allow, "hello")],
    ids=["rewrite", "no-rewrite"],
)
async def test_admitted_turn_completes_with_the_send_of_the_dispatched_event(
    tmp_path, monkeypatch, hook, admitted
):
    """The adapter task that dispatched the message sends the drained reply and completes the
    turn with that send: the token is released and the candidate spawns exactly once, after
    the reply went out -- whether or not a plugin rewrote the text on the way in."""
    timeline, turns = [], {}
    adapter, authority, source = await _chat(
        tmp_path, monkeypatch, hook, timeline, turns
    )
    dispatched = MessageEvent(text="hello", source=source, message_id="m1")

    await adapter.handle_message(dispatched)
    await async_wait_until(
        lambda: (
            _statuses(authority) == {admitted: "terminal"}
            and not adapter._session_tasks
            and not adapter._background_tasks
        ),
        timeout=20,
    )

    sid, profile_key = turns[admitted]
    assert timeline.count(("sent", "reply to " + admitted)) == 1
    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None, (
        "the admitted turn's review token outlived its delivered reply"
    )
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)
    assert timeline.count(("spawn", sid)) == 1
    assert timeline.index(("sent", "reply to " + admitted)) < timeline.index(
        ("spawn", sid)
    )
    assert getattr(authority, "review_completions", {}) == {}
    # Taken by the lifecycle that sent the reply: nothing stranded on the dispatched object.
    assert not hasattr(dispatched, _COMPLETION)


@pytest.mark.asyncio
async def test_a_waiter_cancelled_under_a_rewrite_releases_the_token_once(
    tmp_path, monkeypatch, caplog
):
    """``/stop`` cancels the adapter task parked in ``admit_message`` while the drain runs the
    rewritten turn: the wait raises out of ``admit_message`` and ``_handle_message`` before
    any completion was adopted onto the copy. The drain releases the token unconfirmed exactly
    once; the re-home on that unwind finds nothing and strands nothing on the dispatched
    object."""
    timeline, turns = [], {}
    entered, gate = asyncio.Event(), asyncio.Event()
    with caplog.at_level("INFO"):
        adapter, authority, source = await _chat(
            tmp_path,
            monkeypatch,
            _rewrite,
            timeline,
            turns,
            gate_text="one (rewritten)",
            gate=gate,
            entered=entered,
        )
        dispatched = MessageEvent(text="one", source=source, message_id="m1")
        await adapter.handle_message(dispatched)
        await asyncio.wait_for(entered.wait(), 10)
        sid, profile_key = turns["one (rewritten)"]
        assert (
            review_admission.live_turn_block_reason(sid, None, profile_key)
            == review_admission.REASON_LIVE_TURN
        )
        (session_key,) = list(adapter._session_tasks)
        await adapter.cancel_session_processing(
            session_key, release_guard=False, discard_pending=False
        )
        gate.set()
        await async_wait_until(
            lambda: (
                _statuses(authority) == {"one (rewritten)": "terminal"}
                and not authority.waiters
            ),
            timeout=20,
        )
    await asyncio.sleep(0)

    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None, (
        f"token outlived its turn: review_completions="
        f"{getattr(authority, 'review_completions', {})}"
    )
    assert ("spawn", sid) not in timeline
    assert getattr(authority, "review_completions", {}) == {}
    assert not hasattr(dispatched, _COMPLETION)
    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_DELIVERY_UNCONFIRMED in record.getMessage()
    ]
    assert sum("released without delivery" in line for line in lines) == 1, caplog.text
