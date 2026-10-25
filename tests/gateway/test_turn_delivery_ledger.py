"""Regression: a turn's deferred delivery work follows the TURN, not the chat session.

Covers the three defects the ledger closes:

* the authority drain lane (which sends a recovered / no-waiter reply) must FIRE the turn's
  post-delivery callbacks after the reply -- and must decide "an adapter will send this" from a
  carried ownership bit, never from ``authority.waiters`` (which the API, Webhook, Bot, Hosted
  and Managed lanes also fill);
* a dropped (superseded) turn's record is CLOSED, so post-turn hooks that still run afterwards
  cannot re-arm work on it, while release/cleanup callbacks still run;
* a ``pre_gateway_dispatch`` rewrite (``dataclasses.replace``) rebuilds the MessageEvent, so the
  record must be re-attached / keyed by admission id and still fire exactly once.
"""

import asyncio
import dataclasses

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_ingress import (
    bind_turn_delivery,
    deliver_settled,
    turn_deliveries,
)
from gateway.turn_delivery import (
    TurnDelivery,
    adopt_turn_delivery,
    current_turn_delivery,
    drop_turn_delivery,
    turn_delivery_for_event,
)


class _FakeAuthority:
    """Just the attributes ``deliver_settled`` touches (all of them, so an accidental new
    dependency fails loudly rather than silently creating state on an object)."""

    profile_id = "test-profile"

    def __init__(self):
        self.waiters = {}                 # shared by API/Webhook/Bot/Hosted/Managed
        self.native_waiters = set()       # native-only (admit_message)
        self.pending_deliveries = {}
        self.turn_deliveries = {}
        self.waiter_replies = {}
        self.unsettled_deliveries = {}
        self.held_turn_markers = {}
        self.marker_adopters = {}
        self.runner = None


def _event(text="hello"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user",
        ),
    )


@pytest.mark.asyncio
async def test_drain_lane_fires_the_goal_notice_after_the_reply(monkeypatch):
    """(i) The lane that sends the reply also fires the turn's deferred notice, in that order."""
    from gateway import session_ingress

    authority = _FakeAuthority()
    order = []

    async def _deliver_response(adapter, event, session_key, response):
        order.append("reply")

    monkeypatch.setattr(session_ingress, "deliver_response", _deliver_response)

    event = _event()
    record = turn_delivery_for_event(event)
    record.register(lambda: order.append("goal notice"))
    # A recovered no-waiter turn: non-native, reply owed and captured by the drain.
    authority.pending_deliveries["adm-1"] = (object(), event, "session-key", "the reply", None)
    turn_deliveries(authority)["adm-1"] = record

    await deliver_settled(authority, "adm-1")

    assert order == ["reply", "goal notice"]
    assert "adm-1" not in authority.pending_deliveries
    assert "adm-1" not in turn_deliveries(authority)


@pytest.mark.asyncio
async def test_is_api_turn_notice_follows_the_reply_handover():
    """(ii) An API turn has a waiter in ``authority.waiters`` but no adapter send and no
    delivery tuple.  Judging ownership by ``waiters`` left the record neither sent, fired, nor
    dropped (G1); firing it here -- before ``_resolve_settled_waiter`` hands the reply over --
    put the notice ahead of the answer.  It is deferred to that handover, then fired once."""
    from gateway.session_ingress import fire_settled_turn_delivery

    authority = _FakeAuthority()
    order = []
    waiter = asyncio.get_running_loop().create_future()
    # The API lane registers its reply handoff in ``waiters`` only -- never in native_waiters,
    # and (for is_api) execute_admission writes no delivery tuple at all.
    authority.waiters["adm-api"] = waiter
    record = TurnDelivery()
    record.register(lambda: order.append("goal notice"))
    turn_deliveries(authority)["adm-api"] = record

    await deliver_settled(authority, "adm-api")

    assert order == [], "the notice must not precede the reply handover"
    assert "adm-api" in turn_deliveries(authority), "record consumed before delivery"

    # ``_resolve_settled_waiter`` hands the reply to the API lane, then fires the record.
    waiter.set_result("the reply")
    order.append("reply handed over")
    await fire_settled_turn_delivery(authority, "adm-api")

    assert order == ["reply handed over", "goal notice"]
    assert "adm-api" not in turn_deliveries(authority), "record leaked in the registry"
    assert "adm-api" not in authority.pending_deliveries
    await fire_settled_turn_delivery(authority, "adm-api")  # repeat: no second fire
    assert order == ["reply handed over", "goal notice"]


@pytest.mark.asyncio
async def test_native_turn_is_left_for_the_adapter_lifecycle():
    """The carried bit (not ``waiters``) keeps the drain from double-sending a native reply."""
    authority = _FakeAuthority()
    fired = []
    event = _event()
    record = turn_delivery_for_event(event)
    record.register(lambda: fired.append("notice"))
    bind_turn_delivery(authority, "adm-native", event, adapter_sends=True)
    # A late native waiter also proves ``native_waiters`` is honoured on its own.
    authority.native_waiters.add("adm-native")

    await deliver_settled(authority, "adm-native")

    assert fired == [], "the adapter task, which sends the reply, must fire the record"
    assert turn_deliveries(authority)["adm-native"] is record
    assert "adm-native" not in authority.native_waiters


def test_dropped_record_is_closed_and_refuses_late_registration():
    """(iii) A superseded turn's late registration must not fire; release work still runs."""
    record = TurnDelivery()
    fired, released = [], []
    record.register(lambda: fired.append("early"))
    record.register(lambda: released.append("bg-release"), run_on_drop=True)

    record.drop()

    assert record.closed
    assert released == ["bg-release"], "release/cleanup work must run on drop"
    # The post-turn hooks still run after the stale discard and register again.
    record.register(lambda: fired.append("late"))
    assert fired == [], "a closed record must refuse reply-ordered work"


@pytest.mark.asyncio
async def test_drop_turn_delivery_closes_the_running_turn():
    record = TurnDelivery()
    token = current_turn_delivery.set(record)
    try:
        drop_turn_delivery(None)  # no event needed: the ContextVar is authoritative
    finally:
        current_turn_delivery.reset(token)

    assert record.closed
    record.register(lambda: None)
    assert record._callbacks == []


@pytest.mark.asyncio
async def test_rewritten_event_record_fires_exactly_once():
    """(iv) ``dataclasses.replace`` drops the record attribute; the admission id must survive it."""
    authority = _FakeAuthority()
    fired = []

    event = _event("original")
    record = turn_delivery_for_event(event)
    rewritten = dataclasses.replace(event, text="rewritten")
    # Without this the rebuilt event carries no record and the drain binds a fresh one.
    adopt_turn_delivery(event, rewritten)
    assert getattr(rewritten, "_turn_delivery", None) is record

    # execute_admission binds by ADMISSION ID from the (rewritten) event.
    bound = bind_turn_delivery(authority, "adm-rewrite", rewritten)
    assert bound is record

    # The post-turn hook registers inside the drain's context, on the bound record.
    token = current_turn_delivery.set(bound)
    try:
        bound.register(lambda: fired.append("notice"))
    finally:
        current_turn_delivery.reset(token)

    await deliver_settled(authority, "adm-rewrite")  # no reply owed -> fires
    await deliver_settled(authority, "adm-rewrite")  # repeat resolution: no second fire

    assert fired == ["notice"]


# ── G2 at the integration seam: the stale discard closes the turn, hooks stay release-only ──

class _FakeSessionEntry:
    session_id = "sid-stale"


class _FakeSessionStore:
    async def get_or_create_session(self, source, *, touch_activity=True):
        return _FakeSessionEntry()


def _make_runner():
    runner = object.__new__(GatewayRunner)
    # ``async_session_store`` is a read-only facade property: it re-wraps ``session_store``
    # unless the cached facade is already backed by it, so wire both to the same stub.
    store = _FakeSessionStore()
    store._store = store
    runner.session_store = store
    runner._async_session_store = store
    runner._delivery_adapter_for = lambda source: None
    return runner


def test_stale_discard_closes_the_record_but_runs_release_callbacks():
    runner = _make_runner()
    fired, released = [], []
    record = TurnDelivery()
    record.register(lambda: fired.append("goal notice"))
    record.register(lambda: released.append("bg-release"), run_on_drop=True)

    token = current_turn_delivery.set(record)
    try:
        runner._hmwa_discard_stale_result(object(), "session-key", 7, event=None)
    finally:
        current_turn_delivery.reset(token)

    assert record.closed
    assert released == ["bg-release"]
    record.register(lambda: fired.append("late notice"))
    assert fired == []


@pytest.mark.asyncio
async def test_stale_turn_releases_the_loop_without_judging_the_goal():
    runner = _make_runner()
    calls = []

    async def _loop(**kwargs):
        calls.append(("loop", kwargs))

    async def _goal(**kwargs):
        calls.append(("goal", kwargs))

    runner._post_turn_loop_completion = _loop
    runner._post_turn_goal_continuation = _goal
    runner._publish_local_control = lambda *a, **k: None

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user",
    )
    await GatewayRunner._run_post_turn_hooks(
        runner, agent_result={"final_response": "a discarded answer"}, source=source,
        is_internal=True, stale_turn=True, run_generation=5,
    )

    assert [c[0] for c in calls] == ["loop"]
    _, kwargs = calls[0]
    assert kwargs["final_response"] == ""
    assert kwargs["release_only"] is True
    assert kwargs["run_generation"] == 5


# ── The abandoned-turn paths: a record must never outlive the lane that would fire it ──


def test_abandoned_turn_closes_its_record_but_a_retained_reply_keeps_it():
    """A cancelled drain / lost worker leaves no sender: the record is closed, not stranded.

    ``retain_unsettled_delivery`` is the exception -- a Discard may still deliver that reply,
    and ``deliver_resolved`` closes the record when it does not.
    """
    from gateway.session_ingress import close_unretained_turn_delivery

    authority = _FakeAuthority()
    fired, released = [], []

    abandoned = TurnDelivery()
    abandoned.register(lambda: fired.append("goal notice"))
    abandoned.register(lambda: released.append("bg-release"), run_on_drop=True)
    turn_deliveries(authority)["adm-gone"] = abandoned

    close_unretained_turn_delivery(authority, "adm-gone")

    assert abandoned.closed
    assert fired == [], "no reply will be delivered, so no reply-ordered callback"
    assert released == ["bg-release"], "release/cleanup work still runs"
    assert "adm-gone" not in turn_deliveries(authority)

    retained = TurnDelivery()
    turn_deliveries(authority)["adm-unknown"] = retained
    authority.unsettled_deliveries["adm-unknown"] = (object(),)
    close_unretained_turn_delivery(authority, "adm-unknown")
    assert not retained.closed, "a reply retained for a Discard still owes its callbacks"
    assert turn_deliveries(authority)["adm-unknown"] is retained


@pytest.mark.asyncio
async def test_discard_that_owes_no_answer_closes_the_record_and_runs_releases(monkeypatch):
    """A Discard with no captured answer closes the record -- and must still release."""
    from gateway import session_ingress

    authority = _FakeAuthority()
    fired, released, sent = [], [], []

    async def _deliver_response(adapter, event, session_key, response):
        sent.append(response)

    monkeypatch.setattr(session_ingress, "deliver_response", _deliver_response)

    event = _event()
    record = turn_delivery_for_event(event)
    record.register(lambda: fired.append("goal notice"))
    record.register(lambda: released.append("bg-release"), run_on_drop=True)
    turn_deliveries(authority)["adm-discard"] = record
    authority.unsettled_deliveries["adm-discard"] = (object(), event, "key", "lost reply", None)

    await session_ingress.deliver_resolved(authority, "adm-discard", owed=False)

    assert sent == [], "a Discard owes no answer"
    assert fired == [], "no reply was shown, so no reply-ordered callback"
    assert released == ["bg-release"], "release/cleanup work must still run"
    assert record.closed
    assert "adm-discard" not in turn_deliveries(authority)
    assert "adm-discard" not in authority.unsettled_deliveries


@pytest.mark.asyncio
async def test_cancelled_sender_hands_the_reply_back_to_the_drain(monkeypatch):
    """A cancelled adapter task must not leave the drain believing an adapter still sends."""
    import hermes_state_runtime
    from gateway import session_ingress

    class _Receipt:
        admission_id = "adm-cancel"
        ref = object()
        status = "admitted"

    class _Authority(_FakeAuthority):
        db = object()

        async def admit_native(self, event):
            return _Receipt()

    authority = _Authority()
    monkeypatch.setattr(
        hermes_state_runtime, "get_session_admission",
        lambda db, admission_id=None: {"status": "started"},
    )

    event = _event()
    record = turn_delivery_for_event(event)
    record.register(lambda: None)
    # The drain already captured this reply for the (still live) adapter task.
    authority.waiter_replies["adm-cancel"] = ("delivery",)

    task = asyncio.create_task(session_ingress.admit_message(authority, event))
    for _ in range(50):
        await asyncio.sleep(0)
        if "adm-cancel" in authority.waiters:
            break
    assert "adm-cancel" in authority.native_waiters

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "adm-cancel" not in authority.native_waiters
    assert "adm-cancel" not in authority.waiters
    assert turn_deliveries(authority)["adm-cancel"].adapter_sends is False
    assert authority.pending_deliveries["adm-cancel"] == ("delivery",)


def test_rewrite_carries_the_turn_marker_handoff_flag():
    """``dataclasses.replace`` drops the marker handoff too, not only the record."""
    event = _event()
    event._turn_marker_handoff = True
    rewritten = dataclasses.replace(event, text="rewritten")
    assert not getattr(rewritten, "_turn_marker_handoff", False)

    adopt_turn_delivery(event, rewritten)

    assert rewritten._turn_marker_handoff is True


@pytest.mark.asyncio
async def test_inline_reply_fires_the_record_even_when_the_handler_raises():
    """The inline lane has no session lifecycle: a raised handler must not strand the record."""
    from gateway.platforms.base import BasePlatformAdapter

    fired = []

    class _InlineAdapter:
        name = "fake"

        async def _message_handler(self, event):
            turn_delivery_for_event(event).register(lambda: fired.append("cleanup"))
            raise RuntimeError("boom")

        async def _unwrap_ephemeral(self, response):  # pragma: no cover - not reached
            return response, 0

        async def _send_with_retry(self, **kwargs):  # pragma: no cover - not reached
            raise AssertionError("no reply to send")

    with pytest.raises(RuntimeError):
        await BasePlatformAdapter._dispatch_inline_reply(_InlineAdapter(), _event())

    assert fired == ["cleanup"]


def test_stale_discard_is_remembered_only_for_the_discarded_turn():
    """The post-turn hook asks the discard, not the (possibly since-invalidated) generation."""
    runner = _make_runner()
    record = TurnDelivery()
    token = current_turn_delivery.set(record)
    try:
        runner._hmwa_discard_stale_result(object(), "session-key", 7, event=None)
    finally:
        current_turn_delivery.reset(token)

    assert runner._consume_discarded_turn_run("session-key", 7) is True
    assert runner._consume_discarded_turn_run("session-key", 7) is False
    assert runner._consume_discarded_turn_run("session-key", 8) is False
