"""A turn the session authority's drain executes releases its gateway review ownership.

``gateway.run_turn._handle_message_with_agent`` registers every gateway turn as live in
``agent.review_admission`` (``_GatewayReviewAdmission``) and parks the completion that releases
the token — and spawns the turn's automatic review once its reply is confirmed delivered — where
the turn's delivery owner reads it. A turn the durable FIFO drain executes
(``session_ingress.execute_admission``) has no adapter task of its own: its reply is sent by the
adapter task that admitted the first message, by ``dispatch_shared_busy``'s observer for a busy
follower, or by the drain itself (``deliver_settled``) for a local viewer, an API caller or any
turn with no live waiter. Each must release the token with its own delivery outcome — otherwise
every later automatic review on the session is refused ``live_turn_active`` for the process
lifetime — and spawn the drained turn's candidate exactly once, after its reply went out.

Real GatewayRunner, adapter, SessionAuthority, drain and ``execute_admission`` over SQLite with
the REAL ``_handle_message_with_agent`` prologue (resolve, ``begin``, carrier parking); only
``_hmwa_prepare_turn`` is a stand-in that answers the turn without a model and freezes a review
candidate on the turn's admission the way ``finalize_turn`` does.
"""

import asyncio
import contextlib
import types

import pytest

from agent import review_admission
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_contract import Principal, Submission
from gateway.session_local import create_local_session
from hermes_state import SessionDB
from tests.gateway.test_shared_busy_receive_loop import Adapter, _statuses
from tests.waiting import async_wait_until

_VIEWER = Principal(
    "viewer",
    "default",
    frozenset({"session:create", "session:submit", "session:read"}),
    "fixture",
)


class _TimelineAdapter(Adapter):
    """Records every send beside the review spawns, so delivery gating is observable."""

    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.timeline.append(("sent", content))
        return await super().send(chat_id, content, reply_to, metadata)


async def _owner(tmp_path, monkeypatch, adapter=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = db
    runner._session_db = db
    if adapter is not None:
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "fixture-user")
        adapter.gateway_runner = runner
        runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="owner", db=db
    )
    if adapter is not None:
        runner._wire_adapter_handlers(adapter)
    return runner, authority


def _install_turn(
    runner,
    monkeypatch,
    timeline,
    turns,
    *,
    reply=lambda text: "reply to " + text,
    gate_text=None,
    gate=None,
    entered=None,
    fail=False,
):
    """``_hmwa_prepare_turn`` stand-in: marks the turn active, binds a recording agent to the
    turn's gateway admission and freezes a review candidate on it, then answers without a
    model (a non-``_PreparedTurn`` return is the handler's own early-reply path)."""

    async def prepare(
        event, source, session_entry, session_key, quick_key, run_generation
    ):
        await runner._mark_durable_active_turn(event, quick_key)
        guard = None
        with contextlib.suppress(Exception):
            guard = runner._delivery_adapter_for(source)._active_sessions.get(quick_key)
        completion = getattr(
            event, "_gateway_review_delivery_complete", None
        ) or getattr(guard, "_gateway_review_delivery_complete", None)
        admission = completion.__self__
        session_id = str(session_entry.session_id)
        agent = types.SimpleNamespace(
            session_id=session_id,
            _spawn_background_review=lambda **candidate: timeline.append(
                ("spawn", candidate["_review_session_id"])
            ),
        )
        admission.bind_agent(agent)
        admission.capture_candidate(
            agent,
            [
                {"role": "user", "content": event.text},
                {"role": "assistant", "content": "ok"},
            ],
            review_memory=True,
            review_skills=False,
        )
        turns[event.text] = (session_id, admission.profile_key)
        if gate_text == event.text:
            entered.set()
            await gate.wait()
        if fail:
            raise RuntimeError("prepare exploded")
        return reply(event.text), None

    monkeypatch.setattr(runner, "_hmwa_prepare_turn", prepare)


def _last_index(timeline, item):
    return len(timeline) - 1 - timeline[::-1].index(item)


async def _submit_local(authority, text):
    ref = create_local_session(
        authority,
        _VIEWER,
        dict(request_id="viewer", source="gui", model="fixture", toolsets=[]),
    )
    await authority.submit(
        _VIEWER, Submission("viewer-" + text, ref, {"text": text}, "queue")
    )
    return ref


@pytest.mark.asyncio
async def test_busy_follower_run_by_the_drain_releases_its_token_after_its_reply(
    tmp_path, monkeypatch
):
    """The first message's adapter task and the FIFO follower's ``dispatch_shared_busy``
    observer each complete the turn they deliver: both tokens are released once both replies
    are out, and each candidate spawns once, never before its own reply."""
    timeline, turns = [], {}
    adapter = _TimelineAdapter(timeline)
    runner, authority = await _owner(tmp_path, monkeypatch, adapter)
    source = adapter.build_source(
        chat_id="busy-chat", chat_type="dm", user_id="fixture-user"
    )
    entered, gate = asyncio.Event(), asyncio.Event()
    _install_turn(
        runner,
        monkeypatch,
        timeline,
        turns,
        gate_text="one",
        gate=gate,
        entered=entered,
    )

    await adapter.handle_message(
        MessageEvent(text="one", source=source, message_id="m1")
    )
    await asyncio.wait_for(entered.wait(), 10)
    sid, profile_key = turns["one"]
    assert (
        review_admission.live_turn_block_reason(sid, None, profile_key)
        == review_admission.REASON_LIVE_TURN
    )
    # The receive loop hands the follow-up over once it is committed to the FIFO.
    await asyncio.wait_for(
        adapter.handle_message(
            MessageEvent(text="two", source=source, message_id="m2")
        ),
        10,
    )
    assert _statuses(authority) == {"one": "started", "two": "queued"}
    gate.set()
    await async_wait_until(
        lambda: (
            _statuses(authority) == {"one": "terminal", "two": "terminal"}
            and not adapter._background_tasks
        ),
        timeout=20,
    )
    assert turns["two"][0] == sid
    await async_wait_until(lambda: timeline.count(("spawn", sid)) == 2, timeout=10)

    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None
    assert timeline.index(("sent", "reply to one")) < timeline.index(("spawn", sid))
    assert timeline.index(("sent", "reply to two")) < _last_index(
        timeline, ("spawn", sid)
    )


@pytest.mark.asyncio
async def test_local_viewer_turn_releases_its_token_once_the_drain_delivered(
    tmp_path, monkeypatch
):
    """A viewer's turn has no live delivery waiter: the drain delivers the settled reply
    (``deliver_settled``) and completes the turn with that send."""
    timeline, turns = [], {}
    runner, authority = await _owner(tmp_path, monkeypatch)
    _install_turn(runner, monkeypatch, timeline, turns)

    await _submit_local(authority, "hello")
    await async_wait_until(
        lambda: _statuses(authority) == {"hello": "terminal"}, timeout=20
    )
    sid, profile_key = turns["hello"]
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)

    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None
    assert timeline.count(("spawn", sid)) == 1


@pytest.mark.asyncio
async def test_drained_turn_that_owes_no_send_completes_on_its_terminal_commit(
    tmp_path, monkeypatch
):
    """Nothing to send (an API caller or a finite CLI turn reads the committed result; an
    adapter lifecycle reads an empty reply the same way): the terminal commit is the delivery."""
    timeline, turns = [], {}
    runner, authority = await _owner(tmp_path, monkeypatch)
    _install_turn(runner, monkeypatch, timeline, turns, reply=lambda text: None)

    await _submit_local(authority, "quiet")
    await async_wait_until(
        lambda: _statuses(authority) == {"quiet": "terminal"}, timeout=20
    )
    sid, profile_key = turns["quiet"]
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)

    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None
    assert timeline.count(("spawn", sid)) == 1


@pytest.mark.asyncio
async def test_turn_the_drain_cannot_settle_releases_its_token_unconfirmed(
    tmp_path, monkeypatch, caplog
):
    """A turn fenced ``unknown`` (its terminal outcome never committed) tells nobody an
    answer: ownership is released without a review, under a body-free owner-tagged slug."""
    import gateway.session_settlement_recovery as recovery

    async def unsettled(*args, **kwargs):
        raise RuntimeError("storage fence refused the commit")

    monkeypatch.setattr(recovery, "settle_with_retry", unsettled)
    timeline, turns = [], {}
    runner, authority = await _owner(tmp_path, monkeypatch)
    _install_turn(runner, monkeypatch, timeline, turns)

    with caplog.at_level("INFO"):
        await _submit_local(authority, "lost")
        await async_wait_until(
            lambda: _statuses(authority) == {"lost": "unknown"}, timeout=20
        )
        sid, profile_key = turns["lost"]
        await async_wait_until(
            lambda: (
                review_admission.live_turn_block_reason(sid, None, profile_key) is None
            ),
            timeout=10,
        )
    await asyncio.sleep(0)
    assert ("spawn", sid) not in timeline
    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_TURN_UNSETTLED in record.getMessage()
    ]
    # The drain's release line once; ``finish`` then logs the dropped candidate under the same
    # cause. Owner-tagged and body-free, both.
    assert sum("released without delivery" in line for line in lines) == 1, caplog.text
    assert lines and all("owner=" in line and sid not in line for line in lines)


@pytest.mark.asyncio
async def test_turn_whose_handler_raised_releases_its_token_without_a_review(
    tmp_path, monkeypatch
):
    timeline, turns = [], {}
    runner, authority = await _owner(tmp_path, monkeypatch)
    _install_turn(runner, monkeypatch, timeline, turns, fail=True)

    await _submit_local(authority, "boom")
    await async_wait_until(
        lambda: _statuses(authority) == {"boom": "terminal"}, timeout=20
    )
    sid, profile_key = turns["boom"]
    await async_wait_until(
        lambda: review_admission.live_turn_block_reason(sid, None, profile_key) is None,
        timeout=10,
    )
    await asyncio.sleep(0)
    assert ("spawn", sid) not in timeline


async def _late_waiter(tmp_path, monkeypatch, timeline, turns):
    """Upstream's late-waiter recipe (``test_native_reply_delivered_once``): the admitting
    adapter task resumes only after the drain claimed and started the turn, so
    ``execute_admission`` computed ``native=False`` and the drain owes the reply itself until
    that waiter registers on the already-started row."""
    adapter = _TimelineAdapter(timeline)
    runner, authority = await _owner(tmp_path, monkeypatch, adapter)
    source = adapter.build_source(
        chat_id="busy-chat", chat_type="dm", user_id="fixture-user"
    )
    entered, gate = asyncio.Event(), asyncio.Event()
    _install_turn(
        runner, monkeypatch, timeline, turns, gate_text="one", gate=gate, entered=entered
    )
    real_admit_native = authority.admit_native

    async def late_admit_native(event):
        receipt = await real_admit_native(event)
        await entered.wait()
        return receipt

    monkeypatch.setattr(authority, "admit_native", late_admit_native)
    await adapter.handle_message(MessageEvent(text="one", source=source, message_id="m1"))
    await asyncio.wait_for(entered.wait(), 10)
    await async_wait_until(lambda: bool(authority.native_waiters), timeout=10)
    assert _statuses(authority) == {"one": "started"}
    return authority, adapter, gate


@pytest.mark.asyncio
async def test_late_waiter_kept_alive_completes_the_drained_turn_once_after_its_send(
    tmp_path, monkeypatch
):
    """``deliver_settled`` hands the reply to the late waiter; its adapter task sends it and
    completes the token once, after that send."""
    timeline, turns = [], {}
    authority, adapter, gate = await _late_waiter(tmp_path, monkeypatch, timeline, turns)
    sid, profile_key = turns["one"]
    gate.set()
    await async_wait_until(
        lambda: _statuses(authority) == {"one": "terminal"} and not adapter._session_tasks,
        timeout=20,
    )
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)

    assert timeline.count(("sent", "reply to one")) == 1
    assert timeline.count(("spawn", sid)) == 1
    assert timeline.index(("sent", "reply to one")) < timeline.index(("spawn", sid))
    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None
    assert getattr(authority, "review_completions", {}) == {}


@pytest.mark.asyncio
async def test_late_waiter_cancelled_while_the_drain_runs_releases_the_token_unconfirmed(
    tmp_path, monkeypatch
):
    """``/stop`` cancels the adapter task parked on a waiter that registered after the drain
    claimed the turn. The drain's handover must honour the 'waiter gone' mark although it
    computed ``native=False``: the token is released unconfirmed (nothing confirmed a delivery,
    so nothing is reviewed) instead of being left for a waiter nobody resumes — which would
    refuse every later automatic review on the session for the process lifetime."""
    timeline, turns = [], {}
    authority, adapter, gate = await _late_waiter(tmp_path, monkeypatch, timeline, turns)
    sid, profile_key = turns["one"]
    (session_key,) = list(adapter._session_tasks)
    await adapter.cancel_session_processing(
        session_key, release_guard=False, discard_pending=False
    )
    gate.set()
    # ``_resolve_settled_waiter`` pops the waiter after ``deliver_settled`` ran the drain's own
    # completion step, so an empty waiter map means every completion path has been through.
    await async_wait_until(
        lambda: _statuses(authority) == {"one": "terminal"} and not authority.waiters,
        timeout=20,
    )

    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None, (
        f"token outlived its turn: review_completions="
        f"{getattr(authority, 'review_completions', {})}"
    )
    assert ("spawn", sid) not in timeline
    assert getattr(authority, "review_completions", {}) == {}


class _Completion:
    """``_GatewayReviewAdmission.finish``'s shape: a bound method whose owner has a tag."""

    def __init__(self):
        self.calls = []
        self.__self__ = types.SimpleNamespace(owner_tag="owner")

    def __call__(self, *, delivery_succeeded, cause=None, outer=None):
        self.calls.append((delivery_succeeded, cause))


def _waiting_authority(admission_id):
    return types.SimpleNamespace(
        waiters={admission_id: object()}, native_waiters={admission_id}
    )


def _event_with(completion):
    event = types.SimpleNamespace()
    event._gateway_review_delivery_complete = completion
    return event


def test_waiter_gone_mark_outlives_a_handover_the_drain_computed_as_not_native():
    """The helper sequence of the test above: cancel during execution, handover with
    ``native=False``, ``deliver_settled`` finding the dead waiter still registered."""
    from gateway import session_ingress_review as sir

    authority, done = _waiting_authority("a"), _Completion()
    sir.forget_review_waiter(authority, "a")
    sir.hand_over_review_completion(authority, "a", _event_with(done), native=False)
    sir.hand_review_completion_to_waiter(authority, "a")
    sir.complete_settled_review(authority, "a", delivery_succeeded=True)

    assert done.calls == [(False, review_admission.REASON_DELIVERY_UNCONFIRMED)]
    assert authority.review_completions == {}


def test_a_waiter_registered_again_for_the_same_admission_drops_the_stale_gone_mark():
    """A redelivered message deduped onto the executing row registers a live waiter after an
    earlier one was cancelled: that waiter adopts and completes the turn with its send; the
    stale mark must not release the token unconfirmed under it."""
    from gateway import session_ingress_review as sir

    authority, done = _waiting_authority("a"), _Completion()
    sir.forget_review_waiter(authority, "a")
    sir.note_review_waiter(authority, "a")
    sir.hand_over_review_completion(authority, "a", _event_with(done), native=False)
    sir.hand_review_completion_to_waiter(authority, "a")
    sir.complete_settled_review(authority, "a", delivery_succeeded=True)
    assert done.calls == []  # left for the live waiter
    event = types.SimpleNamespace()
    sir.adopt_review_completion(authority, "a", event)
    assert authority.review_completions == {}
    sir.complete_review_delivery(sir.take_review_completion(event), delivery_succeeded=True)
    assert done.calls == [(True, None)]
