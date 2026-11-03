"""Where the producer-turn lifecycle meets the drain's review completion, both run.

A producer-admitted turn (relay, webhook: ``admit_producer``) has no adapter task of its own: the
executor brackets it (``gateway/session_producer_lifecycle.py``: typing, ``on_processing_start``,
and ``on_processing_complete`` with the outcome once its reply was sent) and the drain completes
its gateway review ownership (``gateway/session_ingress_review.py``). The two meet at four sites:
``deliver_settled``'s two exits, the drain's unsettled path and ``_settle_cancelled``. At each the
producer hook runs first and the review hook in its ``finally``, as the adapter's own task orders
them (``_process_message_background``): an exception in the awaited adapter hook never leaves a
review token live (every later automatic review on the session would be refused
``live_turn_active``), and a review completion that raises never skips the adapter's outcome.

The harness of ``test_background_review_authority_ownership``: real GatewayRunner, adapter,
SessionAuthority, drain and ``execute_admission`` over SQLite; only ``_hmwa_prepare_turn`` is a
stand-in. A producer row is one ``mark_producer_admission`` marked, as ``_admitted`` marks a
delivery ``webhook_ingress._admit_native`` committed.
"""

import asyncio
import types

import pytest

import gateway.session_authority as authority_module
import gateway.session_ingress as ingress_module
import gateway.session_ingress_review as review_module
import gateway.session_settlement_recovery as recovery
from agent import review_admission
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session_contract import SessionRef
from gateway.session_producer_lifecycle import mark_producer_admission
from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_background_review_authority_ownership import (
    _CONTROL,
    _TimelineAdapter,
    _cancelled_queued_waiter,
    _install_turn,
    _owner,
    _submit_local,
)
from tests.gateway.test_shared_busy_receive_loop import _statuses
from tests.waiting import async_wait_until

_PRODUCER = types.SimpleNamespace(_producer_delivery=True)
_REPLIES = {
    "delivered": lambda text: "reply to " + text,
    "nothing_owed": lambda text: None,
}


class _HookAdapter(_TimelineAdapter):
    """Overrides the processing hooks, so the executor brackets a producer row of its chat and the
    bracket's outcome lands on the timeline beside the sends and the review spawns."""

    async def on_processing_start(self, event):
        self.timeline.append(("start", event.text))

    async def on_processing_complete(self, event, outcome):
        self.timeline.append(("complete", event.text, outcome))


def _completions(timeline, text):
    return [item[2] for item in timeline if item[:2] == ("complete", text)]


async def _unsettled(*_args, **_kwargs):
    raise RuntimeError("storage fence refused the commit")


def _exploding_drop(_authority, _admission_id):
    raise RuntimeError("producer drop exploded")


async def _exploding_close(_authority, _admission_id, _outcome):
    raise RuntimeError("producer close exploded")


async def _queued_producer_row(tmp_path, monkeypatch, timeline, turns, **turn):
    """A chat row queued with no waiter and marked as a producer's. Its adapter task was cancelled
    while parked on the unclaimed row and the FIFO paused before claiming it (the recipe of
    ``test_a_pause_before_the_claim_drops_the_mark_so_the_rerun_is_reviewed``), so the next drain
    runs it with no delivery waiter, the way a producer delivery runs. The cancelled adapter task's
    own lifecycle entries are cleared: the timeline keeps the executor's. Returns ``(authority,
    ref, row)``; the drain is scheduled by the caller."""
    adapter = _HookAdapter(timeline)
    runner, authority = await _owner(tmp_path, monkeypatch, adapter)
    source = adapter.build_source(
        chat_id="busy-chat", chat_type="dm", user_id="fixture-user"
    )
    _install_turn(runner, monkeypatch, timeline, turns, **turn)
    claim, refuse = asyncio.Event(), ["runtime_draining"]
    real_claim_next = authority._claim_next

    async def gated_claim_next(ref, live):
        await claim.wait()
        if refuse:
            raise RuntimeStoreError(refuse.pop())
        return await real_claim_next(ref, live)

    monkeypatch.setattr(authority, "_claim_next", gated_claim_next)
    await adapter.handle_message(
        MessageEvent(text="one", source=source, message_id="m1")
    )
    await async_wait_until(lambda: bool(authority.waiters), timeout=10)
    (session_key,) = list(adapter._session_tasks)
    await adapter.cancel_session_processing(
        session_key, release_guard=False, discard_pending=False
    )
    (sid,) = list(authority.sessions)
    live = authority.sessions[sid]
    claim.set()
    await async_wait_until(lambda: live.task.done(), timeout=10)
    assert _statuses(authority) == {"one": "queued"}
    assert not authority.waiters and not authority.native_waiters
    (row,) = list_session_admissions(authority.db, session_id=sid)
    mark_producer_admission(authority, _PRODUCER, row)
    timeline.clear()
    return authority, SessionRef(authority.profile_id, sid), row


async def _drained(authority, ref, turns, timeline, status="terminal"):
    authority._schedule(ref)
    await async_wait_until(lambda: _statuses(authority) == {"one": status}, timeout=20)
    sid, profile_key = turns["one"]
    await async_wait_until(
        lambda: review_admission.live_turn_block_reason(sid, None, profile_key) is None,
        timeout=10,
    )
    return sid


@pytest.mark.asyncio
async def test_a_producer_turn_the_drain_delivers_closes_its_bracket_then_completes_its_review(
    tmp_path, monkeypatch
):
    """``deliver_settled``'s delivered exit: the bracket opened when the row started, closes with
    SUCCESS once the reply went out, and the review completes after that, so the candidate
    spawns once, after both."""
    timeline, turns = [], {}
    authority, ref, _row = await _queued_producer_row(
        tmp_path, monkeypatch, timeline, turns
    )
    sid = await _drained(authority, ref, turns, timeline)
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)

    assert timeline[0] == ("start", "one")
    assert _completions(timeline, "one") == [ProcessingOutcome.SUCCESS]
    assert timeline.count(("spawn", sid)) == 1
    sent = timeline.index(("sent", "reply to one"))
    closed = timeline.index(("complete", "one", ProcessingOutcome.SUCCESS))
    assert sent < closed < timeline.index(("spawn", sid))
    assert authority.producer_turns == {}
    assert getattr(authority, "review_completions", {}) == {}


@pytest.mark.asyncio
async def test_a_producer_turn_owing_no_send_closes_its_bracket_on_the_commit(
    tmp_path, monkeypatch
):
    """``deliver_settled``'s nothing-owed exit: an empty reply is delivered by its commit; the
    bracket closes with SUCCESS and the review completes behind it."""
    timeline, turns = [], {}
    authority, ref, _row = await _queued_producer_row(
        tmp_path, monkeypatch, timeline, turns, reply=lambda text: None
    )
    sid = await _drained(authority, ref, turns, timeline)
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)

    assert not [item for item in timeline if item[0] == "sent"]
    assert _completions(timeline, "one") == [ProcessingOutcome.SUCCESS]
    assert timeline.count(("spawn", sid)) == 1
    closed = timeline.index(("complete", "one", ProcessingOutcome.SUCCESS))
    assert closed < timeline.index(("spawn", sid))
    assert authority.producer_turns == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", list(_REPLIES), ids=list(_REPLIES))
async def test_a_producer_close_that_raises_does_not_skip_the_drain_review_completion(
    tmp_path, monkeypatch, reply
):
    """Both exits of ``deliver_settled``: the awaited adapter hook raising releases the token
    with the drain's own delivery outcome all the same, so the candidate still spawns."""
    monkeypatch.setattr(ingress_module, "close_producer_turn", _exploding_close)
    timeline, turns = [], {}
    runner, authority = await _owner(tmp_path, monkeypatch)
    _install_turn(runner, monkeypatch, timeline, turns, reply=_REPLIES[reply])

    await _submit_local(authority, "hello")
    await async_wait_until(
        lambda: _statuses(authority) == {"hello": "terminal"}, timeout=20
    )
    sid, profile_key = turns["hello"]
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)

    assert review_admission.live_turn_block_reason(sid, None, profile_key) is None
    assert timeline.count(("spawn", sid)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", list(_REPLIES), ids=list(_REPLIES))
async def test_a_review_completion_that_raises_does_not_skip_the_producer_close(
    tmp_path, monkeypatch, reply
):
    """The producer hook runs first: a review completion that raises finds the bracket already
    closed with the delivery's outcome."""
    timeline, turns = [], {}
    authority, ref, _row = await _queued_producer_row(
        tmp_path, monkeypatch, timeline, turns, reply=_REPLIES[reply]
    )

    def exploding_completion(_authority, _admission_id, *, delivery_succeeded):
        timeline.append(("review", delivery_succeeded))
        raise RuntimeError("review completion exploded")

    monkeypatch.setattr(ingress_module, "complete_settled_review", exploding_completion)
    authority._schedule(ref)
    await async_wait_until(lambda: ("review", True) in timeline, timeout=20)

    assert _completions(timeline, "one") == [ProcessingOutcome.SUCCESS]
    assert timeline.index(
        ("complete", "one", ProcessingOutcome.SUCCESS)
    ) < timeline.index(("review", True))
    assert authority.producer_turns == {}


@pytest.mark.asyncio
async def test_an_unsettled_producer_turn_reports_failure_and_releases_its_review(
    tmp_path, monkeypatch, caplog
):
    """The drain's unsettled path: a turn fenced ``unknown`` tells nobody an answer, so the
    bracket reports FAILURE off the drain's path and the token is released without a review."""
    monkeypatch.setattr(recovery, "settle_with_retry", _unsettled)
    timeline, turns = [], {}
    authority, ref, _row = await _queued_producer_row(
        tmp_path, monkeypatch, timeline, turns
    )
    with caplog.at_level("INFO"):
        sid = await _drained(authority, ref, turns, timeline, status="unknown")
    await async_wait_until(
        lambda: _completions(timeline, "one") == [ProcessingOutcome.FAILURE], timeout=10
    )

    assert ("spawn", sid) not in timeline
    assert authority.producer_turns == {}
    assert getattr(authority, "review_completions", {}) == {}
    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_TURN_UNSETTLED in record.getMessage()
    ]
    assert sum("released without delivery" in line for line in lines) == 1, caplog.text


def _register_post_delivery(authority, monkeypatch, fired):
    """The real ``_hmwa_prepare_turn`` registers the turn's review-notice release on the delivery
    adapter under the turn's key (``run_turn_runner``); the stand-in registers a recorder there."""
    runner = authority.runner
    inner = runner._hmwa_prepare_turn

    async def prepare(
        event, source, session_entry, session_key, quick_key, run_generation
    ):
        runner._delivery_adapter_for(source).register_post_delivery_callback(
            session_key, lambda: fired.append(event.text), generation=run_generation
        )
        return await inner(
            event, source, session_entry, session_key, quick_key, run_generation
        )

    monkeypatch.setattr(runner, "_hmwa_prepare_turn", prepare)


@pytest.mark.asyncio
async def test_an_unsettled_producer_turn_runs_its_post_delivery_slot_once_then_releases_its_review(
    tmp_path, monkeypatch, caplog
):
    """The drain's unsettled path runs the turn's post-delivery slot (the release that flushes a
    held review notice, a deferred verdict) exactly once, as an adapter task's finally would, and
    the review token is released ``turn_unsettled`` with no spawn: nothing was delivered."""
    monkeypatch.setattr(recovery, "settle_with_retry", _unsettled)
    timeline, turns, fired = [], {}, []
    authority, ref, _row = await _queued_producer_row(
        tmp_path, monkeypatch, timeline, turns
    )
    _register_post_delivery(authority, monkeypatch, fired)
    with caplog.at_level("INFO"):
        sid = await _drained(authority, ref, turns, timeline, status="unknown")
    await async_wait_until(
        lambda: _completions(timeline, "one") == [ProcessingOutcome.FAILURE], timeout=10
    )

    assert fired == ["one"]
    (adapter,) = authority.runner.adapters.values()
    assert adapter._post_delivery_callbacks == {}
    assert getattr(authority, "post_delivery", {}) == {}
    assert ("spawn", sid) not in timeline
    assert authority.producer_turns == {}
    assert getattr(authority, "review_completions", {}) == {}
    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_TURN_UNSETTLED in record.getMessage()
    ]
    assert sum("released without delivery" in line for line in lines) == 1, caplog.text


@pytest.mark.asyncio
async def test_a_post_delivery_slot_that_raises_does_not_skip_the_unsettled_review_release(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(recovery, "settle_with_retry", _unsettled)

    async def exploding_fire(_authority, _admission_id):
        raise RuntimeError("post-delivery slot exploded")

    monkeypatch.setattr(ingress_module, "fire_post_delivery", exploding_fire)
    timeline, turns = [], {}
    authority, ref, _row = await _queued_producer_row(
        tmp_path, monkeypatch, timeline, turns
    )
    with caplog.at_level("INFO"):
        sid = await _drained(authority, ref, turns, timeline, status="unknown")
    await async_wait_until(
        lambda: _completions(timeline, "one") == [ProcessingOutcome.FAILURE], timeout=10
    )

    assert ("spawn", sid) not in timeline
    assert authority.producer_turns == {}
    assert getattr(authority, "review_completions", {}) == {}
    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_TURN_UNSETTLED in record.getMessage()
    ]
    assert sum("released without delivery" in line for line in lines) == 1, caplog.text


@pytest.mark.asyncio
async def test_a_producer_drop_that_raises_does_not_skip_the_unsettled_review_release(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(recovery, "settle_with_retry", _unsettled)
    # The unsettled drop runs in ``settle_and_deliver``, which binds the name at import.
    monkeypatch.setattr(recovery, "drop_producer_turn", _exploding_drop)
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
    assert getattr(authority, "review_completions", {}) == {}
    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_TURN_UNSETTLED in record.getMessage()
    ]
    assert sum("released without delivery" in line for line in lines) == 1, caplog.text


@pytest.mark.asyncio
async def test_an_unsettled_review_release_that_raises_does_not_skip_the_producer_drop(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(recovery, "settle_with_retry", _unsettled)

    def exploding_release(_authority, _admission_id):
        raise RuntimeError("review release exploded")

    monkeypatch.setattr(review_module, "release_unsettled_review", exploding_release)
    timeline, turns = [], {}
    authority, ref, _row = await _queued_producer_row(
        tmp_path, monkeypatch, timeline, turns
    )
    authority._schedule(ref)
    await async_wait_until(
        lambda: _statuses(authority) == {"one": "unknown"}, timeout=20
    )
    await async_wait_until(
        lambda: _completions(timeline, "one") == [ProcessingOutcome.FAILURE], timeout=10
    )
    assert authority.producer_turns == {}


@pytest.mark.asyncio
async def test_cancelling_a_queued_producer_row_drops_its_producer_and_waiter_marks(
    tmp_path, monkeypatch
):
    """``_settle_cancelled``: a producer row cancelled while queued never runs, so neither the
    executor's bracket mark nor the mark its cancelled waiter left may outlive the row."""
    timeline, turns = [], {}
    authority, ref, claim, _refuse = await _cancelled_queued_waiter(
        tmp_path, monkeypatch, timeline, turns
    )
    (row,) = list_session_admissions(authority.db, session_id=ref.session_id)
    mark_producer_admission(authority, _PRODUCER, row)
    assert row["admission_id"] in authority.review_completions
    assert authority.producer_turns == {row["admission_id"]: None}

    await authority.cancel_queued(_CONTROL, ref, row["admission_id"])
    claim.set()
    await async_wait_until(
        lambda: authority.sessions[ref.session_id].task.done(), timeout=10
    )

    assert _statuses(authority) == {"one": "terminal"}
    assert authority.producer_turns == {}
    assert authority.review_completions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("raising", ["producer", "review"])
async def test_a_cancel_hook_that_raises_does_not_skip_the_other_marks_drop(
    tmp_path, monkeypatch, raising
):
    """Either hook raising in ``_settle_cancelled`` surfaces to the cancel's caller, and the
    other hook's mark is gone all the same."""
    timeline, turns = [], {}
    authority, ref, claim, _refuse = await _cancelled_queued_waiter(
        tmp_path, monkeypatch, timeline, turns
    )
    (row,) = list_session_admissions(authority.db, session_id=ref.session_id)
    mark_producer_admission(authority, _PRODUCER, row)
    if raising == "producer":
        monkeypatch.setattr(authority_module, "drop_producer_turn", _exploding_drop)
    else:

        def exploding_mark_drop(_authority, _admission_id):
            raise RuntimeError("review mark drop exploded")

        monkeypatch.setattr(
            review_module, "drop_review_waiter_mark", exploding_mark_drop
        )

    with pytest.raises(RuntimeError, match="exploded"):
        await authority.cancel_queued(_CONTROL, ref, row["admission_id"])
    claim.set()
    await async_wait_until(
        lambda: authority.sessions[ref.session_id].task.done(), timeout=10
    )

    assert _statuses(authority) == {"one": "terminal"}
    if raising == "producer":
        assert authority.review_completions == {}
    else:
        assert authority.producer_turns == {}
