"""Invariants of the draft per-profile agent inbox (agent/agent_inbox.py)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.agent_inbox import (
    AgentInbox,
    InboxClaimError,
    InboxConflictError,
    Lane,
    Scheduler,
    Source,
    Status,
)

NOW = 1_000.0


@pytest.fixture()
def inbox(tmp_path):
    return AgentInbox(tmp_path / "inbox.db")


def test_enqueue_is_idempotent_and_conflicting_reuse_raises(inbox):
    first = inbox.enqueue("d1", Source.DM, {"message": "hi"}, now=NOW, author={"name": "ana"})
    again = inbox.enqueue("d1", Source.DM, {"message": "hi"}, now=NOW + 5, author={"name": "ana"})
    assert again == first and first.sequence == 1 and first.lane is Lane.AGENT
    with pytest.raises(InboxConflictError):
        inbox.enqueue("d1", Source.DM, {"message": "different"}, now=NOW)
    with pytest.raises(InboxConflictError):
        inbox.enqueue("d1", Source.DM, {"message": "hi"}, now=NOW, author={"name": "bob"})


def test_claim_prefers_lane_then_oldest_sequence(inbox):
    inbox.enqueue("bg", Source.PROCESS, {"n": 1}, now=NOW)
    inbox.enqueue("dm-old", Source.DM, {"n": 2}, now=NOW)
    inbox.enqueue("dm-new", Source.RELAY, {"n": 3}, now=NOW)
    inbox.enqueue("user", Source.USER, {"n": 4}, now=NOW)
    order = []
    while (batch := inbox.claim_next_turn("w", NOW)) is not None:
        order.append(batch.item_ids)
    assert order == [("user",), ("dm-old",), ("dm-new",), ("bg",)]


def test_coalesce_key_bursts_into_one_batch(inbox):
    for i in range(3):
        inbox.enqueue(f"p{i}", Source.PROCESS, {"i": i}, now=NOW, target_session="s1", coalesce_key="route-a")
    inbox.enqueue("other-route", Source.PROCESS, {"i": 9}, now=NOW, target_session="s1", coalesce_key="route-b")
    inbox.enqueue("other-session", Source.PROCESS, {"i": 8}, now=NOW, target_session="s2", coalesce_key="route-a")
    batch = inbox.claim_next_turn("w", NOW)
    assert batch.item_ids == ("p0", "p1", "p2")
    assert {item.claim_id for item in batch.items} == {batch.claim_id}
    assert inbox.claim_next_turn("w", NOW).item_ids == ("other-route",)
    assert inbox.claim_next_turn("w", NOW).item_ids == ("other-session",)


def test_concurrent_claims_never_share_an_item(inbox):
    for i in range(20):
        inbox.enqueue(f"i{i}", Source.KANBAN, {"i": i}, now=NOW)
    start = threading.Barrier(4)

    def _drain(owner):
        start.wait()
        got = []
        while (batch := inbox.claim_next_turn(owner, NOW)) is not None:
            got.extend(batch.item_ids)
        return got

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(_drain, ["a", "b", "c", "d"]))
    claimed = [item for got in results for item in got]
    assert sorted(claimed) == sorted(f"i{i}" for i in range(20))


def test_complete_is_immutable_and_duplicate_identical_is_noop(inbox):
    inbox.enqueue("d1", Source.DM, {"m": 1}, now=NOW)
    batch = inbox.claim_next_turn("w", NOW)
    (done,) = inbox.complete(batch, {"reply": "ok"}, now=NOW + 1)
    assert done.status is Status.SETTLED
    (again,) = inbox.complete(batch, {"reply": "ok"}, now=NOW + 9)
    assert again == done
    with pytest.raises(InboxConflictError):
        inbox.complete(batch, {"reply": "changed"}, now=NOW + 2)
    with pytest.raises(InboxConflictError):
        inbox.complete(batch, {"reply": "ok"}, status=Status.FAILED, now=NOW + 2)


def test_released_batch_requeues_and_old_claim_cannot_complete(inbox):
    inbox.enqueue("d1", Source.DM, {"m": 1}, now=NOW)
    stale = inbox.claim_next_turn("w", NOW)
    assert inbox.release(stale) == 1
    assert inbox.get("d1").status is Status.QUEUED and inbox.get("d1").attempts == 0
    fresh = inbox.claim_next_turn("w2", NOW)
    with pytest.raises(InboxClaimError):
        inbox.complete(stale, {"reply": "late"}, now=NOW)
    inbox.complete(fresh, {"reply": "ok"}, now=NOW)


def test_cancel_loses_to_a_claim_and_wins_when_queued(inbox):
    inbox.enqueue("claimed", Source.DM, {"m": 1}, now=NOW)
    inbox.claim_next_turn("w", NOW)
    assert inbox.cancel_if_queued("claimed", reason="runtime_offline", now=NOW).status is Status.CLAIMED
    inbox.enqueue("queued", Source.DM, {"m": 2}, now=NOW)
    cancelled = inbox.cancel_if_queued("queued", reason="runtime_offline", now=NOW)
    assert cancelled.status is Status.CANCELLED and cancelled.outcome["reason"] == "runtime_offline"
    assert inbox.claim_next_turn("w", NOW) is None


@pytest.mark.parametrize(
    ("source", "after_first", "after_second"),
    [
        (Source.DM, Status.UNKNOWN, None),
        (Source.CRON, Status.UNKNOWN, None),
        (Source.RELAY, Status.QUEUED, Status.FAILED),
        (Source.DELEGATION, Status.QUEUED, Status.FAILED),
        (Source.PROCESS, Status.QUEUED, Status.QUEUED),
        (Source.KANBAN, Status.QUEUED, Status.QUEUED),
    ],
)
def test_recover_applies_per_source_policy(inbox, source, after_first, after_second):
    inbox.enqueue("x", source, {"m": 1}, now=NOW)
    inbox.enqueue("live", source, {"m": 2}, now=NOW, coalesce_key="other")
    inbox.claim_next_turn("dead", NOW)
    inbox.claim_next_turn("alive", NOW)
    dead = lambda owner, _fingerprint: owner == "dead"  # noqa: E731
    (recovered,) = inbox.recover(dead, NOW + 1)
    assert recovered.item_id == "x" and recovered.status is after_first
    assert inbox.get("live").status is Status.CLAIMED
    if after_second is not None:
        inbox.claim_next_turn("dead", NOW + 2)
        inbox.recover(dead, NOW + 3)
        assert inbox.get("x").status is after_second


def test_expire_resolves_an_awaiting_caller(inbox):
    inbox.enqueue("d1", Source.RELAY, {"m": 1}, now=NOW, expires_at=NOW + 10)
    assert inbox.claim_next_turn("w", NOW + 11) is None  # expired items are never claimed
    result = {}
    waiter = threading.Thread(target=lambda: result.setdefault("item", inbox.await_outcome("d1", 5.0, interval=0.01)))
    waiter.start()
    time.sleep(0.05)
    assert inbox.expire(NOW + 11) == 1
    waiter.join(5.0)
    assert not waiter.is_alive()
    assert result["item"].status is Status.EXPIRED and result["item"].outcome["reason"] == "queued_expired"
    inbox.enqueue("pending", Source.DM, {"m": 2}, now=NOW)
    assert inbox.await_outcome("pending", 0.05, interval=0.01) is None


def test_stale_claims_are_surfaced(inbox):
    inbox.enqueue("old", Source.DM, {"m": 1}, now=NOW)
    inbox.enqueue("new", Source.DM, {"m": 2}, now=NOW)
    inbox.claim_next_turn("w", NOW)
    inbox.claim_next_turn("w", NOW + 500)
    assert [item.item_id for item in Scheduler(inbox).stale_claims(NOW + 600, 300)] == ["old"]


@pytest.mark.parametrize("running", list(Lane))
@pytest.mark.parametrize("incoming", list(Lane))
def test_should_preempt_only_interactive_over_background(running, incoming):
    expected = incoming is Lane.INTERACTIVE and running is Lane.BACKGROUND
    assert Scheduler.should_preempt(running, incoming) is expected
