"""M6 T35-T37: Reconcile outbox + effect ledger — transactional, idempotent, recovery."""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_reconcile import (
    EffectLedgerEntry,
    OrchEvent,
    QueueItem,
    ReconcileError,
    apply_effect,
    claim_queue_item,
    enqueue_event,
    recover_expired_claim,
    sweep_dead_letters,
)

D = "a" * 64


def test_transactional_outbox():
    """T35: transition/event/fanout same commit, no event-loss window.

    Enqueue creates event + queue items atomically.
    Duplicate event_key or (event_id, consumer_kind) rejected.
    """
    events: list[OrchEvent] = []
    queue: list[QueueItem] = []

    event, items = enqueue_event(
        events, queue,
        event_id=1,
        consumer_kinds=["gateway", "observer"],
        board="board_0123456789abcdef",
        tenant="",
        orch="orch-1",
        event_kind="lifecycle_transition",
        target_key="orch-1",
        lifecycle_revision=1,
        cancel_epoch=0,
        payload_digest=D,
        commit_seq=100,
    )
    assert event.event_id == 1
    assert len(items) == 2
    assert items[0].consumer_kind == "gateway"
    assert items[1].consumer_kind == "observer"
    assert items[0].state == "pending"
    assert len(events) == 1
    assert len(queue) == 2

    # Duplicate event_key rejected
    with pytest.raises(ReconcileError, match="duplicate_event_key"):
        enqueue_event(
            events, queue,
            event_id=2,
            consumer_kinds=["gateway"],
            board="board_0123456789abcdef",
            tenant="",
            orch="orch-1",
            event_kind="lifecycle_transition",
            target_key="orch-1",
            lifecycle_revision=1,
            cancel_epoch=0,
            payload_digest=D,
            commit_seq=101,
        )

    # Duplicate (event_id, consumer_kind) rejected
    with pytest.raises(ReconcileError, match="duplicate_queue_item"):
        enqueue_event(
            events, queue,
            event_id=1,
            consumer_kinds=["gateway"],
            board="board_0123456789abcdef",
            tenant="",
            orch="orch-1",
            event_kind="different_event",
            target_key="different",
            lifecycle_revision=2,
            cancel_epoch=0,
            payload_digest="b" * 64,
            commit_seq=102,
        )


def test_effect_ledger_atomicity():
    """T36: effect ledger + queue done one commit, replay idempotent.

    Same digest → replay (no-op). Different digest → conflict.
    """
    events: list[OrchEvent] = []
    queue: list[QueueItem] = []
    effects: list[EffectLedgerEntry] = []

    _, items = enqueue_event(
        events, queue, event_id=1, consumer_kinds=["worker"],
        board="b", tenant="", orch="o", event_kind="test", target_key="t",
        lifecycle_revision=1, cancel_epoch=0, payload_digest=D, commit_seq=1,
    )
    item = items[0]
    epoch = claim_queue_item(item, owner="w1", token_hash="t1", ttl=30, now=100)

    # First apply — new effect
    apply_effect(item, effects, board="b", tenant="", effect_digest=D,
                 owner="w1", token_hash="t1", epoch=epoch)
    assert item.state == "done"
    assert item.done_effect_digest == D
    assert len(effects) == 1

    # Second apply with same digest — replay, idempotent
    item.state = "claimed"
    item.claim_owner = "w1"
    item.claim_token_hash = "t1"
    item.claim_epoch = epoch
    apply_effect(item, effects, board="b", tenant="", effect_digest=D,
                 owner="w1", token_hash="t1", epoch=epoch)
    assert item.state == "done"
    assert len(effects) == 1  # no duplicate

    # Different digest → conflict
    item.state = "claimed"
    item.claim_owner = "w1"
    item.claim_token_hash = "t1"
    item.claim_epoch = epoch
    with pytest.raises(ReconcileError, match="effect_digest_conflict"):
        apply_effect(item, effects, board="b", tenant="", effect_digest="c" * 64,
                     owner="w1", token_hash="t1", epoch=epoch)

    # Wrong owner
    item.state = "claimed"
    with pytest.raises(ReconcileError, match="effect_wrong_owner"):
        apply_effect(item, effects, board="b", tenant="", effect_digest=D,
                     owner="wrong", token_hash="t1", epoch=epoch)

    # Not claimed
    item.state = "pending"
    with pytest.raises(ReconcileError, match="effect_not_claimed"):
        apply_effect(item, effects, board="b", tenant="", effect_digest=D,
                     owner="w1", token_hash="t1", epoch=epoch)


def test_queue_recovery_fencing():
    """T37: expired vs active claim recovery/dead-letter.

    Active nonexpired claim cannot be swept to dead-letter.
    Expired claim returns to pending (if budget remains) or dead_letter.
    """
    events: list[OrchEvent] = []
    queue: list[QueueItem] = []
    effects: list[EffectLedgerEntry] = []

    _, items = enqueue_event(
        events, queue, event_id=1, consumer_kinds=["worker"],
        board="b", tenant="", orch="o", event_kind="test", target_key="t",
        lifecycle_revision=1, cancel_epoch=0, payload_digest=D, commit_seq=1,
    )
    item = items[0]
    item.max_attempts = 3

    # Claim with TTL
    epoch = claim_queue_item(item, owner="w1", token_hash="t1", ttl=30, now=100)
    assert item.claim_expires_at == 130

    # Before expiry — active claim cannot be swept
    state = recover_expired_claim(item, now=120, effects=effects)
    assert state == "claimed"

    # After expiry — returns to pending (attempts=1 < max=3)
    state = recover_expired_claim(item, now=200, effects=effects)
    assert state == "pending"
    assert item.claim_owner is None
    assert item.available_at == 200

    # Exhaust attempts
    item.available_at = 200
    claim_queue_item(item, owner="w2", token_hash="t2", ttl=10, now=200)
    recover_expired_claim(item, now=250, effects=effects)  # attempts=2

    item.available_at = 250
    claim_queue_item(item, owner="w3", token_hash="t3", ttl=10, now=250)
    assert item.attempts == 3

    # After expiry with max attempts → dead_letter
    state = recover_expired_claim(item, now=300, effects=effects)
    assert state == "dead_letter"
    assert item.last_error_code == "reconcile_attempts_exhausted"

    # Sweep dead_letters — already dead_letter, no change
    count = sweep_dead_letters(queue, effects)
    assert count == 0  # already dead_letter

    # New item at max_attempts in pending → swept to dead_letter
    _, items2 = enqueue_event(
        events, queue, event_id=2, consumer_kinds=["worker"],
        board="b", tenant="", orch="o", event_kind="test2", target_key="t2",
        lifecycle_revision=2, cancel_epoch=0, payload_digest="e" * 64, commit_seq=2,
    )
    item2 = items2[0]
    item2.max_attempts = 1
    item2.attempts = 1
    item2.state = "pending"
    count = sweep_dead_letters(queue, effects, now=300)
    assert count == 1
    assert item2.state == "dead_letter"

    # Active (non-expired) claim at max_attempts cannot be swept
    _, items3 = enqueue_event(
        events, queue, event_id=3, consumer_kinds=["worker"],
        board="b", tenant="", orch="o", event_kind="test3", target_key="t3",
        lifecycle_revision=3, cancel_epoch=0, payload_digest="f" * 64, commit_seq=3,
    )
    item3 = items3[0]
    item3.max_attempts = 1
    item3.attempts = 1
    item3.state = "claimed"
    item3.claim_expires_at = 999999  # far future
    count = sweep_dead_letters(queue, effects, now=300)
    assert count == 0  # active claim not swept
    assert item3.state == "claimed"

    # Expired claim at max_attempts CAN be swept
    item3.claim_expires_at = 200  # already expired
    count = sweep_dead_letters(queue, effects, now=300)
    assert count == 1
    assert item3.state == "dead_letter"
