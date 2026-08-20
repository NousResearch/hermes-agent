"""ORCH V4 Reconciliation — outbox, effect ledger, queue recovery.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §11 (Event-driven reconciliation).
§13 T35-T37 test contracts.

Runtime-independent pure model. No live DB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ReconcileError(ValueError):
    """Reconciliation protocol violation."""
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class OrchEvent:
    """Immutable event from a state transition."""
    event_id: int
    board_instance_id: str
    tenant_scope: str
    orch_id: str
    event_kind: str
    target_key: str
    lifecycle_revision: int
    cancel_epoch: int
    payload_digest: str
    event_key: str
    commit_seq: int


@dataclass
class QueueItem:
    """Reconcile queue item."""
    event_id: int
    consumer_kind: str
    board_instance_id: str
    tenant_scope: str
    state: str = "pending"  # pending, claimed, done, dead_letter
    claim_owner: str | None = None
    claim_token_hash: str | None = None
    claim_epoch: int = 0
    claim_expires_at: int | None = None
    attempts: int = 0
    max_attempts: int = 5
    available_at: int = 0
    last_error_code: str | None = None
    done_effect_digest: str | None = None


@dataclass(frozen=True)
class EffectLedgerEntry:
    """Immutable effect ledger — exactly-once within SQLite."""
    board_instance_id: str
    tenant_scope: str
    event_id: int
    consumer_kind: str
    effect_digest: str


def enqueue_event(
    events: list[OrchEvent],
    queue: list[QueueItem],
    *,
    event_id: int,
    consumer_kinds: list[str],
    board: str,
    tenant: str,
    orch: str,
    event_kind: str,
    target_key: str,
    lifecycle_revision: int,
    cancel_epoch: int,
    payload_digest: str,
    commit_seq: int,
) -> tuple[OrchEvent, list[QueueItem]]:
    """§11: Transition/event/fanout same commit, no event-loss window.

    Creates one event + one queue item per consumer_kind atomically.
    """
    event_key = f"{board}:{tenant}:{orch}:{event_kind}:{target_key}:{lifecycle_revision}:{cancel_epoch}:{payload_digest}"

    # Check for duplicate event_key
    for e in events:
        if e.event_key == event_key:
            raise ReconcileError("duplicate_event_key")

    event = OrchEvent(
        event_id=event_id,
        board_instance_id=board,
        tenant_scope=tenant,
        orch_id=orch,
        event_kind=event_kind,
        target_key=target_key,
        lifecycle_revision=lifecycle_revision,
        cancel_epoch=cancel_epoch,
        payload_digest=payload_digest,
        event_key=event_key,
        commit_seq=commit_seq,
    )
    events.append(event)

    new_items = []
    for ck in consumer_kinds:
        # Check for duplicate (event_id, consumer_kind)
        for q in queue:
            if q.event_id == event_id and q.consumer_kind == ck:
                raise ReconcileError("duplicate_queue_item")
        item = QueueItem(
            event_id=event_id,
            consumer_kind=ck,
            board_instance_id=board,
            tenant_scope=tenant,
        )
        queue.append(item)
        new_items.append(item)

    return event, new_items


def claim_queue_item(
    item: QueueItem,
    *,
    owner: str,
    token_hash: str,
    ttl: int,
    now: int,
) -> int:
    """§11: Claim CAS with epoch fencing."""
    if item.state != "pending":
        raise ReconcileError("claim_not_pending")
    if item.available_at > now:
        raise ReconcileError("claim_not_available")
    if item.attempts >= item.max_attempts:
        raise ReconcileError("claim_max_attempts")
    item.state = "claimed"
    item.claim_owner = owner
    item.claim_token_hash = token_hash
    item.claim_epoch += 1
    item.claim_expires_at = now + ttl
    item.attempts += 1
    return item.claim_epoch


def apply_effect(
    item: QueueItem,
    effects: list[EffectLedgerEntry],
    *,
    board: str,
    tenant: str,
    effect_digest: str,
    owner: str,
    token_hash: str,
    epoch: int,
) -> None:
    """§11: Effect ledger + queue done one commit, replay idempotent.

    If effect already exists with same digest → replay (no-op, just finish queue).
    Different digest → corruption/conflict.
    """
    if item.state != "claimed":
        raise ReconcileError("effect_not_claimed")
    if item.claim_owner != owner:
        raise ReconcileError("effect_wrong_owner")
    if item.claim_token_hash != token_hash:
        raise ReconcileError("effect_wrong_token")
    if item.claim_epoch != epoch:
        raise ReconcileError("effect_wrong_epoch")

    # Check for existing effect with same (event_id, consumer_kind)
    for e in effects:
        if (e.event_id == item.event_id and e.consumer_kind == item.consumer_kind):
            if e.effect_digest == effect_digest:
                # Replay — idempotent, just finish queue
                item.state = "done"
                item.done_effect_digest = effect_digest
                item.claim_owner = None
                item.claim_token_hash = None
                item.claim_expires_at = None
                return
            else:
                raise ReconcileError("effect_digest_conflict")

    # New effect
    effects.append(EffectLedgerEntry(
        board_instance_id=board,
        tenant_scope=tenant,
        event_id=item.event_id,
        consumer_kind=item.consumer_kind,
        effect_digest=effect_digest,
    ))
    item.state = "done"
    item.done_effect_digest = effect_digest
    item.claim_owner = None
    item.claim_token_hash = None
    item.claim_expires_at = None


def recover_expired_claim(
    item: QueueItem,
    *,
    now: int,
    effects: list[EffectLedgerEntry],
) -> str:
    """§11: Expired claim recovery — return to pending or dead_letter.

    Active nonexpired claim can never be swept to dead-letter.
    """
    if item.state != "claimed":
        return item.state
    if item.claim_expires_at is not None and item.claim_expires_at > now:
        return item.state  # still active

    # Expired — check if effect exists
    has_effect = any(
        e.event_id == item.event_id and e.consumer_kind == item.consumer_kind
        for e in effects
    )

    if has_effect:
        item.state = "done"
    elif item.attempts < item.max_attempts:
        item.state = "pending"
        item.claim_owner = None
        item.claim_token_hash = None
        item.claim_expires_at = None
        item.available_at = now  # immediate retry
    else:
        item.state = "dead_letter"
        item.claim_owner = None
        item.claim_token_hash = None
        item.claim_expires_at = None
        if not item.last_error_code:
            item.last_error_code = "reconcile_attempts_exhausted"

    return item.state


def sweep_dead_letters(queue: list[QueueItem], effects: list[EffectLedgerEntry], *, now: int = 0) -> int:
    """§11: Sweep items at max_attempts with no effect to dead_letter.

    Active nonexpired claim can never be swept to dead-letter.
    """
    count = 0
    for item in queue:
        if item.attempts >= item.max_attempts:
            has_effect = any(
                e.event_id == item.event_id and e.consumer_kind == item.consumer_kind
                for e in effects
            )
            if not has_effect and item.state == "pending":
                item.state = "dead_letter"
                item.claim_owner = None
                item.claim_token_hash = None
                item.claim_expires_at = None
                if not item.last_error_code:
                    item.last_error_code = "reconcile_attempts_exhausted"
                count += 1
            elif not has_effect and item.state == "claimed":
                # Only sweep if claim has expired
                if item.claim_expires_at is not None and item.claim_expires_at <= now:
                    item.state = "dead_letter"
                    item.claim_owner = None
                    item.claim_token_hash = None
                    item.claim_expires_at = None
                    if not item.last_error_code:
                        item.last_error_code = "reconcile_attempts_exhausted"
                    count += 1
    return count


__all__ = [
    "ReconcileError", "OrchEvent", "QueueItem", "EffectLedgerEntry",
    "enqueue_event", "claim_queue_item", "apply_effect",
    "recover_expired_claim", "sweep_dead_letters",
]
