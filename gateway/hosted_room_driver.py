"""Durable execution state for a same-gateway hosted room driver.

Owns only the driver lease and task state machine: no model calls, no sessions, no dependency on the
hosted-room event log. Each function validates its inputs, then runs the same-named verb of the store handle's
``hosted_room_driver`` repository on the file ``placement`` names. Callers supply the handle and clock so recovery
and fencing are testable without process-global state.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, Callable

from gateway.hosted_rooms_common import bounded_int, canonical_json, compact_json, identifier, text
from persistence.gateway.hosted_rooms import (
    SHOWN_TASK_STATUSES, TASK_STATUSES, DriverLease, DriverValidationError, HostedPlacement, HostedTaskIdentity,
    RoomChangeRows, TaskAttempt, TaskStatus, TerminalStatus)

Clock = Callable[[], float]

MAX_IDENTIFIER_CHARS = 128
MAX_PROMPT_BYTES = 128 * 1024
MAX_RESULT_JSON_BYTES = 256 * 1024
TERMINAL_TASK_RETENTION_SECONDS = 30 * 24 * 60 * 60
MAX_RETAINED_TERMINAL_TASKS = 2048
MAX_TASK_PRUNE_BATCH = 1000

_TASK_PAYLOAD_REQUIRED_FIELDS = frozenset({"target_profile", "prompt", "source_event_seq"})
_TASK_PAYLOAD_OPTIONAL_FIELDS = frozenset({"target_member_id", "input_context", "attachments", "lead_routing"})


_identifier = partial(identifier, error=DriverValidationError, max_chars=MAX_IDENTIFIER_CHARS)
_bounded_int = partial(bounded_int, error=DriverValidationError)
_canonical_json = partial(
    canonical_json, error=DriverValidationError, label="result", max_bytes=MAX_RESULT_JSON_BYTES, ensure_ascii=True)


def _finite(compute: Callable[[], Any], message: str, *, positive: bool = False) -> float:
    try:
        value = float(compute())
    except (TypeError, ValueError, OverflowError) as exc:
        raise DriverValidationError(message) from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise DriverValidationError(message)
    return value


def _timestamp(clock: Clock) -> float:
    if not callable(clock):
        raise DriverValidationError("clock must be callable")
    return _finite(clock, "clock must return a finite number")


def _lease_window(ttl_seconds: Any, clock: Clock) -> tuple[float, float]:
    """Validate ttl (first) and clock -> ``(now, ttl)``; the expiry ``now + ttl`` must stay finite."""
    ttl = _finite(lambda: ttl_seconds, "ttl_seconds must be a finite positive number", positive=True)
    now = _timestamp(clock)
    if not math.isfinite(now + ttl):
        raise DriverValidationError("lease expiry must be finite")
    return now, ttl


def validate_bound_task_manifest(value: Any) -> list[dict[str, Any]]:
    """Validate store metadata plus the exact source event for each task input."""
    from gateway.hosted_room_attachments import validate_task_manifest

    if not isinstance(value, list) or not value:
        raise DriverValidationError("attachments must be a non-empty list")
    if any(not isinstance(item, Mapping) or "event_id" not in item for item in value):
        raise DriverValidationError("task attachment requires event_id")
    event_ids = [_identifier(item["event_id"], label="attachment event_id") for item in value]
    try:
        manifest = validate_task_manifest([
            {key: val for key, val in item.items() if key != "event_id"} for item in value])
    except ValueError as exc:
        raise DriverValidationError(str(exc)) from exc
    return [{**item, "event_id": event_id} for item, event_id in zip(manifest, event_ids)]


def _task_payload(value: Any) -> tuple[dict[str, Any], str, str]:
    if not isinstance(value, dict):
        raise DriverValidationError("payload must be an object")
    unknown = set(value) - _TASK_PAYLOAD_REQUIRED_FIELDS - _TASK_PAYLOAD_OPTIONAL_FIELDS
    missing = _TASK_PAYLOAD_REQUIRED_FIELDS - set(value)
    if unknown:
        raise DriverValidationError(f"unknown payload fields: {', '.join(sorted(unknown))}")
    if missing:
        raise DriverValidationError(f"missing payload fields: {', '.join(sorted(missing))}")
    target_profile = _identifier(value["target_profile"], label="target_profile")
    prompt = text(value["prompt"], error=DriverValidationError, label="prompt", max_bytes=MAX_PROMPT_BYTES, strip=False)
    source_event_seq = _bounded_int(
        value["source_event_seq"], message="source_event_seq must be a positive integer", low=1)
    normalized = {"target_profile": target_profile, "prompt": prompt, "source_event_seq": source_event_seq}
    if "input_context" in value:
        from gateway.hosted_room_task_input import validate_task_input

        try:
            normalized["input_context"] = validate_task_input(value["input_context"])
        except ValueError as exc:
            raise DriverValidationError(str(exc)) from exc
    if "target_member_id" in value:
        normalized["target_member_id"] = _identifier(value["target_member_id"], label="target_member_id")
    if "attachments" in value:
        normalized["attachments"] = validate_bound_task_manifest(value["attachments"])
    if "lead_routing" in value:
        # Present only on a lead's routing turn, and then exactly true: absence is an ordinary member turn.
        if value["lead_routing"] is not True:
            raise DriverValidationError("lead_routing must be true when present")
        normalized["lead_routing"] = True
    encoded = compact_json(normalized)
    return normalized, encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _check_same_room(lease: DriverLease, identity: HostedTaskIdentity) -> None:
    if lease.room_id != identity.room_id:
        raise DriverValidationError("lease and task belong to different rooms")


def _cancel_generation(value: int) -> int:
    # Deliberately accepts bool (a bool is an int); do not swap for non_negative_int.
    if not isinstance(value, int) or value < 0:
        raise DriverValidationError("expected_cancel_generation must be non-negative")
    return value


def _expected_generations(
    lease: DriverLease, identity: HostedTaskIdentity, execution_generation: int, cancel_generation: int) -> None:
    _check_same_room(lease, identity)
    if not isinstance(execution_generation, int) or execution_generation < 1:
        raise DriverValidationError("expected_execution_generation must be a positive integer")
    _cancel_generation(cancel_generation)


def _terminal_settlement_id(settlement_id: Any, status: Any) -> str:
    settlement_id = _identifier(settlement_id, label="settlement_id")
    if status not in {"settled", "failed"}:
        raise DriverValidationError("status must be 'settled' or 'failed'")
    return settlement_id


def _settlement(settlement_id: Any, status: Any, result: Any, clock: Clock) -> tuple[str, float]:
    """Validate one terminal settlement -> ``(settlement_id, now)``; the result must encode within its bound."""
    settlement_id = _terminal_settlement_id(settlement_id, status)
    _canonical_json(result)
    return settlement_id, _timestamp(clock)


def acquire_lease(
    store, *, placement: HostedPlacement, room_id: Any, gateway_id: Any, authority_epoch: Any,
    process_generation: Any, ttl_seconds: Any, clock: Clock) -> DriverLease:
    """Acquire an empty or expired room lease with a monotonic generation."""
    room_id = _identifier(room_id, label="room_id")
    gateway_id = _identifier(gateway_id, label="gateway_id")
    authority_epoch = _bounded_int(authority_epoch, message="authority_epoch must be a positive integer", low=1)
    process_generation = _identifier(process_generation, label="process_generation")
    now, ttl = _lease_window(ttl_seconds, clock)
    return store.hosted_room_driver.acquire_lease(
        placement=placement, room_id=room_id, gateway_id=gateway_id, authority_epoch=authority_epoch,
        process_generation=process_generation, ttl_seconds=ttl, now=now)


def renew_lease(
    store, lease: DriverLease, *, placement: HostedPlacement, ttl_seconds: Any, clock: Clock) -> DriverLease:
    """Renew the exact active lease generation or fail closed."""
    now, ttl = _lease_window(ttl_seconds, clock)
    return store.hosted_room_driver.renew_lease(lease, placement=placement, ttl_seconds=ttl, now=now)


def release_lease(store, lease: DriverLease, *, placement: HostedPlacement, clock: Clock) -> dict[str, Any]:
    """Release the exact active lease generation idempotently."""
    released = store.hosted_room_driver.release_lease(lease, placement=placement, now=_timestamp(clock))
    return {"lease": released.lease, "idempotent": released.idempotent}


def admit_task(
    store, identity: HostedTaskIdentity, *, placement: HostedPlacement, payload: Any, clock: Clock
) -> dict[str, Any]:
    """Persist a queued task, or return the identical admission."""
    _task_payload(payload)
    now = _timestamp(clock)
    return store.hosted_room_driver.admit_task(identity, placement=placement, payload=payload, now=now)


def start_task(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_cancel_generation: int, clock: Clock) -> TaskAttempt:
    """Move one queued task to running under the current driver lease."""
    _check_same_room(lease, identity)
    _cancel_generation(expected_cancel_generation)
    now = _timestamp(clock)
    return store.hosted_room_driver.start_task(
        identity, lease, placement=placement, expected_cancel_generation=expected_cancel_generation, now=now)


def settle_task(
    store, attempt: TaskAttempt, *, placement: HostedPlacement, settlement_id: Any, status: TerminalStatus,
    result: Any, clock: Clock, allow_expired_lease: bool = False) -> dict[str, Any]:
    """Commit one terminal result if every lease and task fence still matches.

    ``allow_expired_lease`` serves the live in-process terminal callback only: plain expiry of our
    own unreleased, un-reclaimed lease must not discard an already-completed first-party result, so
    the post-replay check drops only its ``expires_at > now`` term. Reclaim, release and authority
    transfer still fence; historical harvest paths keep the strict check.
    """
    settlement_id, now = _settlement(settlement_id, status, result, clock)
    return store.hosted_room_driver.settle_task(
        attempt, placement=placement, settlement_id=settlement_id, status=status, result=result, now=now,
        allow_expired_lease=allow_expired_lease)


def settle_stopping_task(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, settlement_id: Any, status: TerminalStatus,
    result: Any, clock: Clock) -> dict[str, Any]:
    """Commit a completion that won the race with an unacknowledged Stop."""
    _terminal_settlement_id(settlement_id, status)  # settlement errors take precedence over generation errors
    if expected_execution_generation < 1 or expected_cancel_generation < 1:
        raise DriverValidationError("stopping settlement generations are invalid")
    settlement_id, now = _settlement(settlement_id, status, result, clock)
    return store.hosted_room_driver.settle_stopping_task(
        identity, lease, placement=placement, expected_execution_generation=expected_execution_generation,
        expected_cancel_generation=expected_cancel_generation, settlement_id=settlement_id, status=status,
        result=result, now=now)


def resolve_indeterminate_task(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, settlement_id: Any, status: TerminalStatus,
    result: Any, clock: Clock) -> dict[str, Any]:
    """Commit a verified historical receipt under the current room lease."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    settlement_id, now = _settlement(settlement_id, status, result, clock)
    return store.hosted_room_driver.resolve_indeterminate_task(
        identity, lease, placement=placement, expected_execution_generation=expected_execution_generation,
        expected_cancel_generation=expected_cancel_generation, settlement_id=settlement_id, status=status,
        result=result, now=now)


def resolve_indeterminate_cancellation(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, cancel_id: Any, clock: Clock
) -> dict[str, Any]:
    """Commit a verified terminal cancellation for an uncertain attempt."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    cancel_id = _identifier(cancel_id, label="cancel_id")
    now = _timestamp(clock)
    return store.hosted_room_driver.resolve_indeterminate_cancellation(
        identity, lease, placement=placement, expected_execution_generation=expected_execution_generation,
        expected_cancel_generation=expected_cancel_generation, cancel_id=cancel_id, now=now)


def requeue_indeterminate_task(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Explicitly retry uncertain work after an operator accepts at-least-once risk."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now = _timestamp(clock)
    return store.hosted_room_driver.requeue_indeterminate_task(
        identity, lease, placement=placement, expected_execution_generation=expected_execution_generation,
        expected_cancel_generation=expected_cancel_generation, now=now)


def defer_indeterminate_task(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, reason: Any, clock: Clock
) -> dict[str, Any]:
    """Fence one uncertain attempt and release later room work."""
    return _defer(store.hosted_room_driver.defer_indeterminate_task, identity, lease, placement=placement,
                  expected_execution_generation=expected_execution_generation,
                  expected_cancel_generation=expected_cancel_generation, reason=reason, clock=clock)


def defer_unconfirmed_stop(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, reason: Any, clock: Clock
) -> dict[str, Any]:
    """Release later room work behind a Stop its member never confirmed: deferred, never cancelled."""
    return _defer(store.hosted_room_driver.defer_unconfirmed_stop, identity, lease, placement=placement,
                  expected_execution_generation=expected_execution_generation,
                  expected_cancel_generation=expected_cancel_generation, reason=reason, clock=clock)


def _defer(
    op, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, reason: Any, clock: Clock
) -> dict[str, Any]:
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    reason = _identifier(reason, label="defer_reason")
    now = _timestamp(clock)
    return op(identity, lease, placement=placement, expected_execution_generation=expected_execution_generation,
              expected_cancel_generation=expected_cancel_generation, reason=reason, now=now)


def requeue_deferred_task(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Explicitly retry a fenced deferred turn under a new generation."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now = _timestamp(clock)
    return store.hosted_room_driver.requeue_deferred_task(
        identity, lease, placement=placement, expected_execution_generation=expected_execution_generation,
        expected_cancel_generation=expected_cancel_generation, now=now)


def reopen_deferred_task(
    store, identity: HostedTaskIdentity, lease: DriverLease, *, placement: HostedPlacement,
    expected_execution_generation: int, expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Return a deferred turn to indeterminate at its same generation (its peer still owns that generation).
    The retry restarts the attempt's clock: its time limit counts from now, not from the first start."""
    _expected_generations(lease, identity, expected_execution_generation, expected_cancel_generation)
    now = _timestamp(clock)
    return store.hosted_room_driver.reopen_deferred_task(
        identity, lease, placement=placement, expected_execution_generation=expected_execution_generation,
        expected_cancel_generation=expected_cancel_generation, now=now)


def requeue_not_admitted_task(
    store, attempt: TaskAttempt, *, placement: HostedPlacement, clock: Clock) -> dict[str, Any]:
    """Return a running task to its durable queue after proven non-admission."""
    now = _timestamp(clock)
    _check_same_room(attempt.lease, attempt.identity)
    return store.hosted_room_driver.requeue_not_admitted_task(attempt, placement=placement, now=now)


def cancel_task(
    store, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: Any,
    expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Cancel a queued task before any external work was admitted."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    _cancel_generation(expected_cancel_generation)
    now = _timestamp(clock)
    return store.hosted_room_driver.cancel_task(
        identity, placement=placement, cancel_id=cancel_id, expected_cancel_generation=expected_cancel_generation,
        now=now)


def begin_task_cancel(
    store, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: Any,
    expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Persist a stop intent without claiming the remote run has stopped."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    _cancel_generation(expected_cancel_generation)
    now = _timestamp(clock)
    return store.hosted_room_driver.begin_task_cancel(
        identity, placement=placement, cancel_id=cancel_id, expected_cancel_generation=expected_cancel_generation,
        now=now)


def complete_task_cancel(
    store, identity: HostedTaskIdentity, *, placement: HostedPlacement, cancel_id: Any,
    expected_cancel_generation: int, clock: Clock) -> dict[str, Any]:
    """Commit cancellation only after the transport acknowledges exact Stop."""
    cancel_id = _identifier(cancel_id, label="cancel_id")
    now = _timestamp(clock)
    return store.hosted_room_driver.complete_task_cancel(
        identity, placement=placement, cancel_id=cancel_id, expected_cancel_generation=expected_cancel_generation,
        now=now)


def mark_attempt_indeterminate(
    store, attempt: TaskAttempt, lease: DriverLease, *, placement: HostedPlacement, clock: Clock
) -> None:
    """Retire exactly the unwatched attempt, even if another control renewed its lease."""
    _check_same_room(lease, attempt.identity)
    store.hosted_room_driver.mark_attempt_indeterminate(attempt, lease, placement=placement, now=_timestamp(clock))


def recover_room(
    store, lease: DriverLease, *, placement: HostedPlacement, clock: Clock) -> dict[str, list[HostedTaskIdentity]]:
    """Fence abandoned running attempts without requeueing uncertain work."""
    recovery = store.hosted_room_driver.recover_room(lease, placement=placement, now=_timestamp(clock))
    return {"queued": list(recovery.queued), "indeterminate": list(recovery.indeterminate)}


def get_task(store, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> dict[str, Any]:
    """Read one task without mutating its state."""
    return store.hosted_room_driver.get_task(identity, placement=placement)


def list_tasks(
    store, *, placement: HostedPlacement, room_id: Any, status: TaskStatus | None = None) -> list[dict[str, Any]]:
    """Return room tasks in deterministic admission order."""
    room_id = _identifier(room_id, label="room_id")
    if status is not None and status not in TASK_STATUSES:
        raise DriverValidationError("invalid task status")
    return list(store.hosted_room_driver.list_tasks(placement=placement, room_id=room_id, status=status))


def room_task_summary(
    store, *, placement: HostedPlacement, room_id: Any, statuses: Iterable[str],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """``({status: count}, tasks in those statuses, in admission order)`` for one room from one read
    snapshot: retained terminal tasks are only counted, never decoded."""
    room_id = _identifier(room_id, label="room_id")
    statuses = tuple(statuses)
    if any(status not in TASK_STATUSES for status in statuses):
        raise DriverValidationError("invalid task status")
    counts, tasks = store.hosted_room_driver.room_task_summary(placement=placement, room_id=room_id, statuses=statuses)
    return dict(counts), list(tasks)


def room_change_rows(
    store, *, placement: HostedPlacement, statuses: tuple[str, ...] = SHOWN_TASK_STATUSES,
) -> dict[str, RoomChangeRows]:
    """Per live room in one store, what its log and ``groups.state`` read from the store, for a change
    signature: ``RoomChangeRows(next_seq, revision, ((task_id, status, execution_generation, cancel_generation), ...))``
    over the tasks in ``statuses``. ``next_seq`` moves on every append and ``revision`` on a rename,
    membership or authority edit; the generations move a Retry that keeps its status. A disbanded room is
    left out (its key going away is the change)."""
    return dict(store.hosted_room_driver.room_change_rows(placement=placement, statuses=statuses))


def prune_published_terminal_tasks(
    store, *, placement: HostedPlacement, room_id: Any, clock: Clock,
    retention_seconds: float = TERMINAL_TASK_RETENTION_SECONDS, retain: int = MAX_RETAINED_TERMINAL_TASKS) -> int:
    """Bound execution rows after outcomes are durable in the room log."""
    room_id = _identifier(room_id, label="room_id")
    now = _timestamp(clock)
    if retention_seconds <= 0:
        raise DriverValidationError("retention_seconds must be positive")
    _bounded_int(retain, message="retain must be a non-negative integer")
    return store.hosted_room_driver.prune_published_terminal_tasks(
        placement=placement, room_id=room_id, now=now, retention_seconds=retention_seconds, retain=retain)


def get_task_for_turn(
    store, identity: HostedTaskIdentity, *, placement: HostedPlacement) -> dict[str, Any] | None:
    """Read the immutable admission for a turn, including older payload versions."""
    return store.hosted_room_driver.get_task_for_turn(identity, placement=placement)
