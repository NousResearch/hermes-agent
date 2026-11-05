"""Cross-process coordination contract: the compression lock, the turn lease, the hygiene streak, session handoffs, the delegation ledger and the delivery-obligation ledger.

What each group of verbs does:

1. Compression lock (``compression_locks``, keyed by the session id): acquisition returns ``bool`` (there is no lease token); a row expired strictly before now, or held by a ``pid=<n>`` holder whose local process is provably dead, is reclaimed inside the same write; a live holder re-acquiring gets True without the expiry moving; refresh decides by holder alone, so a holder revives its own expired, unclaimed row; release is holder-qualified and idempotent. A store failure reads as False (acquire, refresh) or is logged (release); the holder read raises it.
2. Turn lease (``session_turn_leases``, keyed by the conversation: the lineage root reached by walking compression parents while the row is a plain continuation): one write sweeps rows expired more than a day ago, resolves the key, reclaims an expired (``expires_at <= now``) or dead-holder row and inserts; the TTL is clamped to at least 0.1 s; an empty id or holder is False with no write. :func:`acquire_turn_lease_waiting` is a wait loop over the single attempt, polling only on :class:`persistence.StoreContention`.
3. Hygiene streak: an atomic ``+1`` returning the new value; reset deletes the row; an empty key answers 1 (or does nothing) without a write.
4. Handoffs (the ``handoff_*`` columns of the session row): request arms from no state, ``completed`` or ``failed``; claim moves ``pending`` to ``running`` once; fail stores the error cut to 500 characters, optionally only from ``only_states``; the stale-running reclaim fails every ``running`` row in one write and never raises.
5. Delegation ledger (``async_delegations``): dispatch is an insert-or-replace of a ``running`` row with a ``pending`` delivery; the delivery claim admits one consumer, spends one attempt, and is settled by complete, release (dropped once the budget is spent), defer (refunds the attempt) or drop; a claim on an id with no row is True with no write. The sweeps take the caller's owner probe and the copies it lost as data (``dead_owners``, ``lost_ids``) and offer nothing themselves: the helpers in tools/async_delegation.py offer the returned rows after the commit.
6. Delivery obligations (``delivery_obligations``): record is an insert-or-replace of a ``pending`` row followed by the retention prune in the same write; the sweeps take the owner probe as data and claim each row guarded on its previous owner stamp.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from ..core.errors import StoreContention
from ..core.identity import DelegationId, JsonValue, SessionId

logger = logging.getLogger(__name__)

__all__ = [
    "DELEGATION_CLAIM_STALE_SECONDS",
    "DELEGATION_COMPLETION_STATUSES",
    "DELEGATION_EVENT_KEYS",
    "DELEGATION_FAILURE_WINDOW_SECONDS",
    "DELEGATION_MAX_DELIVERY_ATTEMPTS",
    "DELEGATION_MAX_DURABLE_PENDING",
    "DELEGATION_MAX_RETAINED_TERMINAL",
    "DELEGATION_ORIGIN_SOURCES",
    "DELEGATION_ORPHAN_STALE_SECONDS",
    "DELEGATION_REPLAY_MAX_AGE_SECONDS",
    "DELEGATION_RETENTION_SECONDS",
    "DELEGATION_ROUTE_KEYS",
    "DELEGATION_TASK_KEYS",
    "OBLIGATION_FLOOD_MARKER",
    "OBLIGATION_MAX_ATTEMPTS",
    "OBLIGATION_MAX_ROWS",
    "OBLIGATION_RECONNECTED_MARKER",
    "OBLIGATION_RETENTION_SECONDS",
    "OBLIGATION_STALE_AFTER_SECONDS",
    "TURN_LEASE_SWEEP_GRACE_SECONDS",
    "ClaimedObligation",
    "CoordinationRepository",
    "DelegationRecord",
    "DelegationRecovery",
    "DelegationState",
    "DeliveryState",
    "FailedObligationRow",
    "FinishedDelegationRow",
    "HandoffState",
    "HandoffStatus",
    "InflightDelegation",
    "OrphanCompletion",
    "OwnerStamp",
    "PendingCompletion",
    "PendingHandoff",
    "RecordedOwner",
    "RetryCandidate",
    "RetryDue",
    "acquire_turn_lease_waiting",
    "flood_not_before",
    "is_flood_error",
    "turn_lease_holder_pid",
]

#: A turn-lease row expired longer than this is swept by the next acquisition.
TURN_LEASE_SWEEP_GRACE_SECONDS = 86400.0

#: Delivery claims a completion may spend before it converges to ``dropped``.
DELEGATION_MAX_DELIVERY_ATTEMPTS = 8
#: A delivery claim older than this no longer blocks another claimant.
DELEGATION_CLAIM_STALE_SECONDS = 300.0
#: A pending completion older than this (completion time, else dispatch time) is dropped instead of replayed.
DELEGATION_REPLAY_MAX_AGE_SECONDS = 48 * 3600.0
#: Idle time before the orphan sweep considers a pending row.
DELEGATION_ORPHAN_STALE_SECONDS = 60.0
#: Delivered rows older than this are pruned.
DELEGATION_RETENTION_SECONDS = 7 * 24 * 60 * 60
#: Terminal rows kept by the prune, delivered ones deleted first.
DELEGATION_MAX_RETAINED_TERMINAL = 50
#: Terminal rows with a pending delivery kept by the prune.
DELEGATION_MAX_DURABLE_PENDING = 1000
#: How far back ``list_finished_delegations`` reads.
DELEGATION_FAILURE_WINDOW_SECONDS = 24 * 3600.0
#: Session sources whose workers may dispatch delegations.
DELEGATION_ORIGIN_SOURCES = ("cli", "tui", "gui", "tool", "oneshot")
#: The terminal statuses a completion may carry; the stored state becomes this value.
DELEGATION_COMPLETION_STATUSES = ("completed", "success", "error", "interrupted", "stalled", "unknown")
#: The task payload keys a worker dispatch carries.
DELEGATION_TASK_KEYS = frozenset({"goal", "goals", "context", "toolsets", "role", "model", "is_batch", "task_indexes"})
#: Event route key -> ledger column.
DELEGATION_ROUTE_KEYS: Mapping[str, str] = {
    "session_key": "origin_session", "parent_session_id": "parent_session_id",
    "origin_ui_session_id": "origin_ui_session_id", "origin_session_id": "origin_session_id",
}
#: The keys a worker completion event may carry.
DELEGATION_EVENT_KEYS = DELEGATION_TASK_KEYS | frozenset(DELEGATION_ROUTE_KEYS) | frozenset({
    "type", "delegation_id", "status", "completed_at", "dispatched_at", "summary", "error",
    "api_calls", "duration_seconds", "total_duration_seconds", "results", "live_transcripts",
    "group", "exit_reason", "stalled_after_quiet_seconds", "stall_threshold_seconds",
    "stall_phase", "stall_grace_seconds"})

#: Redelivery attempts an obligation may spend.
OBLIGATION_MAX_ATTEMPTS = 3
#: An obligation older than this is abandoned by the sweeps.
OBLIGATION_STALE_AFTER_SECONDS = 24 * 60 * 60
#: Delivered and abandoned obligations older than this are pruned.
OBLIGATION_RETENTION_SECONDS = 7 * 24 * 60 * 60
#: Rows the obligation prune keeps.
OBLIGATION_MAX_ROWS = 500
#: The marker a claimed reconnect replay carries.
OBLIGATION_RECONNECTED_MARKER = ("♻️ Recovered reply — the messaging platform reconnected after the original "
                                 "delivery failed, so this may be a duplicate:\n\n")
#: The marker a claimed flood-refused row carries.
OBLIGATION_FLOOD_MARKER = ("♻️ Recovered reply — the messaging platform's rate limit refused the original, so part of "
                           "it may already have arrived above:\n\n")

_FLOOD_ERROR_PREFIX = "flood_control:"
_FLOOD_RETRY_DEFAULT_SECONDS = 60.0
_RAW_FLOOD_RE = re.compile(r"flood control exceeded.*?retry in\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_HOLDER_PID = re.compile(r"(?:^|:)pid=(\d+)(?::|$)")


def turn_lease_holder_pid(holder: str) -> Optional[int]:
    """The local process id a structured lease holder names in its ``pid=<n>`` segment (``pid=123:turn=abc``), or ``None`` for an unstructured holder: the holder format the dead-holder reclaim keys on."""
    match = _HOLDER_PID.search(holder or "")
    if match is None:
        return None
    pid = int(match.group(1))
    return pid if pid > 0 else None


def _raw_flood_wait(text: str) -> Optional[float]:
    match = _RAW_FLOOD_RE.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def is_flood_error(error: Any) -> bool:
    """A flood refusal: the adapters' ``flood_control:<seconds>`` result, or the platform's own flood wording."""
    text = str(error or "").strip().lower()
    return text.startswith(_FLOOD_ERROR_PREFIX) or _raw_flood_wait(text) is not None


def _flood_wait_seconds(error: Any) -> float:
    text = str(error or "").strip().lower()
    wait = _FLOOD_RETRY_DEFAULT_SECONDS
    if text.startswith(_FLOOD_ERROR_PREFIX):
        try:
            wait = float(text[len(_FLOOD_ERROR_PREFIX):].strip())
        except ValueError:
            wait = _FLOOD_RETRY_DEFAULT_SECONDS
    else:
        raw = _raw_flood_wait(text)
        if raw is not None:
            wait = raw
    return wait if wait > 0 else _FLOOD_RETRY_DEFAULT_SECONDS


def flood_not_before(updated_at: Any, last_error: Any) -> float:
    """The earliest resend of a flood-refused row: its ``updated_at`` plus the platform's wait."""
    try:
        stamp = float(updated_at or 0.0)
    except (TypeError, ValueError):
        stamp = 0.0
    return stamp + _flood_wait_seconds(last_error)


def _notify(callback: Optional[Callable[..., Any]], *args: Any) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        logger.debug("turn lease wait callback failed", exc_info=True)


def acquire_turn_lease_waiting(
    coordination: CoordinationRepository,
    *,
    session_id: SessionId,
    holder: str,
    ttl_seconds: float = 300.0,
    wait_seconds: float = 1800.0,
    poll_interval_seconds: float = 1.0,
    on_wait: Optional[Callable[[float], None]] = None,
    wait_notice_interval_seconds: float = 15.0,
    should_abort: Optional[Callable[[], bool]] = None,
    acquire_patience_s: float = 0.5,
    on_contended: Optional[Callable[[], None]] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait for the turn lease without holding the store.

    ``should_abort()`` is consulted before every attempt and True returns False; each attempt is :meth:`CoordinationRepository.try_acquire_turn_lease` with ``patience_s``. A :class:`persistence.StoreContention` (the write patience ran out behind another writer) is not a holder: ``on_contended()`` fires, the next attempt waits ``acquire_patience_s`` plus the poll interval for the writer (never past the deadline) after one short jitter, and no ``on_wait`` notice is sent. Any other error propagates. A miss notices ``on_wait(elapsed)`` at most every ``wait_notice_interval_seconds`` (``0``: every miss) and sleeps the poll interval, clamped to the deadline; past the deadline the loop returns False. Callback errors are logged and swallowed. ``clock`` and ``sleep`` are injectable for tests.
    """
    deadline = clock() + max(0.0, float(wait_seconds))
    patience = acquire_patience_s
    wait_started: Optional[float] = None
    last_notice_at: Optional[float] = None
    notice_every = max(0.0, float(wait_notice_interval_seconds))
    while True:
        try:
            if should_abort is not None and should_abort():
                return False
        except Exception:
            logger.debug("turn lease should_abort callback failed", exc_info=True)
        try:
            if coordination.try_acquire_turn_lease(session_id=session_id, holder=holder, ttl_seconds=ttl_seconds, patience_s=patience):
                return True
        except StoreContention:
            _notify(on_contended)
            patience = min(acquire_patience_s + max(0.0, float(poll_interval_seconds)), max(acquire_patience_s, deadline - clock()))
            now = clock()
            if now >= deadline:
                return False
            sleep(min(random.uniform(0.020, 0.150), max(deadline - now, 0.001)))
            continue
        patience = acquire_patience_s
        now = clock()
        remaining = deadline - now
        if remaining <= 0:
            return False
        if wait_started is None:
            wait_started = now
        if on_wait is not None and (last_notice_at is None or notice_every == 0.0 or (now - last_notice_at) >= notice_every):
            _notify(on_wait, max(0.0, now - wait_started))
            last_notice_at = now
        sleep(min(max(0.01, float(poll_interval_seconds)), remaining))


class HandoffStatus(StrEnum):
    """A session's stored ``handoff_state``; ``completed`` and ``failed`` are re-requestable."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HandoffState:
    """One session's handoff columns: state, target platform and the last error."""

    state: HandoffStatus
    platform: Optional[str]
    error: Optional[str]


@dataclass(frozen=True, slots=True)
class PendingHandoff:
    """One session in ``handoff_state='pending'``: what the gateway's dispatcher reads from the row (``id``, ``handoff_platform``, ``title``) and its ``started_at``, the listing order."""

    session_id: SessionId
    platform: Optional[str]
    title: Optional[str]
    started_at: float


@dataclass(frozen=True, slots=True)
class OwnerStamp:
    """The calling process as a ledger owner: its pid and its process start time (``None`` when unreadable), minted by the caller."""

    pid: int
    started_at: Optional[int]


@dataclass(frozen=True, slots=True)
class RecordedOwner:
    """An owner stamp as stored on a ledger row, NULLs kept; the verbs match it with ``IS``."""

    pid: Optional[int]
    started_at: Optional[int]


@dataclass(frozen=True, slots=True)
class OrphanCompletion:
    """One orphan-sweep candidate as the caller judges it: the row's id, its stored owner stamp and ``updated_at``."""

    delegation_id: DelegationId
    owner: RecordedOwner
    updated_at: float


class DelegationState(StrEnum):
    """The delegation states the ledger stores: the active and recovery states and the completion statuses."""

    RUNNING = "running"
    STALLING = "stalling"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    SUCCESS = "success"
    ERROR = "error"
    FAILED = "failed"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    STALLED = "stalled"
    UNKNOWN = "unknown"


class DeliveryState(StrEnum):
    """A completion's delivery state: ``pending`` admits claims; ``delivered`` and ``dropped`` are terminal for claims."""

    PENDING = "pending"
    DELIVERED = "delivered"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class DelegationRecord:
    """One ledger row; ``result`` is the stored payload, which the caller redacts, ``origin_session_id`` is ``""`` when unset."""

    delegation_id: DelegationId
    origin_session: str
    state: str
    dispatched_at: float
    completed_at: Optional[float]
    result: Optional[JsonValue]
    delivery_state: DeliveryState
    delivery_attempts: int
    origin_session_id: str

    def __post_init__(self) -> None:
        if not self.delegation_id:
            raise ValueError("DelegationRecord.delegation_id must be non-empty")
        if self.delivery_attempts < 0:
            raise ValueError("DelegationRecord.delivery_attempts must be >= 0")


@dataclass(frozen=True, slots=True)
class PendingCompletion:
    """A completion to offer: the stored event (parsed ``event_json``) with its completion and dispatch times."""

    delegation_id: DelegationId
    event: JsonValue
    completed_at: Optional[float]
    dispatched_at: float


@dataclass(frozen=True, slots=True)
class InflightDelegation:
    """A ``running`` or ``finalizing`` row with what the recovery helper needs to build its event."""

    delegation_id: DelegationId
    origin_session: str
    origin_ui_session_id: str
    parent_session_id: Optional[SessionId]
    dispatched_at: float
    owner_pid: Optional[int]
    owner_started_at: Optional[int]
    task: JsonValue
    origin_session_id: str
    result: Optional[JsonValue]
    state: str
    updated_at: float


@dataclass(frozen=True, slots=True)
class DelegationRecovery:
    """One abandoned delegation to mark ``unknown``, built outside the store: written only while the row still has ``expected_state`` and ``expected_updated_at``."""

    delegation_id: DelegationId
    expected_state: str
    expected_updated_at: float
    completed_at: float
    event: JsonValue
    result: JsonValue


@dataclass(frozen=True, slots=True)
class FinishedDelegationRow:
    """A terminal row of ``list_finished_delegations``: ``task`` and ``result`` are the parsed payloads, ``{}`` when absent or not an object."""

    delegation_id: DelegationId
    state: str
    dispatched_at: float
    completed_at: Optional[float]
    task: JsonValue
    result: JsonValue


@dataclass(frozen=True, slots=True)
class ClaimedObligation:
    """An obligation a sweep claimed or adopted. Fields a row leaves unset (``marker``, ``runtime_recovery``, ``last_error``, ``adopted``, ``not_before``, and ``needs_marker`` on an adopted row) are ``None``/``False``. ``attempts`` is the count after the claim (unchanged on adoption)."""

    obligation_id: str
    session_key: str
    platform: str
    chat_id: str
    thread_id: Optional[str]
    content: str
    profile: str
    attempts: int
    needs_marker: bool = False
    marker: Optional[str] = None
    runtime_recovery: bool = False
    last_error: Optional[str] = None
    adopted: bool = False
    not_before: Optional[float] = None


@dataclass(frozen=True, slots=True)
class FailedObligationRow:
    """A ``failed`` obligation of this owner."""

    platform: str
    adapter_profile: Optional[str]
    updated_at: float
    last_error: Optional[str]
    attempts: int
    created_at: float


@dataclass(frozen=True, slots=True)
class RetryCandidate:
    """A ``failed`` row of this owner, platform and profile as the runtime sweep reads it."""

    obligation_id: str
    updated_at: float
    last_error: Optional[str]
    attempts: int


@dataclass(frozen=True, slots=True)
class RetryDue:
    """A candidate with the earliest resend the helper computed (``retry_not_before``); the claim re-checks the other fields unchanged."""

    obligation_id: str
    updated_at: float
    last_error: Optional[str]
    attempts: int
    not_before: float


@runtime_checkable
class CoordinationRepository(Protocol):
    """The coordination verbs (module docstring); the runtime check is an assembly diagnostic only, never contract coverage."""

    # ── compression lock ──

    def try_acquire_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        """One write: a row expired before now or held by a provably dead ``pid=`` holder is deleted, then insert-or-ignore ours; True when the row names ``holder`` (a live holder's re-acquire keeps its expiry). An empty ``session_id`` is False with no write; an empty holder is admitted. A store failure is logged and reads as False; a replaced store raises :class:`persistence.StoreReplaced`."""
        ...

    def refresh_compression_lock(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0) -> bool:
        """Set ``expires_at`` to now plus the TTL while the row names ``holder``, expired or not; empty arguments and store failures are False; a replaced store raises :class:`persistence.StoreReplaced`."""
        ...

    def release_compression_lock(self, *, session_id: SessionId, holder: str) -> None:
        """Delete the row while it names ``holder``; idempotent; an empty ``session_id`` does nothing; a store failure is logged; a replaced store raises :class:`persistence.StoreReplaced`."""
        ...

    # ── turn lease ──

    def try_acquire_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0, patience_s: Optional[float] = None) -> bool:
        """One write: sweep rows expired more than :data:`TURN_LEASE_SWEEP_GRACE_SECONDS` ago, resolve the conversation key, reclaim an expired (``expires_at <= now``) or dead-holder row, insert-or-ignore ours with a TTL of at least 0.1 s; True when the row names ``holder``. Empty arguments are False with no write. ``patience_s`` is the write patience; exhausted patience raises :class:`persistence.StoreContention`."""
        ...

    def refresh_turn_lease(self, *, session_id: SessionId, holder: str, ttl_seconds: float = 300.0, patience_s: Optional[float] = None) -> bool:
        """Extend the conversation's row while it names ``holder`` (expired or not); ``expires_at`` is stamped before the write waits. Empty arguments are False."""
        ...

    def release_turn_lease(self, *, session_id: SessionId, holder: str) -> None:
        """Delete the conversation's row while it names ``holder``; idempotent; empty arguments do nothing."""
        ...

    # ── hygiene streak ──

    def increment_hygiene_failure_streak(self, *, session_key: str) -> int:
        """Upsert ``+1`` and return the new streak in one write; an empty key returns 1 without a write."""
        ...

    def reset_hygiene_failure_streak(self, *, session_key: str) -> None:
        """Delete the streak row; an empty key does nothing."""
        ...

    # ── handoffs ──

    def request_handoff(self, *, session_id: SessionId, platform: str) -> bool:
        """Arm ``pending`` for ``platform`` and clear the error while no handoff is in flight (state none, ``completed`` or ``failed``); False for an in-flight handoff or no such session."""
        ...

    def claim_handoff(self, *, session_id: SessionId) -> bool:
        """``pending`` to ``running``; True when this call moved it."""
        ...

    def complete_handoff(self, *, session_id: SessionId) -> None:
        """State ``completed``, error cleared, unconditionally."""
        ...

    def fail_handoff(self, *, session_id: SessionId, error: str, only_states: Optional[tuple[HandoffStatus, ...]] = None) -> bool:
        """State ``failed`` with ``error[:500]``; with ``only_states`` only from those states. True when a row changed."""
        ...

    def reclaim_stale_running_handoffs(self, *, error: str) -> tuple[SessionId, ...]:
        """Fail every ``running`` handoff with ``error[:500]`` in one write and return their ids; a store failure is logged and returns ()."""
        ...

    # ── delegation ledger (tools/async_delegation.py) ──

    def record_delegation_dispatch(self, *, delegation_id: DelegationId, origin_session: str, origin_ui_session_id: str, parent_session_id: Optional[SessionId], dispatched_at: float, owner: OwnerStamp, task: JsonValue, origin_session_id: str = '') -> None:
        """Insert or replace the row: state ``running``, delivery ``pending`` with 0 attempts, ``updated_at`` now, the owner stamp and the task payload."""
        ...

    def prune_delegation_ledger(self) -> None:
        """Delete delivered rows not updated for :data:`DELEGATION_RETENTION_SECONDS`, then cap terminal rows at :data:`DELEGATION_MAX_RETAINED_TERMINAL` (delivered, then oldest first) and terminal pending rows at :data:`DELEGATION_MAX_DURABLE_PENDING` (oldest first), in one write."""
        ...

    def record_delegation_completion(self, *, delegation_id: DelegationId, state: str, completed_at: float, event: JsonValue, result: JsonValue) -> None:
        """Store the state, completion time, ``updated_at`` now, the caller-redacted event and result, and delivery ``pending``; a missing row writes nothing."""
        ...

    def record_unit_child(self, *, delegation_id: DelegationId, entry: JsonValue) -> None:
        """Only for a running row: merges the caller-redacted ``entry`` by ``task_index`` into the stored partial and writes ``{results, partial: true}``; a missing or non-running row writes nothing; the stored partial is not re-redacted."""
        ...

    def recover_abandoned_delegations(self, *, recoveries: Sequence[DelegationRecovery]) -> int:
        """One write: each recovery whose row still has its expected state and ``updated_at`` becomes ``unknown`` with the given event, result and completion time, ``updated_at`` now and delivery ``pending``; others are skipped. Returns the rows written."""
        ...

    def take_replayable_completions(self) -> tuple[PendingCompletion, ...]:
        """One write over the startup select (state not ``running``, delivery ``pending``, an event; completion time then id): rows older than :data:`DELEGATION_REPLAY_MAX_AGE_SECONDS` are dropped with their claim cleared, the rest returned."""
        ...

    def sweep_orphaned_completions(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(), dead_owners: frozenset[RecordedOwner], lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        """One write over the orphan candidates (terminal, pending, an event, idle past :data:`DELEGATION_ORPHAN_STALE_SECONDS`, no claim younger than :data:`DELEGATION_CLAIM_STALE_SECONDS`): rows outside ``lost_ids`` (the caller's lost in-memory copies) that are in ``exclude_ids`` or whose owner is not in ``dead_owners`` are skipped, spent rows (``attempts >= 8``) and rows older than 48 h are dropped, the rest returned."""
        ...

    def claim_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        """Claim a pending completion with no claim or a stale one, spending one attempt; a missing row is True with no write."""
        ...

    def release_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        """Under this claim: spent budget converges to ``dropped`` (True), else the claim is cleared; another claim is False."""
        ...

    def defer_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        """Under this claim: clear it and refund one attempt (not below 0)."""
        ...

    def drop_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        """Under this claim: delivery ``dropped``."""
        ...

    def complete_delivery(self, *, delegation_id: DelegationId, claim_id: str) -> bool:
        """Under this claim: delivery ``delivered`` with ``delivered_at`` stamped."""
        ...

    def mark_delegation_delivered(self, *, delegation_id: DelegationId) -> bool:
        """Delivery ``delivered`` with ``delivered_at`` and ``updated_at`` stamped now, unless already delivered; no claim check, a held claim stays stored. True when the row changed (tools/async_delegation.py:465-476)."""
        ...

    def reopen_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        """One write: a delivered row with a spent budget becomes ``dropped`` (None); a delivered row with an event returns to ``pending`` and is returned, or dropped when older than 48 h (None)."""
        ...

    def delete_unscheduled_delegation(self, *, delegation_id: DelegationId) -> None:
        """Delete the row unconditionally."""
        ...

    # ── delivery obligations (gateway/delivery_ledger.py) ──

    def record_delivery_obligation(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str, thread_id: Optional[str], content: str, adapter_profile: Optional[str] = None, owner: OwnerStamp) -> None:
        """Insert or replace a ``pending`` row with 0 attempts, stamped to ``owner`` (``adapter_profile`` stripped, ``'default'`` when unset), then the retention prune in the same write."""
        ...

    def record_crash_left_reply(self, *, obligation_id: str, session_key: str, platform: str, chat_id: str, thread_id: Optional[str], content: str, since: float, adapter_profile: Optional[str] = None) -> None:
        """Insert-or-ignore an unowned ``attempting`` row unless a row with this session key and content was created at or after ``since``."""
        ...

    def mark_obligation_attempting(self, *, obligation_id: str) -> None:
        """State ``attempting``, ``updated_at`` now, error cleared."""
        ...

    def mark_obligation_delivered(self, *, obligation_id: str) -> None:
        """State ``delivered``, ``updated_at`` now, error cleared."""
        ...

    def mark_obligation_failed(self, *, obligation_id: str, error: str = '') -> None:
        """State ``failed``, ``updated_at`` now, ``last_error`` ``error[:500]`` or NULL."""
        ...

    def release_runtime_obligation_claim(self, *, obligation_id: str, error: str = '', owner: OwnerStamp) -> bool:
        """An ``attempting`` row stamped exactly to ``owner`` returns to ``failed`` with one attempt refunded (not below 0) and the error; ``owner.started_at`` None is False with no store access."""
        ...

    def sweep_recoverable_obligations(self, *, now: Optional[float] = None, deliverable_platforms: Optional[frozenset[str]] = None, deliverable_targets: Optional[frozenset[tuple[str, str]]] = None, owner: OwnerStamp, dead_owners: frozenset[RecordedOwner]) -> tuple[ClaimedObligation, ...]:
        """One write: the sweep loop, with ``dead_owners`` as the owner probe (a stamp not in the set counts as live); exhausted rows abandoned, a flood row inside its wait adopted, else a claim guarded on the previous owner."""
        ...

    def sweep_failed_obligations_for_runtime(self, platform: str, *, now: Optional[float] = None, profile: Optional[str] = None, owner: OwnerStamp, due: Sequence[RetryDue]) -> tuple[ClaimedObligation, ...]:
        """``owner.started_at`` None returns () without store access; one write over ``due`` in order, each row re-checked unchanged (state, platform, profile, exact owner, ``updated_at``, ``last_error``, ``attempts``), exhausted abandoned, not yet due skipped, else claimed with the reconnect (or flood) marker."""
        ...

    # ── reads ──

    def get_compression_lock_holder(self, *, session_id: SessionId) -> Optional[str]:
        """The holder of an unexpired (``expires_at >= now``) lock, else None; an empty id is None without a read. A store failure raises; it never reads as no holder."""
        ...

    def turn_lease_expires_at(self, *, session_id: SessionId, holder: str) -> Optional[float]:
        """The committed ``expires_at`` of ``holder``'s turn lease on the conversation, else None; empty arguments are None."""
        ...

    def get_handoff(self, *, session_id: SessionId) -> Optional[HandoffState]:
        """The handoff columns, or None for a missing session or one with no handoff state; never raises."""
        ...

    def list_pending_handoffs(self) -> tuple[PendingHandoff, ...]:
        """Every ``pending`` handoff, oldest ``started_at`` first; never raises."""
        ...

    def has_pending_handoffs(self) -> bool:
        """Whether any session is ``pending``."""
        ...

    def get_delegation(self, *, delegation_id: DelegationId) -> Optional[DelegationRecord]:
        """The ledger row, or None."""
        ...

    def list_finished_delegations(self, *, origin_ui_session_id: str = '', parent_session_id: str = '', limit: int = 20, now: Optional[float] = None) -> tuple[FinishedDelegationRow, ...]:
        """Terminal rows matching either selector completed within :data:`DELEGATION_FAILURE_WINDOW_SECONDS`, newest first, at most ``limit``; no selector returns () without a read."""
        ...

    def list_owned_failed_obligations(self, *, owner: OwnerStamp) -> tuple[FailedObligationRow, ...]:
        """``failed`` rows stamped exactly to ``owner``; ``owner.started_at`` None returns () without a read."""
        ...

    def list_inflight_delegations(self) -> tuple[InflightDelegation, ...]:
        """One read: every ``running`` or ``finalizing`` delegation row."""
        ...

    def has_replayable_completions(self) -> bool:
        """One read: whether :meth:`take_replayable_completions` has a row to return or drop."""
        ...

    def list_orphan_completion_owners(self, *, now: float) -> frozenset[RecordedOwner]:
        """One read: the distinct owner stamps over the orphan-completion candidate predicate."""
        ...

    def list_orphan_completions(self, *, now: float) -> tuple[OrphanCompletion, ...]:
        """One read: every orphan-completion candidate (``sweep_orphaned_completions``'s predicate), by completion time then id."""
        ...

    def list_orphan_completion_offers(self, *, now: float, exclude_ids: frozenset[DelegationId] = frozenset(), dead_owners: frozenset[RecordedOwner], lost_ids: frozenset[DelegationId] = frozenset()) -> tuple[PendingCompletion, ...]:
        """One read: what :meth:`sweep_orphaned_completions` returns for the same arguments, without its drops, so the caller offers them before the sweep commits."""
        ...

    def get_admitted_completion(self, *, delegation_id: DelegationId) -> Optional[PendingCompletion]:
        """One read: what :meth:`reopen_admitted_completion` returns, without its write, so the caller offers it before the reopen commits."""
        ...

    def list_obligation_owners(self) -> frozenset[RecordedOwner]:
        """One read: the distinct owner stamps of obligations in ``pending``, ``attempting`` or ``failed``."""
        ...

    def list_runtime_retry_candidates(self, platform: str, *, profile: Optional[str] = None, owner: OwnerStamp) -> tuple[RetryCandidate, ...]:
        """One read: the ``failed`` rows on ``platform`` stamped exactly to ``owner`` with the expected profile (``'default'`` for None); ``owner.started_at`` None returns () without a read."""
        ...
