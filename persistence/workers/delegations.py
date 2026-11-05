"""The worker's delegation ledger: the ``delegation.*`` operations of the worker receipt stream.

A compute worker that dispatches background delegations keeps their rows on its owner (``tools/async_delegation_worker.py``); the owner settles each operation through these verbs after its claim check (gateway/session_worker_delegation.py). Each receipted verb is one operation of the receipt transaction documented in :mod:`persistence.workers.executions`, on the ``async_delegations`` table :class:`persistence.authority.coordination.CoordinationRepository` also serves. Each record carries the wire payload key for key, so the digest matches the producer's. A caller holding the worker's payload as sent passes it as a :class:`WorkerDelegationPayload` in place of the record (or of ``delegation_id``), so a payload no record can hold (a missing or extra key) still meets the fences first; one that is not a mapping is ``invalid_params`` before them, as every receipt payload. Payload checks refuse with ``invalid_params`` after the receipt fences and the replay.

Every receipted verb takes the worker's verified process identity, ``worker_pid`` and ``worker_birth`` (the claim's ``pid`` and ``birth``), apart from the payload, and checks it first (gateway/session_worker_delegation.py:176-181): ``execution_id`` a non-empty string of at most 512 characters, ``worker_pid`` a positive int, ``worker_birth`` a finite non-negative int or float, each else ``invalid_params``, before the payload's mapping check, the receipt fences and the replay of a stored receipt.

Ownership, the first step of every verb but ``dispatch_delegation``: ``delegation_id`` a non-empty string of at most 512 characters (``invalid_params``); an unknown id ``not_found``; a delegation of another session or another execution ``permission_denied``. The delivery attempt cap is ``DELEGATION_MAX_DELIVERY_ATTEMPTS`` and the claim staleness ``DELEGATION_CLAIM_STALE_SECONDS`` (``persistence/authority/coordination.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol

from ..core.fences import AuthorityEpoch, ExecutionGeneration
from ..core.identity import JsonValue, SessionId

if TYPE_CHECKING:
    from .executions import WorkerExecutionId

__all__ = [
    "WorkerDelegationChild",
    "WorkerDelegationClaim",
    "WorkerDelegationCompletion",
    "WorkerDelegationDispatch",
    "WorkerDelegationLedger",
    "WorkerDelegationPayload",
    "WorkerDelegationView",
    "delegation_payload",
]


@dataclass(frozen=True, slots=True)
class WorkerDelegationDispatch:
    """A ``delegation.dispatch`` payload: the delegation's id, task document, dispatch time and the four route keys a completion repeats."""

    delegation_id: str
    task: Mapping[str, JsonValue]
    dispatched_at: float
    session_key: str
    parent_session_id: str
    origin_ui_session_id: str
    origin_session_id: str


@dataclass(frozen=True, slots=True)
class WorkerDelegationCompletion:
    """A ``delegation.complete`` payload: the notification ``event`` (its ``delegation_id``, ``status``, ``completed_at`` and route keys are checked) and the ``result`` document."""

    event: Mapping[str, JsonValue]
    result: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class WorkerDelegationChild:
    """A ``delegation.child`` payload: one child's partial result ``entry``, keyed by its ``task_index``."""

    delegation_id: str
    entry: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class WorkerDelegationClaim:
    """A delivery claim token on one delegation (claim, ack, drop, defer, release)."""

    delegation_id: str
    claim_id: str


@dataclass(frozen=True, slots=True)
class WorkerDelegationPayload:
    """A ``delegation.*`` payload exactly as the worker sent it, well formed or not, in place of the operation's record."""

    payload: JsonValue


def delegation_payload(value: Any) -> Any:
    """The operation payload a verb argument stands for: a :class:`WorkerDelegationPayload`'s payload as sent, a record's fields key for key, a bare ``delegation_id`` as ``{"delegation_id": value}``."""
    if isinstance(value, WorkerDelegationPayload):
        return value.payload
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return {"delegation_id": value}


@dataclass(frozen=True, slots=True)
class WorkerDelegationView:
    """The ``delegation.read`` answer: state, delivery bookkeeping and the decoded result document."""

    delegation_id: str
    origin_session: str
    state: str
    dispatched_at: float
    completed_at: float | None
    delivery_state: str
    delivery_attempts: int
    origin_session_id: str
    result: Mapping[str, JsonValue] | None


class WorkerDelegationLedger(Protocol):
    """The worker delegation operations; the receipted ones run on the worker's session ``session_id``."""

    def dispatch_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, dispatch: WorkerDelegationDispatch | WorkerDelegationPayload, worker_pid: int, worker_birth: float, owner_started_at: int | None, probe_refused: Literal['permission_denied', 'worker_not_live'] | None = None) -> None:
        """``delegation.dispatch`` (gateway/session_worker_delegation.py:48-86). The payload: exactly the seven keys, the id text, a finite non-negative ``dispatched_at``, task keys within ``DELEGATION_TASK_KEYS``; a session that is missing or whose source is not in ``DELEGATION_ORIGIN_SOURCES`` is ``unsupported_producer``; route keys other than the session's own are ``permission_denied``; an existing id is checked for ownership then refused ``admission_conflict``; then ``probe_refused`` (the caller's probe of process ``worker_pid`` against ``worker_birth``, run before the call) is raised; then the row is inserted ``running`` with ``owner_pid`` ``worker_pid`` and ``owner_started_at``. An exact replay returns the stored receipt and ignores both probe values."""
        ...

    def complete_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, completion: WorkerDelegationCompletion | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        """``delegation.complete`` (:89-105): event keys within ``DELEGATION_EVENT_KEYS``, ``type`` ``async_delegation``, a status in ``DELEGATION_COMPLETION_STATUSES``, a dict result; ownership; route keys repeating the dispatch (``permission_denied``); a finite non-negative ``completed_at``; a delegation no longer running is ``admission_conflict``. The row takes the status, completion time, event and result."""
        ...

    def record_delegation_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, child: WorkerDelegationChild | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        """``delegation.child`` (:108-121): ownership; an entry dict with a non-negative int ``task_index``; a running delegation keeps one entry per index in its partial result, any other is left untouched."""
        ...

    def read_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> WorkerDelegationView:
        """``delegation.read`` (:124-129): ownership, then the view."""
        ...

    def claim_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        """``delegation.claim`` (:132-143): ownership and token text; a running or finalizing delegation answers False; a pending delivery whose claim is free or older than ``DELEGATION_CLAIM_STALE_SECONDS`` takes the token and one attempt."""
        ...

    def ack_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        """``delegation.ack`` (:146-164): for a pending delivery held by the token, delivered now and the claim released; else False."""
        ...

    def drop_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        """``delegation.drop``: as ``ack``, the delivery dropped."""
        ...

    def defer_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        """``delegation.defer``: as ``ack``, one attempt refunded (never below 0)."""
        ...

    def release_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        """``delegation.release``: as ``ack``, the delivery dropped once its attempts reach ``DELEGATION_MAX_DELIVERY_ATTEMPTS``."""
        ...

    def withdraw_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        """``delegation.unscheduled`` (:167-173): ownership; only a running delegation without a result is deleted (``admission_conflict`` otherwise)."""
        ...

    def pending_delegation_event(self, *, execution_id: WorkerExecutionId, session_id: SessionId, delegation_id: str) -> Mapping[str, JsonValue] | None:
        """One read after a completion commits (:221-224): ownership, then the stored event while its delivery is pending, else ``None``."""
        ...
