"""Worker execution contract: registration, adoption, reservations, the strict receipt stream and the execution reads.

A background worker (``agent/managed_worker.py``) owns one execution row fenced by the authority epoch and the session's runtime generation; every durable write it makes is one ``worker.persist`` operation the owner settles through these verbs (``gateway/session_worker.py``). Every verb that takes ``sequence`` runs the worker receipt transaction, shared by all seven worker protocols:

- before the store: a ``sequence`` that is not an int of at least 1, or a payload whose JSON encodes to more than :data:`persistence.workers.vocabulary.RECEIPT_PAYLOAD_MAX_BYTES`, is ``invalid_params``;
- in one transaction, in this order: the epoch fence (``stale_epoch``); the assignment (no such execution ``not_found``, an execution of another session ``permission_denied``, a generation other than the row's or the session's runtime generation ``stale_generation``; an identical committed publication still addressed to the parent it left resolves to the worker's current session); a worker owned by another epoch (``stale_epoch``); a committed receipt for the sequence returns its stored result when the payload digest matches, ``admission_conflict`` when it differs, ``stale_generation`` when it was compacted after the execution ended; a worker that is not ``registered`` or ``running`` (``stale_generation``); a sequence other than ``last_sequence + 1`` (``invalid_params``); then the operation, which validates its payload (``invalid_params``) and refuses with its own reasons; then the receipt, ``last_sequence`` and status ``running`` (``terminal`` for ``finish``), and ``runtime_revision`` + 1 when the operation wrote durable state (always for ``finish``).

The receipt digest is ``admission_fingerprint(canonical_target=session_id, payload={"operation", "payload"})`` over the producer's wire payload, which each record carries key for key (``TypedDict`` where key presence matters), so a receipt committed from the raw request replays through the verb and the reverse. A refusal raises :class:`persistence.AuthorityRefusal` with the reason named above (``FenceStale``, ``ProofRejected`` and ``UnknownExecution`` for their reasons) and commits nothing. Receipt and record times are epoch-second floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, NewType, Protocol, TypedDict, Unpack, runtime_checkable

from ..authority.admissions import AdmissionId
from ..authority.vocabulary import OwnerRule
from ..core.fences import AuthorityEpoch, ExecutionGeneration
from ..core.identity import JsonValue, SessionId
from .compaction import WorkerCompactionVerbs
from .compression import WorkerCompressionRepository
from .compression_reads import WorkerCompressionReads
from .delegations import WorkerDelegationLedger, WorkerDelegationPayload
from .receipts import WorkerReceiptReads
from .sessions import WorkerSessionVerbs

__all__ = [
    "AdmissionWorkerScope",
    "ChildExecution",
    "ChildReservation",
    "WorkerAuxiliaryUsage",
    "WorkerExecutionId",
    "WorkerExecutionRecord",
    "WorkerExecutionRepository",
    "WorkerKind",
    "WorkerMainUsage",
    "WorkerStatus",
    "WorkerTurnLease",
]

#: Caller-minted identity of one worker execution (``worker_executions.execution_id``).
WorkerExecutionId = NewType("WorkerExecutionId", str)


class WorkerKind(StrEnum):
    """The stored producer kinds (``worker_executions.kind``)."""

    CRON = "cron"
    CHILD = "child"
    COMPUTE = "compute"
    KANBAN = "kanban"


class WorkerStatus(StrEnum):
    """The stored execution statuses: ``registered`` on insert, ``running`` once adopted or writing, ``unknown`` after an owner restart, ``terminal`` for good."""

    REGISTERED = "registered"
    RUNNING = "running"
    UNKNOWN = "unknown"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class WorkerExecutionRecord:
    """One ``worker_executions`` row without its adoption digest."""

    execution_id: WorkerExecutionId
    session_id: SessionId
    kind: WorkerKind
    owner_epoch: AuthorityEpoch
    generation: ExecutionGeneration
    status: WorkerStatus
    last_sequence: int

    def __post_init__(self) -> None:
        if self.last_sequence < 0:
            raise ValueError("WorkerExecutionRecord.last_sequence must be >= 0")


@dataclass(frozen=True, slots=True)
class AdmissionWorkerScope:
    """The private bootstrap scope of an admission-reserved worker (gateway/session_worker_reservation.py:285-291): the claim fields and the reserving epoch. ``secret`` is the worker's secret; never publish it."""

    profile_id: str
    session_id: SessionId
    execution_id: WorkerExecutionId
    generation: ExecutionGeneration
    pid: int
    birth: float
    secret: str
    epoch: AuthorityEpoch


@dataclass(frozen=True, slots=True)
class ChildExecution:
    """A reserved delegated child: its session, execution id and generation (always 0)."""

    session_id: SessionId
    execution_id: WorkerExecutionId
    generation: ExecutionGeneration


@dataclass(frozen=True, slots=True)
class ChildReservation:
    """The ``child.reserve`` receipt: the reserved ``child``, or the ``refused`` code (``delegation_depth_exceeded``, ``delegation_capacity`` or the caller's refusal)."""

    refused: str | None = None
    child: ChildExecution | None = None


class WorkerMainUsage(TypedDict, total=False):
    """A ``usage.main`` payload as the worker sends it (agent/turn_usage.py:264, agent/codex_runtime.py:144): any subset of these keys, each absent key absent from the digest."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    api_call_count: int
    estimated_cost_usd: float | None
    actual_cost_usd: float | None
    cost_status: str | None
    cost_source: str | None
    pricing_version: str | None
    model: str | None
    billing_provider: str | None
    billing_base_url: str | None
    billing_mode: str | None
    absolute: bool
    task: str
    source: str | None


class WorkerAuxiliaryUsage(TypedDict, total=False):
    """A ``usage.auxiliary`` payload without its ``task``: any subset of these keys."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    api_call_count: int
    estimated_cost_usd: float | None
    model: str | None
    billing_provider: str | None
    billing_base_url: str | None


@dataclass(frozen=True, slots=True)
class WorkerTurnLease:
    """One lease request (turn lease or compression lock): the payload ``{"holder", "ttl_seconds"}``. The verb checks it: ``holder`` a non-empty string of at most 1024 characters, ``ttl_seconds`` a finite number in ``[0.1, 3600]``."""

    holder: str
    ttl_seconds: float = 300.0


@runtime_checkable
class WorkerExecutionRepository(WorkerSessionVerbs, WorkerCompressionRepository, WorkerCompactionVerbs, WorkerCompressionReads, WorkerDelegationLedger, WorkerReceiptReads, Protocol):
    """The worker execution verbs of one profile's store; the store's ``worker_executions`` repository also implements the six other worker protocols. The runtime check is an assembly diagnostic only."""

    def register(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, kind: WorkerKind, adoption_secret: str, require_idle: bool = False) -> WorkerExecutionRecord:
        """Register an execution on ``session_id``.

        Before the store: ``execution_id``, ``session_id`` and ``adoption_secret`` must be non-empty strings of at most 1024 characters and ``kind`` a :class:`WorkerKind` value (``invalid_params``). In one write: epoch fence; the session (``not_found``); its runtime generation equal to ``generation`` (``stale_generation``); an existing row with the same session, generation, kind, owner epoch and secret digest is returned, any other existing row is ``admission_conflict``; over the session's runtime lineage (its physical target, its local owner when the owner's lineage holds it, and every compression relative), ``require_idle`` refuses a queued, started or unknown admission (``stale_generation``), a registered, running or unknown worker refuses (``stale_generation``), an unknown admission refuses (``unknown_execution``); a lineage walk past the chain cap is ``storage_unavailable``. The new row is ``registered`` under ``epoch`` with ``last_sequence`` 0 and the SHA-256 of the secret as its adoption digest.
        """
        ...

    def adopt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        """Re-own the execution under ``epoch`` on proof of its secret.

        A secret outside the text bound is ``invalid_params`` before the store. In one write: epoch fence; the assignment; a terminal execution ``stale_generation``; a digest mismatch (compared in constant time) ``permission_denied``; an ``admission-worker:`` execution owned by another epoch ``stale_epoch``; the one started or unknown admission linked to the session (target or lineage) at the generation and the row's owner epoch moves to ``epoch`` as ``started`` (two linked admissions are ``admission_conflict``); the row becomes ``running`` under ``epoch``.
        """
        ...

    def reserve_for_admission(self, *, epoch: AuthorityEpoch, admission_id: AdmissionId, principal_id: str, profile_id: str, pid: int, birth: float, secret: str, owner: OwnerRule | None, worker_subject: str | None = None) -> AdmissionWorkerScope:
        """Reserve a private compute worker for an admission ``epoch`` started (gateway/session_worker_reservation.py:251-293). The caller verifies the process identity (``pid``, ``birth``) and mints ``secret`` before the call.

        In one write: epoch fence; the admission (live or tombstone, else ``not_found``); not ``started`` under ``epoch`` is ``producer_not_started``; a stored principal that does not belong to ``principal_id`` under ``owner`` (``session_rules.legacy_local.same_owner``; without a rule only the equal stamp) is ``permission_denied``; the target's runtime generation differing from the admission's is ``stale_generation`` (a missing target ``not_found``); the physical target is resolved (an invalid binding receipt is ``storage_unavailable``); an existing ``admission-worker:<admission_id>`` execution is ``admission_conflict``; a non-terminal worker on the physical target ``stale_generation``. Then the physical target takes the admission's generation, joins the admission's lineage when it differs from the target, and a ``registered`` compute execution is inserted whose digest is the SHA-256 of ``admission_fingerprint`` over the scope and the principal the worker authenticates as (``worker_subject``, else ``principal_id``). Returns the scope and ``epoch``.
        """
        ...

    def lose_reservation(self, *, epoch: AuthorityEpoch, admission_id: AdmissionId, execution_id: WorkerExecutionId, generation: ExecutionGeneration) -> None:
        """Revoke a reserved worker whose process was lost (gateway/session_worker_reservation.py:296-317); ``epoch`` is the reservation's. In one write: epoch fence; the admission (``not_found``); not ``started`` under ``epoch`` at ``generation`` is ``stale_generation``; a missing worker or one at another generation or epoch is ``stale_generation``. The worker becomes terminal (read receipts compacted, children retired), the admission ``unknown``, and the target's runtime revision moves by one."""
        ...

    def reserve_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, secret: str | WorkerDelegationPayload, profile_id: str, pid: int, birth: float, principal: str, max_concurrent_children: int, max_spawn_depth: int, refused: str | None = None) -> ChildReservation:
        """Operation ``child.reserve`` (``gateway/session_worker_child.py``): the digest covers the payload ``{"secret"}`` only; the claim fields, the two limits and ``refused`` (the caller's admission-closed or unreadable-policy code) are trusted values read before the call. A caller holding the worker's payload as sent passes it as a :class:`WorkerDelegationPayload` in place of ``secret``, so a payload of any other shape meets the fences first.

        After the fences and the replay, a payload other than ``{"secret"}`` with a string of 16 to 256 characters is ``invalid_params``. A refusal is a committed receipt, not an error: ``refused`` when given, else ``delegation_depth_exceeded`` when the execution id already holds ``max_spawn_depth`` child separators, else ``delegation_capacity`` when the execution has ``max_concurrent_children`` live direct children. Otherwise a ``subagent`` session row is minted (``model_config`` ``{"_delegate_from": session_id}``, the parent's profile) and a ``running`` child execution at generation 0 under ``epoch``, whose digest is the SHA-256 of ``admission_fingerprint`` over the child's scope and ``principal``.
        """
        ...

    def persist_operation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, operation: str, payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Apply one worker operation from its payload as sent, as the owner's ``worker.persist`` request carries it (``gateway/session_worker.py``): every operation a typed verb of the seven worker protocols names except ``child.reserve`` and the ``delegation.*`` ledger, whose verbs take values the caller reads. Returns the stored receipt result, the wire shape the typed verbs decode (``{"value": ...}`` for most operations).

        An operation outside that set is ``invalid_params`` before the store, as are the sequence and payload checks of the receipt transaction; the fences, the replay and the receipt run as for the typed verb, then the operation checks its payload and refuses as the typed verb's operation does, ``compression.publish`` raising :class:`persistence.CompressionPublishRefused` as :meth:`publish_compression_child` does. A typed verb's own checks before the store (an open keyword key set) do not apply: the payload as sent reaches the fences first, and the digest is the producer's.
        """
        ...

    def finish(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        """Operation ``execution.finish`` with payload ``{}``: the execution becomes terminal, its read receipts are compacted, its live descendants are retired (their rows ended ``agent_close``) and a child's own session row is ended."""
        ...

    def record_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, **usage: Unpack[WorkerMainUsage]) -> None:
        """Operation ``usage.main``: ``source`` is set aside; the other keys must lie in :class:`WorkerMainUsage` within these bounds (counters ints in ``[0, 2**53]``, costs ``None`` or numbers in ``[0, 1e12]``, ``absolute`` a bool, text ``None`` or at most 4096 characters), else ``invalid_params``. The session's counters and route are updated as ``update_token_counts`` does (``absolute`` sets the counters and skips attribution), then ``source`` mapped through ``created_surface`` fills an unset ``created_source`` and the compression ancestors that share it."""
        ...

    def record_auxiliary_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, task: str, **usage: Unpack[WorkerAuxiliaryUsage]) -> None:
        """Operation ``usage.auxiliary`` with payload ``dict(usage, task=task)``: keys within :class:`WorkerAuxiliaryUsage` plus a non-empty ``task``, bounded as ``record_usage`` (``invalid_params``). Only the attribution row of ``task`` is written (``api_call_count`` defaults to 1; a missing model is ``unknown``, missing billing fields stay unset)."""
        ...

    def acquire_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        """Operation ``turn.acquire``: the turn lease of the session's conversation key for ``lease.holder`` until now + ``ttl_seconds``. An expired row, or one whose holder names a process this host proves dead, is reclaimed; a live row of the same holder answers True unchanged; another holder's live row answers False."""
        ...

    def renew_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        """Operation ``turn.renew``: move the expiry of ``lease.holder``'s row on the conversation key to now + ``ttl_seconds``, expired or not; True when a row changed."""
        ...

    def cleanup_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        """Operation ``turn.cleanup`` with payload ``{"target", "holder"}``: the target must be readable by the worker (its lineage root to tip plus its compression chain) and share the worker session's conversation key (``permission_denied``); then ``holder``'s turn lease on that key is deleted. The receipt is ``{"released": True}``."""
        ...

    def verify_claim(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        """The owner's per-request claim check (gateway/session_worker.py:48-62), one read: no such execution ``not_found``; assigned to another session than ``session_id`` (the target ``publication_target`` resolved) ``permission_denied``; ``generation`` not an int or not the row's generation ``stale_generation`` (no session counter, no epoch fence); the secret's digest unequal to the stored one ``permission_denied``. Returns the current record, terminal included."""
        ...

    def execution_session(self, *, execution_id: WorkerExecutionId) -> SessionId | None:
        """The session an execution is assigned to, ``None`` when there is no such execution (gateway/session_managed_worker.py:492-497, gateway/session_worker_child.py:131-135)."""
        ...

    def has_live_worker(self) -> bool:
        """Whether any execution is not terminal (gateway/run_idle_exit.py:131)."""
        ...
