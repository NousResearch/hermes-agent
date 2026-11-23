"""The worker's compression state: the ``compression.*`` lock, cooldown, counter, watermark, reopen and boundary operations of the worker receipt stream.

A managed worker drives its compressor's durable state through the owner (``agent/runtime_session_compression.py``); each verb is one operation of the receipt transaction documented in :mod:`persistence.workers.executions`. Each writes the state the owner's own verbs read: the compression lock and turn lease of :class:`persistence.authority.coordination.CoordinationRepository`, the cooldown and counters on the session row, the transcript's active rows. Values are kept as sent (epoch-second floats), so the digest matches the producer's. Payload checks refuse with ``invalid_params`` after the receipt fences and the replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..conversation.messages import ConversationPeer
from ..core.fences import AuthorityEpoch, ExecutionGeneration
from ..core.identity import SessionId

if TYPE_CHECKING:
    from .executions import WorkerExecutionId, WorkerTurnLease

__all__ = ["WorkerCompressionRepository", "WorkerCooldown", "WorkerCooldownSnapshot"]


@dataclass(frozen=True, slots=True)
class WorkerCooldown:
    """A ``compression.cooldown.record`` payload ``{"cooldown_until", "error"}``: a finite epoch-second deadline and the latest diagnostic."""

    cooldown_until: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerCooldownSnapshot:
    """The cooldown a worker captured before an attempt (``{"session_exists", "cooldown_until", "error"}``), restored after a failed one."""

    session_exists: bool
    cooldown_until: float | None = None
    error: str | None = None


class WorkerCompressionRepository(Protocol):
    """The worker compression state operations; each runs the receipt transaction on the worker's session ``session_id``."""

    def acquire_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        """``compression.lock.acquire``: the session's compression lock for ``lease.holder`` until now + ``ttl_seconds``; an expired lock or one whose holder this host proves dead is reclaimed; a live lock of the same holder answers True unchanged, another holder's False."""
        ...

    def renew_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        """``compression.lock.renew``: move the expiry of ``lease.holder``'s lock to now + ``ttl_seconds``, expired or not; True when a row changed."""
        ...

    def compression_lock_holder(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        """``compression.lock.holder`` with ``{}``: the holder of the unexpired lock, else ``None``; a replay answers the holder recorded then."""
        ...

    def cleanup_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        """``compression.cleanup`` with ``{"target", "holder"}``: the target readable by the worker and on its conversation key (``permission_denied``); then ``holder``'s lock on the target is deleted."""
        ...

    def record_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, cooldown: WorkerCooldown) -> None:
        """``compression.cooldown.record``: the stored deadline merge-maxes with ``cooldown_until``; the error is always replaced."""
        ...

    def restore_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, snapshot: WorkerCooldownSnapshot) -> None:
        """``compression.cooldown.restore`` with ``{"snapshot"}``: deadline and error written as captured; a snapshot taken without a session row, or a read-back that differs, is :class:`persistence.CooldownRestoreRefused` and commits nothing."""
        ...

    def clear_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        """``compression.cooldown.clear`` with ``{}``: the restore of ``(True, None, None)``."""
        ...

    def set_compression_fallback_streak(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, streak: int) -> None:
        """``compression.fallback`` with ``{"value"}``: a non-negative int."""
        ...

    def set_compression_ineffective_count(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, count: int) -> None:
        """``compression.ineffective`` with ``{"value"}``: a non-negative int."""
        ...

    def set_compression_recovery_deadline(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, deadline: float | None) -> None:
        """``compression.recovery`` with ``{"value"}``: a finite number of epoch seconds, or ``None`` to disarm."""
        ...

    def compression_watermark(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> int:
        """``compression.watermark`` with ``{}``: the newest active row id of the session, 0 when none."""
        ...

    def reopen_orphaned_compression(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> bool:
        """``compression.reopen`` with ``{}``: clear the compression end stamp of a session with no continuation child and no live lock (an expired lock row is deleted); False otherwise."""
        ...

    def compression_boundary(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, peer: ConversationPeer) -> int | None:
        """``compression.boundary`` with ``{"session_key", "source"}``: the peer must be the worker session's own (``permission_denied``); the conversation generation ``MessageRepository.latest_conversation_boundary`` reads."""
        ...
