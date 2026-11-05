"""The worker's row-carrying compression writes: ``compression.archive``, ``compression.publish`` and ``compression.append``.

A managed worker's compressor compacts its transcript in place, publishes a compression continuation and appends under its compression lock through the owner (``agent/runtime_session_compression.py``); each verb is one operation of the receipt transaction documented in :mod:`persistence.workers.executions`, running the owner's own compaction, publication and append bodies inside the receipt. Rows are :class:`persistence.workers.sessions.WorkerMessage` dicts within :data:`persistence.workers.vocabulary.HANDOFF_MESSAGE_FIELDS`; each record carries the producer's payload key for key, so the digest matches the producer's. Payload checks refuse with ``invalid_params`` after the receipt fences and the replay; a ``_row_id`` outside the worker's transcript is ``permission_denied``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from ..core.fences import AuthorityEpoch, ExecutionGeneration
from ..core.identity import JsonValue, SessionId
from .sessions import RowAnnotation, WorkerAppendResult, WorkerMessage

if TYPE_CHECKING:
    from .executions import WorkerExecutionId

__all__ = [
    "WorkerArchiveResult",
    "WorkerCompaction",
    "WorkerCompactionVerbs",
    "WorkerCompressionPublication",
    "WorkerPublishResult",
]


def _rows(rows: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    return None if rows is None else [dict(row) for row in rows]


@dataclass(frozen=True, slots=True)
class WorkerCompaction:
    """A ``compression.archive`` payload: the replacement ``messages``, the sidecar ``model_config_patch``, the ``watermark`` the summary covered, the compression ``lock_holder`` (``None`` for a lease-less micro-compaction or prune), the ``tail_count`` the replacement carries, and the held-row proofs (``carried_messages``, ``covered_ids``, ``unresolved_held``, ``held_for_cas``). :meth:`payload` sends all nine keys, as the producer does."""

    messages: Sequence[WorkerMessage]
    model_config_patch: Mapping[str, JsonValue] | None = None
    watermark: int | None = None
    lock_holder: str | None = None
    tail_count: int = 0
    carried_messages: Sequence[Mapping[str, JsonValue]] = ()
    covered_ids: Sequence[int] | None = None
    unresolved_held: Sequence[Mapping[str, JsonValue]] | None = None
    held_for_cas: Sequence[Mapping[str, JsonValue]] | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "messages": _rows(self.messages), "model_config_patch": None if self.model_config_patch is None else dict(self.model_config_patch),
            "watermark": self.watermark, "lock_holder": self.lock_holder, "tail_count": self.tail_count,
            "carried_messages": _rows(self.carried_messages), "covered_ids": None if self.covered_ids is None else list(self.covered_ids),
            "unresolved_held": _rows(self.unresolved_held), "held_for_cas": _rows(self.held_for_cas),
        }


@dataclass(frozen=True, slots=True)
class WorkerArchiveResult:
    """The archive receipt: ``value`` the compaction's return (the active row count after it) and one annotation per replacement message."""

    value: int
    annotations: tuple[RowAnnotation, ...]


@dataclass(frozen=True, slots=True)
class WorkerCompressionPublication:
    """A ``compression.publish`` payload: the fourteen keys of ``publish_compression_child`` without the parent, which is the worker's session. :meth:`payload` sends all fourteen."""

    child_session_id: SessionId
    source: str
    messages: Sequence[WorkerMessage]
    model: str | None = None
    model_config: Mapping[str, JsonValue] | None = None
    system_prompt: str | None = None
    cwd: str | None = None
    profile_name: str | None = None
    compression_lock_holder: str | None = None
    require_compression_lease: bool = True
    require_lease_refresh: bool = False
    lease_ttl_seconds: float = 300.0
    watermark: int | None = None
    watermark_ceiling: int | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "child_session_id": self.child_session_id, "source": self.source, "messages": _rows(self.messages),
            "model": self.model, "model_config": None if self.model_config is None else dict(self.model_config),
            "system_prompt": self.system_prompt, "cwd": self.cwd, "profile_name": self.profile_name,
            "compression_lock_holder": self.compression_lock_holder, "require_compression_lease": self.require_compression_lease,
            "require_lease_refresh": self.require_lease_refresh, "lease_ttl_seconds": self.lease_ttl_seconds,
            "watermark": self.watermark, "watermark_ceiling": self.watermark_ceiling,
        }


@dataclass(frozen=True, slots=True)
class WorkerPublishResult:
    """The publication receipt: the worker's new assignment (``parent`` it left, ``session_id`` the child) and one annotation per handoff message."""

    parent: SessionId
    session_id: SessionId
    annotations: tuple[RowAnnotation, ...]


class WorkerCompactionVerbs(Protocol):
    """The worker's row-carrying compression writes on its session ``session_id``."""

    def compact_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, compaction: WorkerCompaction) -> WorkerArchiveResult:
        """``compression.archive``: the owner's in-place compaction (``archive_and_compact``) on the worker's session. Checks before it: the rows; ``carried_messages`` a list of dicts; ``covered_ids`` positive ints; ``unresolved_held`` and ``held_for_cas`` lists of dicts; both watermarks non-negative ints; a non-None ``lock_holder`` within the text bound; patch keys within the sidecar set; ``tail_count`` a non-negative int. A lost compression lease is :class:`persistence.CompressionInProgress` and a rewritten ``held_for_cas`` row :class:`persistence.HeldRowRewritten`; both commit nothing."""
        ...

    def publish_compression_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, publication: WorkerCompressionPublication) -> WorkerPublishResult:
        """``compression.publish``: ``child_session_id`` and ``compression_lock_holder`` within the text bound, both watermarks non-negative ints, ``require_compression_lease`` True, ``require_lease_refresh`` a bool, ``lease_ttl_seconds`` in ``[0.1, 3600]``, the rows; a non-None ``source``, ``cwd`` or ``profile_name`` unequal to the parent's, a lineage marker unequal to the parent's, or an existing child id is ``permission_denied``; no registered or running worker on the parent is ``stale_generation``. In one transaction: the owner's publication body (lease refresh and check, :class:`persistence.CompressionBusy`; the parent's closure, :class:`persistence.CompressionPublishRefused`; the child row and its rows), the goal, heartbeat and loop state moved to the child, the child's runtime generation set to the worker's, the worker moved onto the child, and its started or unknown admissions whose lineage ends at the parent extended by the child. An identical replay addressed to the parent after the move resolves through the recorded assignment."""
        ...

    def append_compression_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], compression_lock_holder: str | None, turn_lease_holder: str | None, turn_lease_ttl_seconds: float) -> WorkerAppendResult:
        """``compression.append``: the rows; non-None holders within the text bound; ``turn_lease_ttl_seconds`` in ``[0.1, 3600]``; then the transcript append under the compression lock and the turn lease (``TurnLeaseLost``, ``CompressionInProgress``, ``CompressionClosed``, ``CompressionBusy``). Backend lease reclaim: a stale compression lock or turn lease whose holder process is dead is deleted inside the append guard."""
        ...
