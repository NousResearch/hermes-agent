"""The receipt-stream reads the owner makes outside any receipted write, before it serves a worker request (gateway/session_worker.py:19-62).

Only a compression publication moves a worker to another session, and its receipt is the one that records the move (``worker_assignment``: the parent it left and the child it moved to). Both reads follow that record; neither takes an epoch nor writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol

from ..core.fences import ExecutionGeneration
from ..core.identity import JsonValue, SessionId

if TYPE_CHECKING:
    from .executions import WorkerExecutionId

__all__ = ["WorkerReceiptReads"]


class WorkerReceiptReads(Protocol):
    """The receipt-stream reads of the worker repository."""

    def claim_origin(self, *, execution_id: WorkerExecutionId, session_id: SessionId) -> SessionId:
        """The session the execution's claim is bound to: the parent recorded by its lowest-sequence publication receipt, else ``session_id`` unchanged. Nothing is validated."""
        ...

    def publication_target(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, operation: str, payload: Mapping[str, JsonValue]) -> SessionId:
        """The session a worker request addresses. Any ``operation`` other than ``compression.publish`` answers ``session_id`` without reading. For a publication, ``operation`` and ``payload`` are digest inputs only: when the receipt at ``sequence`` has the same digest and recorded a move from ``session_id``, the worker's current session is resolved instead; then the assignment check runs (no such execution ``not_found``, another session ``permission_denied``, a generation other than the row's or the session's ``stale_generation``) and the resolved session is returned."""
        ...
