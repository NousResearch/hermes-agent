"""Reference worker receipt reads (``persistence.workers.receipts.WorkerReceiptReads``).

Mixed into ``InMemoryWorkerExecutionRepository``. Both read the committed authority state: ``claim_origin`` and ``publication_target`` read through the receipt-aware assignment of :mod:`.receipt`.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.fences import ExecutionGeneration
from persistence.core.identity import JsonValue, SessionId
from persistence.workers.executions import WorkerExecutionId

from .._authority import authority_state
from . import receipt

__all__ = ["InMemoryWorkerReceiptReads"]


class InMemoryWorkerReceiptReads:
    """Host: ``InMemoryWorkerExecutionRepository`` (``_backend``)."""

    _backend: Any

    def claim_origin(self, *, execution_id: WorkerExecutionId, session_id: SessionId) -> SessionId:
        receipts = authority_state(self._backend, self._backend.scope).snapshot["receipts"]
        moves = sorted((sequence, stored["result"]["worker_assignment"]["parent"]) for (eid, sequence), stored in receipts.items()
                       if eid == execution_id and isinstance(stored["result"], dict) and (stored["result"].get("worker_assignment") or {}).get("parent") is not None)
        return SessionId(moves[0][1]) if moves else session_id

    def publication_target(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, operation: str, payload: Mapping[str, JsonValue]) -> SessionId:
        if operation != "compression.publish":
            return session_id
        digest = admission_fingerprint(canonical_target=session_id, payload={"operation": operation, "payload": json.loads(json.dumps(payload))})
        state = authority_state(self._backend, self._backend.scope)
        tables = self._backend.scope_store(self._backend.scope)
        row = receipt.receipt_assignment(state.snapshot, tables, execution_id, session_id, generation, sequence, digest)
        return SessionId(row["session_id"])
