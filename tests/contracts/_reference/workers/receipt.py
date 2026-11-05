"""The worker receipt transaction of the reference and the checks its operations share.

:func:`apply` is the one dispatcher every receipted worker verb calls, and it is private to this package. It runs these steps in this order inside one ``AuthorityState.transaction``: the payload's encoding and size, then the epoch fence, the receipt-aware assignment, the owner-epoch fence, the stored receipt (replay, conflict or retired), the live status, the strict next sequence, the operation, and the receipt write with the sequence advance, the status and the revision bump. The operation runs with the scope slot pointed at the transaction's staged tables (:func:`staged_scope`), so the sibling repositories it calls write the staged copy and publish with the receipt or not at all.

Execution rows live in the authority state's ``workers`` table keyed by execution id (``secret_digest`` holds ``adoption_digest``); receipts in ``receipts`` keyed ``(execution_id, sequence)`` as ``{"digest", "result"}`` with the stored result dict.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, NoReturn

from persistence.authority.vocabulary import admission_fingerprint
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, FenceStale, PermanentStoreError, ProofRejected, StaleFenceKind
from persistence.workers.vocabulary import CHILD_SEPARATOR, RECEIPT_PAYLOAD_MAX_BYTES

from .._authority import _AUTHORITY_KEY, Staged, authority_state, session_runtime
from ..authority.admissions_rows import _bound_id

__all__ = ["WRITE_FAILPOINT", "Step", "apply", "assignment", "compact_terminal_receipts", "staged_scope"]

#: The failpoint between an operation's staged write and its receipt; arming it proves neither-or-both.
WRITE_FAILPOINT = "worker_executions.between-write-and-receipt"
#: What a read receipt shrinks to when its execution ends.
RETIRED_RECEIPT = {"retired": True}
#: The operations whose handler writes nothing, so they never count as a change; the sibling reads they call may still materialize a default row in this backend, which is not a write.
READ_ONLY = frozenset({
    "compression.context", "compression.lineage", "compression.history", "compression.watermark", "compression.lock.holder",
    "compression.boundary", "session.lifecycle", "session.tail_role", "session.row_role", "session.next_title", "delegation.read",
})


class Step:
    """One operation's view inside the receipt transaction: the store handle, the staged tables, the addressed session and the execution row. ``wrote`` marks a write that matched a row, even when it left the values as they were (it still counts as a change)."""

    __slots__ = ("session_id", "staged", "store", "worker", "wrote")

    def __init__(self, store: Any, staged: Staged, session_id: str, worker: dict[str, Any]) -> None:
        self.store, self.staged, self.session_id, self.worker, self.wrote = store, staged, session_id, worker, False

    @property
    def tables(self) -> dict[str, Any]:
        """The live scope store, which holds the staged tables while the operation runs (:func:`staged_scope`)."""
        table: dict[str, Any] = self.store.scope_store(self.store.scope)
        return table

    @property
    def authority(self) -> dict[str, Any]:
        return self.staged.authority

    @property
    def sessions(self) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = self.tables.setdefault("sessions", {})
        return table

    @property
    def scope(self) -> Any:
        return self.store.scope


def refuse(reason: str) -> NoReturn:
    raise AuthorityRefusal(reason)


def stale_generation(entity: str, entity_id: object) -> FenceStale:
    return FenceStale(StaleFenceKind.STALE_GENERATION, entity, str(entity_id))


def denied(entity: str, entity_id: object) -> ProofRejected:
    return ProofRejected(entity, str(entity_id))


def text(value: Any) -> str:
    """A non-empty string of at most 1024 characters, else ``invalid_params``."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        refuse("invalid_params")
    return value


def number(value: Any) -> float:
    """A finite int or float, else ``invalid_params``."""
    if type(value) not in (int, float) or not math.isfinite(value):
        refuse("invalid_params")
    return value  # type: ignore[no-any-return]


def require_fields(payload: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    """Exactly the required keys plus any of the optional ones."""
    if not isinstance(payload, dict) or set(payload) - set(required) - set(optional) or set(required) - set(payload):
        refuse("invalid_params")


def secret_digest(secret: Any) -> str:
    """The SHA-256 hex digest of the secret text."""
    return hashlib.sha256(text(secret).encode("utf-8")).hexdigest()


def worker_row(authority: Mapping[str, Any], execution_id: Any) -> dict[str, Any] | None:
    """The execution row of ``execution_id``: the id bound as SQLite binds it, an unbindable id ``PermanentStoreError``."""
    try:
        return authority["workers"].get(_bound_id(execution_id))  # type: ignore[no-any-return]
    except sqlite3.ProgrammingError as exc:
        raise PermanentStoreError(str(exc)) from exc


def assignment(authority: Mapping[str, Any], tables: Mapping[str, Any], execution_id: Any, session_id: Any, generation: Any) -> dict[str, Any]:
    """The execution row assigned to ``session_id`` at ``generation``: ``not_found``, denied for another session, else a stale generation unless the row's and the session's runtime generation both match."""
    row = worker_row(authority, execution_id)
    if row is None:
        refuse("not_found")
    if row["session_id"] != session_id:
        raise denied("worker_execution", execution_id)
    if type(generation) is not int or row["generation"] != generation or _runtime_generation(authority, tables, session_id) != generation:
        raise stale_generation("worker_execution", execution_id)
    return row  # type: ignore[no-any-return]


def _runtime_generation(authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str) -> int:
    if session_id not in tables.get("sessions", {}):
        refuse("not_found")
    return int(authority["sessions"].get(session_id, {}).get("generation", 0))


def receipt_assignment(authority: Mapping[str, Any], tables: Mapping[str, Any], execution_id: Any, session_id: Any, generation: Any, sequence: int, digest: str) -> dict[str, Any]:
    """The receipt-aware assignment: only an identical committed publication may still address the parent it left."""
    receipt = authority["receipts"].get((_bound_id(execution_id), sequence))
    if receipt is not None and receipt["digest"] == digest and isinstance(receipt["result"], dict):
        moved = receipt["result"].get("worker_assignment")
        if moved and moved["parent"] == session_id:
            current = worker_row(authority, execution_id)["session_id"]  # type: ignore[index]
            return assignment(authority, tables, execution_id, current, generation)
    return assignment(authority, tables, execution_id, session_id, generation)


def payload_digest(operation: str, payload: Any, session_id: Any) -> tuple[dict[str, Any], str]:
    """The payload as the receipt sees it (its canonical JSON decoded) and its receipt digest; an unencodable or oversized payload is ``invalid_params``."""
    encoded = canonical_json(payload)
    if len(encoded.encode("utf-8", errors="surrogatepass")) > RECEIPT_PAYLOAD_MAX_BYTES:
        refuse("invalid_params")
    decoded = json.loads(encoded)
    return decoded, admission_fingerprint(canonical_target=session_id, payload={"operation": operation, "payload": decoded})


@contextmanager
def staged_scope(store: Any, staged_tables: dict[str, Any]) -> Iterator[None]:
    """Point the scope slot sibling repositories read at the transaction's staged tables for one operation, then fold every table the operation touched back into the staged copy; the committed tables are restored either way. Sound because every repository method runs under the backend-wide lock."""
    live = store.scope_store(store.scope)
    committed = dict(live)
    live.clear()
    live.update(staged_tables)
    live[_AUTHORITY_KEY] = committed[_AUTHORITY_KEY]
    try:
        yield
    finally:
        staged_tables.clear()
        staged_tables.update({key: value for key, value in live.items() if key != _AUTHORITY_KEY})
        live.clear()
        live.update(committed)


Handler = Callable[[Step, dict[str, Any]], dict[str, Any]]


def apply(store: Any, operation: str, payload: Any, handler: Handler, *, epoch: Any, execution_id: Any, session_id: Any, generation: Any, sequence: Any) -> dict[str, Any]:
    """The stored result of an exact retry, else ``handler``'s result committed with its receipt. A replay returns a detached copy of the stored result."""
    if type(sequence) is not int or sequence < 1:
        refuse("invalid_params")
    decoded, digest = payload_digest(operation, payload, session_id)
    state = authority_state(store, store.scope)
    with state.transaction() as staged:
        authority = staged.authority
        state.require_epoch(authority, epoch)
        worker = receipt_assignment(authority, staged.scope, execution_id, session_id, generation, sequence, digest)
        if worker["owner_epoch"] != epoch:
            raise FenceStale(StaleFenceKind.STALE_EPOCH, "worker_execution", str(execution_id))
        old = authority["receipts"].get((worker["execution_id"], sequence))
        if old is not None:
            if old["digest"] != digest:
                refuse("admission_conflict")
            if old["result"] == RETIRED_RECEIPT:
                raise stale_generation("worker_receipt", execution_id)
            return copy.deepcopy(old["result"])  # type: ignore[no-any-return]
        if worker["status"] not in ("registered", "running"):
            raise stale_generation("worker_execution", execution_id)
        if sequence != worker["last_sequence"] + 1:
            refuse("invalid_params")
        step = Step(store, staged, session_id, worker)
        before = copy.deepcopy((staged.scope, authority))
        with staged_scope(store, staged.scope):
            result = handler(step, decoded)
        changed = operation not in READ_ONLY and (step.wrote or (staged.scope, authority) != before) or operation == "execution.finish"
        state.failpoints.hit(WRITE_FAILPOINT)
        authority["receipts"][(worker["execution_id"], sequence)] = {"digest": digest, "result": json.loads(json.dumps(result, ensure_ascii=True, allow_nan=False))}
        worker["last_sequence"] = sequence
        worker["status"] = "terminal" if operation == "execution.finish" else "running"
        if changed:
            session_runtime(authority, session_id)["revision"] += 1
        if operation == "execution.finish":
            compact_terminal_receipts(authority, staged.scope, worker["execution_id"])
        return copy.deepcopy(result)


def compact_terminal_receipts(authority: dict[str, Any], tables: dict[str, Any], execution_id: str) -> None:
    """Retire the live descendants, shrink this execution's read receipts (results carrying ``messages`` or ``session``) to the retired marker, and end a child's own session row."""
    low = execution_id + CHILD_SEPARATOR
    for child in [row for eid, row in authority["workers"].items() if eid.startswith(low) and row["status"] != "terminal"]:
        child["status"] = "terminal"
        _compact(authority, child["execution_id"])
        _end_child_row(authority, tables, child["execution_id"])
    _compact(authority, execution_id)
    if CHILD_SEPARATOR in execution_id:
        _end_child_row(authority, tables, execution_id)


def _compact(authority: dict[str, Any], execution_id: str) -> None:
    for (eid, _sequence), receipt in authority["receipts"].items():
        result = receipt["result"]
        if eid == execution_id and isinstance(result, dict) and ("messages" in result or "session" in result):
            receipt["result"] = dict(RETIRED_RECEIPT)


def _end_child_row(authority: Mapping[str, Any], tables: dict[str, Any], execution_id: str) -> None:
    """The execution's session row ends ``agent_close`` unless it already ended."""
    import time

    worker = authority["workers"].get(execution_id)
    row = tables.get("sessions", {}).get(worker["session_id"]) if worker is not None else None
    if row is not None and row["ended_at"] is None:
        row["ended_at"] = time.time()
        row["end_reason"] = "agent_close"


def live_children(authority: Mapping[str, Any], execution_id: str) -> int:
    """The live direct children of an execution."""
    prefix = execution_id + CHILD_SEPARATOR
    return sum(1 for eid, row in authority["workers"].items() if eid.startswith(prefix) and "/" not in eid[len(prefix):] and row["status"] != "terminal")
