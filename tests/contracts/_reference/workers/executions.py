"""Reference implementation of ``WorkerExecutionRepository`` and the store's ``worker_executions`` repository.

``InMemoryWorkerExecutionRepository`` implements all seven worker protocols: the session, compression, compaction, read, delegation and receipt-read verbs live in their sibling modules as mixins, the reservations in :mod:`.executions_reserve`. Every receipted verb builds the wire payload and calls the one receipt transaction of :mod:`.receipt` with its handler; ``persist_operation`` takes the handler from :data:`OPERATION_HANDLERS`, the table of those same handlers by operation. Registration and adoption are their own write transactions.

Reference limits: the runtime lineage walk is not capped, so ``register`` never answers ``storage_unavailable`` for an overlong chain; ``usage.main`` with ``absolute`` sets the session counters instead of adding to them and records no attribution.
"""

from __future__ import annotations

import hmac
from typing import Any, Mapping

from persistence.core.errors import FenceStale, StaleFenceKind, UnknownExecution
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import JsonValue, SessionId
from persistence.workers.executions import WorkerExecutionId, WorkerExecutionRecord, WorkerKind, WorkerMainUsage, WorkerAuxiliaryUsage, WorkerStatus, WorkerTurnLease
from typing import Unpack

from .._authority import authority_state, session_runtime
from ..authority import admissions_rows as rows
from ..authority.coordination import _claim_lease_row, _holder_process_is_dead, conversation_key
from ..backend import register_repository
from . import compaction, compression, compression_reads, receipt, sessions
from .compaction import InMemoryWorkerCompactionVerbs
from .compression import InMemoryWorkerCompressionVerbs
from .compression_reads import InMemoryWorkerCompressionReads, authorize
from .delegations import InMemoryWorkerDelegationLedger
from .executions_reserve import ReservationVerbs
from .receipt import Step, denied, refuse, require_fields, secret_digest, stale_generation, text
from .receipts import InMemoryWorkerReceiptReads
from .sessions import InMemoryWorkerSessionVerbs
from .usage import TOKEN_COUNTERS, apply_auxiliary_usage, apply_main_usage, stamp_created_source

__all__ = ["OPERATION_HANDLERS", "WRITE_FAILPOINT", "InMemoryWorkerExecutionRepository", "record"]

WRITE_FAILPOINT = receipt.WRITE_FAILPOINT
_KINDS = tuple(kind.value for kind in WorkerKind)
#: The model-usage fields and the two usage operations' allowed keys.
_MODEL_USAGE_FIELDS = frozenset({
    "model", "billing_provider", "billing_base_url", "billing_mode", *TOKEN_COUNTERS, "estimated_cost_usd",
    "actual_cost_usd", "cost_status", "cost_source", "api_call_count",
})
_MAIN_USAGE_KEYS = _MODEL_USAGE_FIELDS | {"pricing_version", "absolute", "task"}
_AUX_USAGE_KEYS = (_MODEL_USAGE_FIELDS - {"billing_mode", "actual_cost_usd", "cost_status", "cost_source"}) | {"task"}


def record(row: Mapping[str, Any]) -> WorkerExecutionRecord:
    """The row without its digest."""
    return WorkerExecutionRecord(
        execution_id=WorkerExecutionId(row["execution_id"]), session_id=SessionId(row["session_id"]), kind=WorkerKind(row["kind"]),
        owner_epoch=AuthorityEpoch(row["owner_epoch"]), generation=ExecutionGeneration(row["generation"]),
        status=WorkerStatus(row["status"]), last_sequence=row["last_sequence"],
    )


def _valid_usage_value(key: str, value: Any) -> bool:
    """Whether ``value`` is within the bounds of usage key ``key``."""
    if key in (*TOKEN_COUNTERS, "api_call_count"):
        return type(value) is int and 0 <= value <= 2**53
    if key in ("estimated_cost_usd", "actual_cost_usd"):
        return value is None or (type(value) in (int, float) and 0 <= value <= 1e12)
    if key == "absolute":
        return type(value) is bool
    return value is None or (isinstance(value, str) and len(value) <= 4096)


def _usage_handler(auxiliary: bool) -> receipt.Handler:
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        source = None
        if not auxiliary:
            source = payload.get("source") if isinstance(payload.get("source"), str) else None
            payload = {key: value for key, value in payload.items() if key != "source"}
        if set(payload) - (_AUX_USAGE_KEYS if auxiliary else _MAIN_USAGE_KEYS):
            refuse("invalid_params")
        if not all(_valid_usage_value(key, value) for key, value in payload.items()):
            refuse("invalid_params")
        step.wrote = True
        if auxiliary:
            if not payload.get("task"):
                refuse("invalid_params")
            apply_auxiliary_usage(step.store, step.session_id, payload)
        else:
            apply_main_usage(step.store, step.session_id, payload)
            stamp_created_source(step.store, step.session_id, source)
        return {"value": None}
    return handle


def _turn_handler(operation: str) -> receipt.Handler:
    """The turn-lease handler for ``turn.acquire`` and ``turn.renew``."""
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        import time

        if set(payload) != {"holder", "ttl_seconds"}:
            refuse("invalid_params")
        holder = text(payload["holder"])
        ttl = payload.get("ttl_seconds", 300)
        if type(ttl) not in (float, int) or not 0.1 <= ttl <= 3600:
            refuse("invalid_params")
        key = conversation_key(step.store, step.scope, step.session_id)
        leases: dict[str, tuple[str, float]] = step.tables.setdefault("coord_turn_leases", {})
        now = time.time()
        if operation == "turn.acquire":
            return {"value": _claim_lease_row(leases, key, holder, now + ttl, lambda h, e: float(e) <= now or _holder_process_is_dead(h))}
        if leases.get(key, ("",))[0] != holder or key not in leases:
            return {"value": False}
        leases[key] = (holder, now + ttl)
        return {"value": True}
    return handle


def _turn_cleanup(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    """Release ``holder``'s turn lease on the conversation; ``target`` must share the worker session's conversation."""
    require_fields(payload, ("target", "holder"))
    target, holder = text(payload["target"]), text(payload["holder"])
    authorize(step, target)
    key = conversation_key(step.store, step.scope, step.session_id)
    if key != conversation_key(step.store, step.scope, target):
        raise denied("session", target)
    leases: dict[str, tuple[str, float]] = step.tables.setdefault("coord_turn_leases", {})
    if key in leases and leases[key][0] == holder:
        del leases[key]
    return {"released": True}


def _finish(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if payload:
        refuse("invalid_params")
    return {"status": "terminal"}


#: Every operation a typed worker verb runs, except ``child.reserve`` and the ``delegation.*`` ledger, with the handler that verb runs.
OPERATION_HANDLERS: dict[str, receipt.Handler] = {
    "session.title": sessions._title,
    "session.title_source": sessions._title_source,
    "session.next_title": sessions._next_title,
    "session.activity": sessions._activity,
    "session.activity_clear": sessions._activity_clear,
    "session.billing_route": sessions._billing_route,
    "session.api_content": sessions._api_content,
    "session.message_api_content": sessions._message_api_content,
    "session.user_content": sessions._user_content,
    "session.create": sessions._create,
    "session.end": sessions._end,
    "session.lifecycle": sessions._lifecycle,
    "session.prompt": sessions._prompt,
    "session.sidecars": sessions._sidecars,
    "session.tools": sessions._tools,
    "session.tail_role": sessions._tail_role,
    "session.row_role": sessions._row_role,
    "transcript.append": sessions._append,
    "turn.cleanup": _turn_cleanup,
    "compression.append": compaction._append,
    "compression.reopen": compression._reopen,
    "compression.cleanup": compression._cleanup,
    "compression.watermark": compression._watermark,
    "compression.archive": compaction._archive,
    "compression.publish": compaction._publish,
    "compression.context": compression_reads._context,
    "compression.lineage": compression_reads._lineage,
    "compression.boundary": compression._boundary,
    "compression.history": compression_reads._history,
    **{"compression.lock." + action: compression._lock(action) for action in ("acquire", "renew", "holder")},
    "compression.cooldown.record": compression._cooldown_record,
    "compression.cooldown.restore": compression._cooldown_restore,
    "compression.cooldown.clear": compression._cooldown_clear,
    "compression.fallback": compression._counter("compression_fallback_streak"),
    "compression.ineffective": compression._counter("compression_ineffective_count"),
    "compression.recovery": compression._counter("compression_recovery_deadline"),
    "execution.finish": _finish,
    "usage.main": _usage_handler(False),
    "usage.auxiliary": _usage_handler(True),
    "turn.acquire": _turn_handler("turn.acquire"),
    "turn.renew": _turn_handler("turn.renew"),
}


class InMemoryWorkerExecutionRepository(
    ReservationVerbs, InMemoryWorkerSessionVerbs, InMemoryWorkerCompressionVerbs, InMemoryWorkerCompactionVerbs,
    InMemoryWorkerCompressionReads, InMemoryWorkerDelegationLedger, InMemoryWorkerReceiptReads,
):
    """See ``persistence.workers.executions.WorkerExecutionRepository`` and its six sibling protocols."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def _apply(self, operation: str, payload: Any, handler: receipt.Handler, *, epoch: Any, execution_id: Any, session_id: Any, generation: Any, sequence: Any) -> dict[str, Any]:
        return receipt.apply(self._backend, operation, payload, handler, epoch=epoch, execution_id=execution_id,
                             session_id=session_id, generation=generation, sequence=sequence)

    # ── registration and adoption ─────────────────────────────────────────────

    def register(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, kind: WorkerKind, adoption_secret: str, require_idle: bool = False) -> WorkerExecutionRecord:
        for value in (execution_id, session_id):
            text(value)
        if kind not in _KINDS:
            refuse("invalid_params")
        digest = secret_digest(adoption_secret)
        state = authority_state(self._backend, self._backend.scope)
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            rows.require_session(staged.scope, session_id)
            if type(generation) is not int or session_runtime(authority, session_id)["generation"] != generation:
                raise stale_generation("session", session_id)
            old = authority["workers"].get(execution_id)
            if old is not None:
                if (old["session_id"], old["generation"], old["kind"], old["owner_epoch"], old["secret_digest"]) != (session_id, generation, kind, epoch, digest):
                    refuse("admission_conflict")
                return record(old)
            targets = rows.runtime_lineage(self._backend, authority, staged.scope, session_id)
            linked = [row for row in authority["admissions"].values() if row["target_session_id"] in targets]
            if require_idle and any(row["status"] in ("queued", "started", "unknown") for row in linked):
                raise stale_generation("session", session_id)
            if any(row["session_id"] in targets and row["status"] in ("registered", "running", "unknown") for row in authority["workers"].values()):
                raise stale_generation("session", session_id)
            if any(row["status"] == "unknown" for row in linked):
                raise UnknownExecution("admission", session_id)
            row = authority["workers"][execution_id] = {
                "execution_id": execution_id, "session_id": session_id, "kind": str(kind), "owner_epoch": epoch,
                "generation": generation, "status": "registered", "secret_digest": digest, "last_sequence": 0,
            }
            return record(row)

    def adopt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        digest = secret_digest(adoption_secret)
        state = authority_state(self._backend, self._backend.scope)
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            row = receipt.assignment(authority, staged.scope, execution_id, session_id, generation)
            if row["status"] == "terminal":
                raise stale_generation("worker_execution", execution_id)
            if not hmac.compare_digest(row["secret_digest"], digest):
                raise denied("worker_execution", execution_id)
            if str(execution_id).startswith("admission-worker:") and row["owner_epoch"] != epoch:
                raise FenceStale(StaleFenceKind.STALE_EPOCH, "worker_execution", str(execution_id))
            linked = [
                admission for admission in authority["admissions"].values()
                if admission["generation"] == generation and admission["owner_epoch"] == row["owner_epoch"]
                and admission["status"] in ("started", "unknown")
                and (admission["target_session_id"] == session_id or session_id in (admission.get("lineage") or ()))
            ]
            if len(linked) > 1:
                refuse("admission_conflict")
            if linked:
                linked[0].update(owner_epoch=epoch, status="started")
            row.update(owner_epoch=epoch, status="running")
            return record(row)

    def verify_claim(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        row = authority_state(self._backend, self._backend.scope).snapshot["workers"].get(rows._bound_id(execution_id))
        if row is None:
            refuse("not_found")
        if row["session_id"] != session_id:
            raise denied("worker_execution", execution_id)
        if type(generation) is not int or row["generation"] != generation:
            raise stale_generation("worker_execution", execution_id)
        if not hmac.compare_digest(row["secret_digest"], secret_digest(adoption_secret)):
            raise denied("worker_execution", execution_id)
        return record(row)

    def execution_session(self, *, execution_id: WorkerExecutionId) -> SessionId | None:
        row = authority_state(self._backend, self._backend.scope).snapshot["workers"].get(rows._bound_id(execution_id))
        return None if row is None else SessionId(row["session_id"])

    def has_live_worker(self) -> bool:
        return any(row["status"] != "terminal" for row in authority_state(self._backend, self._backend.scope).snapshot["workers"].values())

    # ── receipted execution operations ────────────────────────────────────────

    def persist_operation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, operation: str, payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if not isinstance(operation, str) or operation not in OPERATION_HANDLERS:
            refuse("invalid_params")
        return self._apply(operation, payload, OPERATION_HANDLERS[operation], epoch=epoch, execution_id=execution_id, session_id=session_id,
                           generation=generation, sequence=sequence)

    def finish(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._apply("execution.finish", {}, _finish, epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)

    def record_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, **usage: Unpack[WorkerMainUsage]) -> None:
        self._apply("usage.main", dict(usage), _usage_handler(False), epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)

    def record_auxiliary_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, task: str, **usage: Unpack[WorkerAuxiliaryUsage]) -> None:
        self._apply("usage.auxiliary", dict(usage, task=task), _usage_handler(True), epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)

    def acquire_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {"holder": lease.holder, "ttl_seconds": lease.ttl_seconds}
        return bool(self._apply("turn.acquire", payload, _turn_handler("turn.acquire"), epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)["value"])

    def renew_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {"holder": lease.holder, "ttl_seconds": lease.ttl_seconds}
        return bool(self._apply("turn.renew", payload, _turn_handler("turn.renew"), epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)["value"])

    def cleanup_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        self._apply("turn.cleanup", {"target": target_session_id, "holder": holder}, _turn_cleanup, epoch=epoch, execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)


register_repository("worker_executions", InMemoryWorkerExecutionRepository, {"receipted_writes": True})
