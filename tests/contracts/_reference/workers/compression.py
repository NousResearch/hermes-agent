"""Reference verbs of the worker compression state (``persistence.workers.compression.WorkerCompressionRepository``).

Mixed into ``InMemoryWorkerExecutionRepository``. Each verb runs its handler inside the receipt transaction of :mod:`.receipt`. The compression lock is the coordination reference's ``coord_leases`` row keyed by the session id; the cooldown, counters, watermark, boundary and reopen go through the messages and lineage references' own verbs on the staged tables.
"""

from __future__ import annotations

import time
from typing import Any

from persistence.conversation.messages import ConversationPeer, CooldownSnapshot
from persistence.core.errors import CooldownRestoreRefused
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import SessionId
from persistence.workers.compression import WorkerCooldown, WorkerCooldownSnapshot
from persistence.workers.executions import WorkerExecutionId, WorkerTurnLease

from ..authority.coordination import _claim_lease_row, _holder_process_is_dead, conversation_key
from .compression_reads import authorize
from .receipt import Handler, Step, denied, number, refuse, require_fields, text

__all__ = ["InMemoryWorkerCompressionVerbs"]


def _locks(step: Step) -> dict[str, tuple[str, float]]:
    table: dict[str, tuple[str, float]] = step.tables.setdefault("coord_leases", {})
    return table


def _lock(action: str) -> Handler:
    """The compression-lock handler for ``acquire``, ``renew`` and ``holder``."""
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        locks, now = _locks(step), time.time()
        if action == "holder":
            require_fields(payload, ())
            row = locks.get(step.session_id)
            return {"value": row[0] if row is not None and row[1] > now else None}
        require_fields(payload, ("holder",), ("ttl_seconds",))
        holder = text(payload["holder"])
        ttl = number(payload.get("ttl_seconds", 300.0))
        if not 0.1 <= ttl <= 3600:
            refuse("invalid_params")
        if action == "acquire":
            return {"value": _claim_lease_row(locks, step.session_id, holder, now + ttl, lambda h, e: e < now or _holder_process_is_dead(h))}
        if locks.get(step.session_id, ("",))[0] != holder or step.session_id not in locks:
            return {"value": False}
        locks[step.session_id] = (holder, now + ttl)
        return {"value": True}
    return handle


def _cleanup(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ("target", "holder"))
    target, holder = text(payload["target"]), text(payload["holder"])
    authorize(step, target)
    if conversation_key(step.store, step.scope, step.session_id) != conversation_key(step.store, step.scope, target):
        raise denied("session", target)
    locks = _locks(step)
    if target in locks and locks[target][0] == holder:
        del locks[target]
    return {"value": None}


def _cooldown_record(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ("cooldown_until", "error"))
    deadline, error = number(payload["cooldown_until"]), payload["error"]
    if error is not None and not isinstance(error, str):
        refuse("invalid_params")
    step.store.messages.record_compression_cooldown(SessionId(step.session_id), deadline, error)
    step.wrote = True
    return {"value": None}


def _restore(step: Step, snapshot: Any) -> dict[str, Any]:
    require_fields(snapshot, ("session_exists", "cooldown_until", "error"))
    if type(snapshot["session_exists"]) is not bool:
        refuse("invalid_params")
    sid = SessionId(step.session_id)
    if not snapshot["session_exists"]:
        raise CooldownRestoreRefused(sid, "cannot restore absent compression cooldown row: session now exists")
    deadline, error = snapshot["cooldown_until"], snapshot["error"]
    if deadline is not None:
        number(deadline)
    if error is not None and not isinstance(error, str):
        refuse("invalid_params")
    step.store.messages.restore_compression_cooldown(sid, CooldownSnapshot(present=True, until=deadline, error=error))
    actual = step.store.messages.read_compression_cooldown_snapshot(sid)
    if (actual.until, actual.error) != (deadline, error):
        raise CooldownRestoreRefused(sid, "compression cooldown rollback verification failed")
    step.wrote = True
    return {"value": None}


def _cooldown_restore(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ("snapshot",))
    return _restore(step, payload["snapshot"])


def _cooldown_clear(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ())
    return _restore(step, {"session_exists": True, "cooldown_until": None, "error": None})


def _counter(column: str) -> Handler:
    """The counter write: the value written to the column as sent; the recovery deadline takes any finite number or ``None``, the two counters a non-negative int."""
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        require_fields(payload, ("value",))
        value = payload["value"]
        if value is not None:
            number(value)
        if column != "compression_recovery_deadline" and (type(value) is not int or value < 0):
            refuse("invalid_params")
        step.sessions[step.session_id][column] = value
        step.wrote = True
        return {"value": None}
    return handle


def _watermark(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ())
    return {"value": int(step.store.messages.active_watermark(SessionId(step.session_id)).last_active or 0)}


def _reopen(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ())
    return {"value": step.store.lineage.reopen_orphaned_compression(SessionId(step.session_id))}


def _boundary(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ("session_key", "source"))
    row = step.sessions[step.session_id]
    if (row["session_key"], row["source"]) != (payload["session_key"], payload["source"]):
        raise denied("session", step.session_id)
    return {"value": step.store.messages.latest_conversation_boundary(ConversationPeer(session_key=payload["session_key"], source=payload["source"]))}


class InMemoryWorkerCompressionVerbs:
    """Host: ``InMemoryWorkerExecutionRepository`` (``_apply``)."""

    def _compression_op(self, operation: str, payload: Any, handler: Handler, epoch: Any, execution_id: Any, session_id: Any, generation: Any, sequence: Any) -> Any:
        return self._apply(operation, payload, handler, epoch=epoch, execution_id=execution_id, session_id=session_id,  # type: ignore[attr-defined]
                           generation=generation, sequence=sequence)["value"]

    def acquire_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {"holder": lease.holder, "ttl_seconds": lease.ttl_seconds}
        return bool(self._compression_op("compression.lock.acquire", payload, _lock("acquire"), epoch, execution_id, session_id, generation, sequence))

    def renew_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {"holder": lease.holder, "ttl_seconds": lease.ttl_seconds}
        return bool(self._compression_op("compression.lock.renew", payload, _lock("renew"), epoch, execution_id, session_id, generation, sequence))

    def compression_lock_holder(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        return self._compression_op("compression.lock.holder", {}, _lock("holder"), epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def cleanup_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        self._compression_op("compression.cleanup", {"target": target_session_id, "holder": holder}, _cleanup, epoch, execution_id, session_id, generation, sequence)

    def record_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, cooldown: WorkerCooldown) -> None:
        payload = {"cooldown_until": cooldown.cooldown_until, "error": cooldown.error}
        self._compression_op("compression.cooldown.record", payload, _cooldown_record, epoch, execution_id, session_id, generation, sequence)

    def restore_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, snapshot: WorkerCooldownSnapshot) -> None:
        payload = {"snapshot": {"session_exists": snapshot.session_exists, "cooldown_until": snapshot.cooldown_until, "error": snapshot.error}}
        self._compression_op("compression.cooldown.restore", payload, _cooldown_restore, epoch, execution_id, session_id, generation, sequence)

    def clear_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._compression_op("compression.cooldown.clear", {}, _cooldown_clear, epoch, execution_id, session_id, generation, sequence)

    def set_compression_fallback_streak(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, streak: int) -> None:
        self._compression_op("compression.fallback", {"value": streak}, _counter("compression_fallback_streak"), epoch, execution_id, session_id, generation, sequence)

    def set_compression_ineffective_count(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, count: int) -> None:
        self._compression_op("compression.ineffective", {"value": count}, _counter("compression_ineffective_count"), epoch, execution_id, session_id, generation, sequence)

    def set_compression_recovery_deadline(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, deadline: float | None) -> None:
        self._compression_op("compression.recovery", {"value": deadline}, _counter("compression_recovery_deadline"), epoch, execution_id, session_id, generation, sequence)

    def compression_watermark(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> int:
        return int(self._compression_op("compression.watermark", {}, _watermark, epoch, execution_id, session_id, generation, sequence))

    def reopen_orphaned_compression(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> bool:
        return bool(self._compression_op("compression.reopen", {}, _reopen, epoch, execution_id, session_id, generation, sequence))

    def compression_boundary(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, peer: ConversationPeer) -> int | None:
        payload = {"session_key": peer.session_key, "source": peer.source}
        return self._compression_op("compression.boundary", payload, _boundary, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]
