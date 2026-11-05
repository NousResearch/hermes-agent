"""In-memory worker delegation ledger: the reference of :class:`persistence.workers.delegations.WorkerDelegationLedger`.

Mixed into ``InMemoryWorkerExecutionRepository``. Each receipted verb runs its handler inside the receipt transaction of :mod:`.receipt` on the scope table ``async_delegations`` the coordination reference serves (rows shaped by :func:`_dispatch_row`; ``owner_birth`` holds the owner's start time). Every verb checks the worker's identity (``execution_id``, ``worker_pid``, ``worker_birth``) before the receipt. The worker's process probe is the caller's: ``dispatch_delegation`` takes its outcome as ``owner_started_at`` and ``probe_refused``.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Literal, Mapping, Optional

from persistence.authority.coordination import (
    DELEGATION_CLAIM_STALE_SECONDS,
    DELEGATION_COMPLETION_STATUSES,
    DELEGATION_EVENT_KEYS,
    DELEGATION_MAX_DELIVERY_ATTEMPTS,
    DELEGATION_ORIGIN_SOURCES,
    DELEGATION_ROUTE_KEYS,
    DELEGATION_TASK_KEYS,
)
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import JsonValue, SessionId
from persistence.workers.delegations import (
    WorkerDelegationChild,
    WorkerDelegationClaim,
    WorkerDelegationCompletion,
    WorkerDelegationDispatch,
    WorkerDelegationPayload,
    WorkerDelegationView,
    delegation_payload,
)
from persistence.workers.executions import WorkerExecutionId

from .receipt import Handler, Step, denied, refuse

__all__ = ["InMemoryWorkerDelegationLedger", "_dispatch_row"]

_TABLE = "async_delegations"
_VIEW_FIELDS = ("delegation_id", "origin_session", "state", "dispatched_at", "completed_at", "delivery_state", "delivery_attempts", "origin_session_id")


def _dispatch_row(delegation_id: str, origin_session: str, origin_session_id: Optional[str], dispatched_at: float) -> dict[str, Any]:
    """The one row shape of ``async_delegations`` in this backend."""
    return {
        "delegation_id": delegation_id, "origin_session": origin_session,
        "origin_ui_session_id": "", "parent_session_id": None,
        "origin_session_id": origin_session_id or "", "owner_execution_id": None,
        "owner_pid": None, "owner_birth": None, "task": None,
        "state": "running", "dispatched_at": dispatched_at, "completed_at": None, "updated_at": time.time(),
        "event": None, "result": None, "children": None, "delivery_state": "pending", "delivery_attempts": 0,
        "delivery_claim": None, "delivery_claimed_at": None, "delivered_at": None,
    }


def _text(value: Any) -> str:
    """A non-empty string of at most 512 characters, else ``invalid_params``."""
    if not isinstance(value, str) or not value or len(value) > 512:
        refuse("invalid_params")
    return value  # type: ignore[no-any-return]


def _number(value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        refuse("invalid_params")


def _shape(payload: Any, keys: set[str]) -> None:
    if not isinstance(payload, dict) or set(payload) != keys:
        refuse("invalid_params")


def _ledger(step: Step) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = step.tables.setdefault(_TABLE, {})
    return table


def _owned(table: Mapping[str, dict[str, Any]], session_id: str, execution_id: str, delegation_id: Any) -> dict[str, Any]:
    """The ledger row this session and execution own: ``not_found`` when missing, denied when another owns it."""
    row = table.get(_text(delegation_id))
    if row is None:
        refuse("not_found")
    if row["parent_session_id"] != session_id or row["owner_execution_id"] != execution_id:
        raise denied("delegation", delegation_id)
    return row


def _dispatch(*, worker_pid: int, owner_started_at: Optional[int], probe_refused: Optional[str]) -> Handler:
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        _shape(payload, {"delegation_id", "task", "dispatched_at", *DELEGATION_ROUTE_KEYS})
        _text(payload["delegation_id"])
        _number(payload["dispatched_at"])
        if not isinstance(payload["task"], dict) or set(payload["task"]) - DELEGATION_TASK_KEYS:
            refuse("invalid_params")
        sid, eid = step.session_id, step.worker["execution_id"]
        session = step.sessions.get(sid)
        if session is None or session["source"] not in DELEGATION_ORIGIN_SOURCES:
            refuse("unsupported_producer")
        if (payload["parent_session_id"] != sid or payload["session_key"] != (session["session_key"] or "")
                or payload["origin_ui_session_id"] not in ("", sid) or payload["origin_session_id"] not in ("", sid)):
            raise denied("delegation", payload["delegation_id"])
        table = _ledger(step)
        if payload["delegation_id"] in table:
            _owned(table, sid, eid, payload["delegation_id"])
            refuse("admission_conflict")
        if probe_refused == "permission_denied":
            raise denied("worker_execution", eid)
        if probe_refused is not None:
            refuse(probe_refused)
        row = _dispatch_row(payload["delegation_id"], payload["session_key"], payload["origin_session_id"], payload["dispatched_at"])
        row.update(origin_ui_session_id=payload["origin_ui_session_id"], parent_session_id=sid, owner_execution_id=eid,
                   owner_pid=worker_pid, owner_birth=owner_started_at, task=payload["task"])
        table[payload["delegation_id"]] = row
        return {"value": None}
    return handle


def _complete(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    _shape(payload, {"event", "result"})
    event = payload["event"]
    if (not isinstance(event, dict) or set(event) - DELEGATION_EVENT_KEYS or not isinstance(payload["result"], dict)
            or event.get("type") != "async_delegation" or event.get("status") not in DELEGATION_COMPLETION_STATUSES):
        refuse("invalid_params")
    row = _owned(_ledger(step), step.session_id, step.worker["execution_id"], event.get("delegation_id"))
    if any(event.get(key, "") != (row[column] or "") for key, column in DELEGATION_ROUTE_KEYS.items()):
        raise denied("delegation", row["delegation_id"])
    _number(event.get("completed_at"))
    if row["state"] != "running":
        refuse("admission_conflict")
    row.update(state=event["status"], completed_at=event["completed_at"], updated_at=time.time(), event=event, result=payload["result"])
    return {"value": None}


def _child(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    _shape(payload, {"delegation_id", "entry"})
    row = _owned(_ledger(step), step.session_id, step.worker["execution_id"], payload["delegation_id"])
    entry = payload["entry"]
    if not isinstance(entry, dict) or type(entry.get("task_index")) is not int or entry["task_index"] < 0:
        refuse("invalid_params")
    if row["state"] != "running":
        return {"value": None}
    previous = row["result"] or {}
    results = [item for item in previous.get("results", []) if item.get("task_index") != entry["task_index"]]
    row.update(result={"results": [*results, entry], "partial": True}, updated_at=time.time())
    return {"value": None}


def _read(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    _shape(payload, {"delegation_id"})
    row = _owned(_ledger(step), step.session_id, step.worker["execution_id"], payload["delegation_id"])
    return {"value": {key: row[key] for key in _VIEW_FIELDS} | {"result": copy.deepcopy(row["result"])}}


def _claim(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    _shape(payload, {"delegation_id", "claim_id"})
    row = _owned(_ledger(step), step.session_id, step.worker["execution_id"], payload["delegation_id"])
    _text(payload["claim_id"])
    now = time.time()
    if row["state"] in ("running", "finalizing"):
        return {"value": False}
    if row["delivery_state"] != "pending" or not (row["delivery_claim"] is None or row["delivery_claimed_at"] < now - DELEGATION_CLAIM_STALE_SECONDS):
        return {"value": False}
    row.update(delivery_claim=payload["claim_id"], delivery_claimed_at=now, delivery_attempts=row["delivery_attempts"] + 1, updated_at=now)
    return {"value": True}


def _ack(row: dict[str, Any], now: float) -> None:
    row.update(delivery_state="delivered", delivered_at=now)


def _drop(row: dict[str, Any], now: float) -> None:
    row["delivery_state"] = "dropped"


def _defer(row: dict[str, Any], now: float) -> None:
    row["delivery_attempts"] = max(0, row["delivery_attempts"] - 1)


def _release(row: dict[str, Any], now: float) -> None:
    if row["delivery_attempts"] >= DELEGATION_MAX_DELIVERY_ATTEMPTS:
        row["delivery_state"] = "dropped"


_DELIVERY = {"ack": _ack, "drop": _drop, "defer": _defer, "release": _release}


def _delivery(action: str) -> Handler:
    """For a pending delivery this token holds, the action, then the claim released."""
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        _shape(payload, {"delegation_id", "claim_id"})
        row = _owned(_ledger(step), step.session_id, step.worker["execution_id"], payload["delegation_id"])
        _text(payload["claim_id"])
        if row["delivery_state"] != "pending" or row["delivery_claim"] != payload["claim_id"]:
            return {"value": False}
        now = time.time()
        _DELIVERY[action](row, now)
        row.update(updated_at=now, delivery_claim=None, delivery_claimed_at=None)
        return {"value": True}
    return handle


def _unscheduled(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    _shape(payload, {"delegation_id"})
    table = _ledger(step)
    row = _owned(table, step.session_id, step.worker["execution_id"], payload["delegation_id"])
    if row["state"] != "running" or row["result"] is not None:
        refuse("admission_conflict")
    del table[row["delegation_id"]]
    return {"value": None}


class InMemoryWorkerDelegationLedger:
    """Host: ``InMemoryWorkerExecutionRepository`` (``_backend``, ``_apply``)."""

    _backend: Any

    def _ledger_op(self, operation: str, payload: Any, handler: Handler, epoch: Any, execution_id: Any, session_id: Any, generation: Any, sequence: Any, worker_pid: Any, worker_birth: Any) -> Any:
        _text(execution_id)
        if type(worker_pid) is not int or worker_pid <= 0:
            refuse("invalid_params")
        _number(worker_birth)
        return self._apply(operation, payload, handler, epoch=epoch, execution_id=execution_id, session_id=session_id,  # type: ignore[attr-defined]
                           generation=generation, sequence=sequence)["value"]

    def dispatch_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, dispatch: WorkerDelegationDispatch | WorkerDelegationPayload, worker_pid: int, worker_birth: float, owner_started_at: int | None, probe_refused: Literal['permission_denied', 'worker_not_live'] | None = None) -> None:
        handler = _dispatch(worker_pid=worker_pid, owner_started_at=owner_started_at, probe_refused=probe_refused)
        self._ledger_op("delegation.dispatch", delegation_payload(dispatch), handler, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def complete_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, completion: WorkerDelegationCompletion | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger_op("delegation.complete", delegation_payload(completion), _complete, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def record_delegation_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, child: WorkerDelegationChild | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger_op("delegation.child", delegation_payload(child), _child, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def read_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> WorkerDelegationView:
        return WorkerDelegationView(**self._ledger_op("delegation.read", delegation_payload(delegation_id), _read, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth))

    def _delivery_op(self, action: str, claim: WorkerDelegationClaim | WorkerDelegationPayload, *identity: Any) -> bool:
        handler = _claim if action == "claim" else _delivery(action)
        return bool(self._ledger_op("delegation." + action, delegation_payload(claim), handler, *identity))

    def claim_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery_op("claim", claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def ack_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery_op("ack", claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def drop_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery_op("drop", claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def defer_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery_op("defer", claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def release_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery_op("release", claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def withdraw_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger_op("delegation.unscheduled", delegation_payload(delegation_id), _unscheduled, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def pending_delegation_event(self, *, execution_id: WorkerExecutionId, session_id: SessionId, delegation_id: str) -> Mapping[str, JsonValue] | None:
        table = self._backend.scope_store(self._backend.scope).get(_TABLE, {})
        row = _owned(table, str(session_id), str(execution_id), delegation_id)
        return copy.deepcopy(row["event"]) if row["delivery_state"] == "pending" else None
