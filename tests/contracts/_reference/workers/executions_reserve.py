"""The reservation verbs of the worker reference: ``reserve_for_admission`` and ``lose_reservation`` and the receipted ``child.reserve``.

Mixed into ``InMemoryWorkerExecutionRepository``. A child reservation mints its session id with ``session_rules.session_ids.new_session_id``.
"""

from __future__ import annotations

import secrets
from typing import Any

from persistence.authority.vocabulary import OwnerRule, admission_fingerprint
from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import SessionId
from persistence.workers.delegations import WorkerDelegationPayload
from persistence.workers.executions import AdmissionWorkerScope, ChildExecution, ChildReservation, WorkerExecutionId
from persistence.workers.vocabulary import CHILD_SEPARATOR
from session_rules.legacy_local import same_owner
from session_rules.session_ids import new_session_id

from .._authority import authority_state, session_runtime
from ..authority import admissions_rows as rows
from . import receipt
from .receipt import Step, denied, refuse, secret_digest, stale_generation


def _child_handler(*, epoch: Any, execution_id: str, identity: dict[str, Any], principal: str, limits: tuple[int, int], refused: str | None) -> receipt.Handler:
    """The child-execution reservation handler with its trusted inputs bound."""
    def handle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
        secret = payload.get("secret") if isinstance(payload, dict) else None
        if set(payload) != {"secret"} or not isinstance(secret, str) or not 16 <= len(secret) <= 256:
            refuse("invalid_params")
        max_children, max_depth = limits
        prefix = execution_id + CHILD_SEPARATOR
        code = refused
        if code is None and prefix.count(CHILD_SEPARATOR) > max_depth:
            code = "delegation_depth_exceeded"
        if code is None and receipt.live_children(step.authority, execution_id) >= max_children:
            code = "delegation_capacity"
        if code is not None:
            return {"refused": code}
        parent = step.sessions[step.session_id]
        child_sid, child_eid = new_session_id(), prefix + secrets.token_hex(8)
        step.store.sessions.create(NewSession(session_id=SessionId(child_sid), source="subagent", parent_session_id=SessionId(step.session_id),
                                              lineage_kind=SessionLineageKind.DELEGATE, profile_name=parent["profile_name"]))
        scope = dict(identity, session_id=child_sid, execution_id=child_eid, generation=0, secret=secret)
        claim = admission_fingerprint(canonical_target=child_sid, payload=scope | {"principal": principal})
        session_runtime(step.authority, child_sid)
        step.authority["workers"][child_eid] = {
            "execution_id": child_eid, "session_id": child_sid, "kind": "child", "owner_epoch": epoch,
            "generation": 0, "status": "running", "secret_digest": secret_digest(claim), "last_sequence": 0,
        }
        return {"child": {"session_id": child_sid, "execution_id": child_eid, "generation": 0}}
    return handle


class ReservationVerbs:
    """Host: ``InMemoryWorkerExecutionRepository`` (``_backend``, ``_apply``)."""

    _backend: Any

    def reserve_for_admission(self, *, epoch: AuthorityEpoch, admission_id: Any, principal_id: str, profile_id: str, pid: int, birth: float, secret: str,
                              owner: OwnerRule | None, worker_subject: str | None = None) -> AdmissionWorkerScope:
        state = authority_state(self._backend, self._backend.scope)
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            admission = rows.admission(authority, admission_id)
            if admission["status"] != "started" or admission["owner_epoch"] != epoch:
                refuse("producer_not_started")
            if not same_owner(admission["principal_id"], principal_id, owner):
                raise denied("admission", admission_id)
            owner, generation = admission["target_session_id"], admission["generation"]
            rows.require_session(staged.scope, owner)
            if session_runtime(authority, owner)["generation"] != generation:
                raise stale_generation("session", owner)
            physical = rows.physical_target(self._backend, authority, staged.scope, owner)
            execution_id = "admission-worker:" + str(admission_id)
            if execution_id in authority["workers"]:
                refuse("admission_conflict")
            if any(row["session_id"] == physical and row["status"] != "terminal" for row in authority["workers"].values()):
                raise stale_generation("session", physical)
            session_runtime(authority, physical)["generation"] = generation
            if physical != owner:
                live = authority["admissions"].get(admission_id)
                if live is not None:
                    live["lineage"] = [*(live.get("lineage") or [owner]), physical]
            scope = dict(profile_id=profile_id, session_id=physical, execution_id=execution_id, generation=generation, pid=pid, birth=birth, secret=secret)
            claim = admission_fingerprint(canonical_target=physical, payload=scope | {"principal": worker_subject or principal_id})
            authority["workers"][execution_id] = {
                "execution_id": execution_id, "session_id": physical, "kind": "compute", "owner_epoch": epoch,
                "generation": generation, "status": "registered", "secret_digest": secret_digest(claim), "last_sequence": 0,
            }
            return AdmissionWorkerScope(**scope, epoch=epoch)

    def lose_reservation(self, *, epoch: AuthorityEpoch, admission_id: Any, execution_id: WorkerExecutionId, generation: ExecutionGeneration) -> None:
        state = authority_state(self._backend, self._backend.scope)
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            current = rows.admission(authority, admission_id)
            if current["status"] != "started" or current["owner_epoch"] != epoch or current["generation"] != generation:
                raise stale_generation("admission", admission_id)
            worker = authority["workers"].get(rows._bound_id(execution_id))
            if worker is None or worker["generation"] != generation or worker["owner_epoch"] != epoch:
                raise stale_generation("worker_execution", execution_id)
            worker["status"] = "terminal"
            receipt.compact_terminal_receipts(authority, staged.scope, worker["execution_id"])
            current["status"] = "unknown"
            rows.bump_revision(authority, current["target_session_id"])

    def reserve_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, secret: str | WorkerDelegationPayload, profile_id: str, pid: int, birth: float, principal: str, max_concurrent_children: int, max_spawn_depth: int, refused: str | None = None) -> ChildReservation:
        handler = _child_handler(epoch=epoch, execution_id=str(execution_id), identity={"profile_id": profile_id, "pid": pid, "birth": birth},
                                 principal=principal, limits=(max_concurrent_children, max_spawn_depth), refused=refused)
        payload = secret.payload if isinstance(secret, WorkerDelegationPayload) else {"secret": secret}
        result = self._apply("child.reserve", payload, handler, epoch=epoch, execution_id=execution_id,  # type: ignore[attr-defined]
                             session_id=session_id, generation=generation, sequence=sequence)
        child = result.get("child")
        if child is None:
            return ChildReservation(refused=result["refused"])
        return ChildReservation(child=ChildExecution(session_id=SessionId(child["session_id"]), execution_id=WorkerExecutionId(child["execution_id"]),
                                                     generation=ExecutionGeneration(child["generation"])))
