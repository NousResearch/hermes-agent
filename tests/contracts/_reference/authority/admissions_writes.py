"""The admission lifecycle writes of the reference after ``admit``: claim, return, settle, cancel, steer, unknown marking, rehoming, recovery and resolution.

Each verb is one ``AuthorityState.transaction``: the authority state and the sibling tables (sessions, messages) are staged together, so the discarded-turn closer and the stored result publish with the terminal write or not at all. :data:`CLAIM_FAILPOINT` sits between the generation bump and the row start; :data:`SETTLE_FAILPOINT` between worker retirement and the terminal writes.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping

from persistence.authority.admissions import ANY_HEAD, AdmissionRecord, CapturedResult, ClaimedAdmission, GateRefusal, TurnBoundary, steer_replayed
from persistence.authority.vocabulary import STEERING
from persistence.core.errors import AuthorityRefusal

from .._authority import session_runtime
from . import admissions_rows as rows
from .admissions_results import close_discarded_turn, compact_result, encode_result

#: Failpoint between claim's generation bump and the admission start.
CLAIM_FAILPOINT = "admissions.between-generation-and-start"
#: Failpoint between worker retirement and the terminal writes in settle and resolve_unknown.
SETTLE_FAILPOINT = "admissions.between-retire-and-terminal"


class AdmissionWritesMixin:
    """The lifecycle writes; mixed into ``InMemoryAdmissionRepository``."""

    def claim(self, *, epoch: int, session_id: str, gate: GateRefusal | None = None, head: Any = ANY_HEAD) -> ClaimedAdmission | Literal[False] | None:
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            if gate is not None:
                raise gate.refusal()
            rows.require_session(staged.scope, session_id)
            session_rows = rows.session_rows(authority, session_id)
            blocked = [row for row in session_rows if row["status"] in ("started", "unknown")]
            if any(row["status"] == "unknown" for row in blocked):
                raise rows.unknown("admission", session_id)
            workers = self._worker_states(authority, staged.scope, rows._bound_id(session_id))
            if "unknown" in workers:
                raise rows.unknown("worker_execution", session_id)
            if blocked or workers:
                return None
            first = next((row for row in session_rows if row["status"] == "queued"), None)
            if head is not ANY_HEAD and (first["admission_id"] if first is not None else None) != head:
                return False
            if first is None:
                return None
            entry = session_runtime(authority, rows._bound_id(session_id))
            queued_owner_epoch = first["owner_epoch"]
            entry["generation"] += 1
            state.failpoints.hit(CLAIM_FAILPOINT)
            first.update(status="started", owner_epoch=epoch, generation=entry["generation"])
            return rows.claimed(first, rows.max_message_id(staged.scope), queued_owner_epoch)

    def return_claim(self, *, epoch: int, admission_id: str, generation: int, queued_owner_epoch: int | None) -> AdmissionRecord:
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            row = rows.admission(authority, admission_id)
            if row["status"] == "queued" and row["generation"] is None:
                return rows.record(row)
            target = row["target_session_id"]
            targets = {target, (row.get("lineage") or [target])[-1]}
            bound_worker = any(w["owner_epoch"] == epoch and w["generation"] == generation and w["session_id"] in targets for w in authority["workers"].values())
            bound_steer = any(other["status"] == "terminal" and other["outcome"] == STEERING and other["owner_epoch"] == epoch and other["generation"] == generation
                              and other["target_session_id"] == target for other in authority["admissions"].values())
            if (row["status"] != "started" or row["owner_epoch"] != epoch or type(generation) is not int or row["generation"] != generation
                    or session_runtime(authority, target)["generation"] != generation or bound_worker or bound_steer):
                raise rows.stale("admission", admission_id)
            row.update(status="queued", owner_epoch=queued_owner_epoch, generation=None)
            session_runtime(authority, target)["generation"] = generation - 1
            return rows.record(row)

    def settle(self, *, epoch: int, admission_id: str, generation: int, outcome: str, result: Mapping[str, Any] | None = None,
               transfers: Mapping[str, bool] | None = None, boundary: TurnBoundary | None = None) -> AdmissionRecord:
        if outcome not in ("completed", "interrupted", "rejected", "failed"):
            raise AuthorityRefusal("invalid_params")
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            row = rows.admission(authority, admission_id)
            target = row["target_session_id"]
            rows.require_session(staged.scope, target)
            if (row["status"] != "started" or row["owner_epoch"] != epoch or type(generation) is not int or row["generation"] != generation
                    or session_runtime(authority, target)["generation"] != generation):
                raise rows.stale("admission", admission_id)
            rows.retire_workers(authority, row, epoch)
            state.failpoints.hit(SETTLE_FAILPOINT)
            if boundary is not None:
                close_discarded_turn(staged.scope, self._turn_target(authority, staged.scope, target), boundary.notice,
                                     replace_after=boundary.replace_after, display_metadata=boundary.display_metadata)
            if result is not None:
                self._store_result(authority, row, result)
            row.update(status="terminal", outcome=str(outcome))
            rows.settle_transfers(authority, epoch=epoch, generation=generation, transfers=transfers, outcome=outcome)
            rows.bump_revision(authority, target)
            return rows.record(row)

    @staticmethod
    def _store_result(authority: dict[str, Any], row: Mapping[str, Any], result: Any) -> None:
        user_message = row["payload"].get("text") if isinstance(row["payload"], dict) else None
        authority["meta"][rows._result_key(row["admission_id"])] = rows.encode(encode_result(compact_result(result, user_message=user_message)))

    def cancel(self, *, epoch: int, admission_id: str) -> AdmissionRecord:
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            row = rows.admission(authority, admission_id)
            if row["status"] == "unknown":
                raise rows.unknown("admission", admission_id)
            if row["status"] == "started" or row["outcome"] == STEERING:
                raise rows.stale("admission", admission_id)
            if row["status"] == "queued":
                row.update(status="terminal", outcome="cancelled")
            return rows.record(row)

    def steer(self, *, epoch: int, admission_id: str, generation: int) -> tuple[AdmissionRecord, bool]:
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            row = rows.admission(authority, admission_id)
            if steer_replayed(row):
                return rows.record(row), False
            if row["status"] != "queued":
                raise AuthorityRefusal("admission_started")
            if type(generation) is not int or not any(
                    other["target_session_id"] == row["target_session_id"] and other["status"] == "started" and other["owner_epoch"] == epoch
                    and other["generation"] == generation for other in authority["admissions"].values()):
                raise AuthorityRefusal("session_idle")
            row.update(status="terminal", outcome=STEERING, owner_epoch=epoch, generation=generation)
            return rows.record(row), True

    def revert_steer(self, *, epoch: int, admission_id: str, generation: int) -> bool:
        state = self._state()
        with state.transaction() as staged:
            state.require_epoch(staged.authority, epoch)
            return rows.requeue_transfer(staged.authority, admission_id, epoch, generation)

    def mark_unknown(self, *, epoch: int, admission_id: str, generation: int, transfers: Mapping[str, bool] | None = None) -> str:
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            row = rows.admission(authority, admission_id)
            if row["owner_epoch"] != epoch or row["generation"] != generation:
                raise rows.stale("admission", admission_id)
            if row["status"] in {"terminal", "unknown"}:
                return row["status"]
            if row["status"] != "started":
                raise rows.stale("admission", admission_id)
            row["status"] = "unknown"
            rows.settle_transfers(authority, epoch=epoch, generation=generation, transfers=transfers, outcome="unknown")
            rows.bump_revision(authority, row["target_session_id"])
            return "unknown"

    def rehome_queued(self, *, epoch: int, old_session_id: str, new_session_id: str, principal_prefix: str) -> list[str]:
        for value in (old_session_id, new_session_id, principal_prefix):
            rows.text(value)
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            rows.require_session(staged.scope, new_session_id)
            old_rows = rows.session_rows(authority, old_session_id)
            if any(row["status"] == "unknown" for row in old_rows):
                return []
            moved = [row for row in old_rows if row["status"] == "queued" and row["principal_id"][:len(principal_prefix)] == principal_prefix]
            for row in moved:
                row.update(target_session_id=new_session_id, lineage=[new_session_id], digest=rows.digest(new_session_id, row["payload"], row["intent"]))
            if moved:
                for sid in (old_session_id, new_session_id):
                    rows.bump_revision(authority, sid)
            return [row["admission_id"] for row in moved]

    def retarget_local_automation(self, *, epoch: int, session_id: str, principal_id: str, old_targets: Iterable[str], new_target: str) -> list[str]:
        old_targets = list(old_targets)
        for value in (session_id, principal_id, new_target, *old_targets):
            rows.text(value)
        targets = set(old_targets)
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            rows.require_session(staged.scope, new_target)
            session_rows = rows.session_rows(authority, session_id)
            if any(row["status"] == "unknown" for row in session_rows):
                return []
            moved = []
            for row in session_rows:
                if row["status"] != "queued" or row["principal_id"] != principal_id:
                    continue
                payload = row["payload"]
                descriptor = payload.get("local_automation_v1") if isinstance(payload, dict) else None
                if not isinstance(descriptor, dict) or descriptor.get("target") not in targets:
                    continue
                descriptor["target"] = new_target
                row.update(payload=rows.encode(payload), digest=rows.digest(session_id, payload, row["intent"]))
                moved.append(row["admission_id"])
            if moved:
                rows.bump_revision(authority, session_id)
            return moved

    def recover(self, *, epoch: int) -> int:
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            sessions = staged.scope.get("sessions", {})
            for worker in authority["workers"].values():
                # A child worker of another epoch dies with its owner.
                if worker.get("kind") == "child" and worker["status"] != "terminal" and worker["owner_epoch"] != epoch:
                    worker["status"] = "terminal"
                    child = sessions.get(worker["session_id"])
                    if child is not None and child.get("ended_at") is None:
                        child.update(ended_at=rows.now(), end_reason="agent_close")
            for worker in authority["workers"].values():
                if worker["status"] in ("registered", "running") and worker["owner_epoch"] != epoch:
                    worker["status"] = "unknown"
            marked = 0
            for row in authority["admissions"].values():
                if row["status"] == "terminal" and row["outcome"] == STEERING and row["owner_epoch"] != epoch:
                    row["outcome"] = "interrupted"
                elif row["status"] == "started" and row["owner_epoch"] != epoch:
                    row["status"] = "unknown"
                    marked += 1
            return marked

    def resolve_unknown(self, *, epoch: int, admission_id: str, generation: int, notice: str | None = None, captured: CapturedResult | None = None) -> AdmissionRecord:
        state = self._state()
        with state.transaction() as staged:
            authority = staged.authority
            state.require_epoch(authority, epoch)
            row = rows.admission(authority, admission_id)
            if (type(generation) is int and row["generation"] == generation and row["status"] == "terminal"
                    and (row["outcome"] == "interrupted" or rows._result_key(admission_id) in authority["meta"])):
                return rows.record(row)
            if type(generation) is not int or row["status"] != "unknown" or row["generation"] != generation:
                raise rows.stale("admission", admission_id)
            outcome = "interrupted"
            committed = captured is not None and row["owner_epoch"] == epoch
            if committed:
                outcome = captured.outcome
                if outcome not in ("completed", "interrupted", "failed"):
                    raise AuthorityRefusal("invalid_params")
                self._store_result(authority, row, dict(captured.result))
            target = row["target_session_id"]
            close_discarded_turn(staged.scope, self._turn_target(authority, staged.scope, target), None if committed else notice)
            rows.retire_workers(authority, row, row["owner_epoch"])
            state.failpoints.hit(SETTLE_FAILPOINT)
            row.update(status="terminal", outcome=outcome)
            rows.bump_revision(authority, target)
            return rows.record(row)

    # Shared lookups

    def _worker_states(self, authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str) -> set[str]:
        """The live worker statuses on the session's runtime lineage."""
        lineage = rows.runtime_lineage(self._backend, authority, tables, session_id)
        return {w["status"] for w in authority["workers"].values() if w["session_id"] in lineage and w["status"] in ("registered", "running", "unknown")}

    def _turn_target(self, authority: Mapping[str, Any], tables: Mapping[str, Any], session_id: str) -> str:
        """The closer's transcript: the compression tip of the local physical target."""
        return rows.chain_tip(self._backend, tables, rows.physical_target(self._backend, authority, tables, session_id))
