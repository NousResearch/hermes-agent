from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from hermes_cli import control_db as cp
from hermes_cli.control_contracts import ContractError, make_child_payload, validate_statute_dispatch_v1
from hermes_cli.control_worker import CONTROL_RESULT_SUCCESS_STATUSES, DispatchWorkItem, validate_control_result

SpawnChild = Callable[[str, dict[str, Any], Path | None, str], int | None]


def _dispatch_item(row, epoch: int) -> DispatchWorkItem:
    return DispatchWorkItem(
        dispatch_id=row["dispatch_id"],
        sender_profile=row["sender_profile"],
        receiver_profile=row["receiver_profile"],
        lease_epoch=epoch,
        payload=json.loads(row["payload_json"]),
    )


class StatutePMFlow:
    def __init__(
        self,
        *,
        root: Path | None,
        pm_instance_id: str,
        admin_profile: str = "default",
        pm_profile: str = "statutepm",
        worker_profile: str = "statute-worker",
        spawn_child: SpawnChild | None = None,
        poll_interval_s: float = 1.0,
        child_timeout_s: float = 60.0,
    ):
        self.root = root
        self.admin_profile = admin_profile
        self.pm_profile = pm_profile
        self.worker_profile = worker_profile
        self.pm_instance_id = pm_instance_id
        if spawn_child is None:
            from hermes_cli.control_spawn import spawn_statute_worker

            self.spawn_child = lambda child_id, payload, child_root, parent_id: spawn_statute_worker(
                child_id,
                payload,
                child_root,
                parent_id,
                live=child_root is None,
                timeout_s=self.child_timeout_s,
            )
        else:
            self.spawn_child = spawn_child
        self.poll_interval_s = poll_interval_s
        self.child_timeout_s = child_timeout_s

    def _connect(self):
        return cp.connect(root=self.root)

    def _parent_lease_ms(self) -> int:
        return max(300_000, int((self.child_timeout_s + max(self.poll_interval_s, 0) + 30) * 1000))

    def heartbeat(self, *, lease_ms: int = 120_000) -> None:
        conn = self._connect()
        try:
            if not cp.heartbeat_instance(conn, self.pm_instance_id, lease_ms=lease_ms):
                cp.register_instance(
                    conn,
                    self.pm_profile,
                    instance_id=self.pm_instance_id,
                    lease_ms=lease_ms,
                    actor_type="bootstrap",
                    metadata={"finite_pm_runtime": True, "seeded_by_bootstrap": False},
                )
        finally:
            conn.close()

    def offline(self) -> None:
        conn = self._connect()
        try:
            cp.mark_instance_offline(conn, self.pm_instance_id)
        finally:
            conn.close()

    def run_once(self) -> dict[str, Any] | None:
        """Consume the oldest pending PM dispatch.

        This remains the queue-consumer path for long-running PM loops.  Finite
        Galt-owned wave dispatches must use ``run_dispatch`` so they supervise
        the exact parent dispatch they created.
        """
        try:
            self.heartbeat()
            parent_lease_ms = self._parent_lease_ms()
            conn = self._connect()
            try:
                claimed = cp.claim_next_for_profile(conn, receiver_profile=self.pm_profile, instance_id=self.pm_instance_id, lease_ms=parent_lease_ms)
                if not claimed:
                    return None
                parent_id, parent_epoch = claimed
                parent_row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (parent_id,)).fetchone()
                parent = _dispatch_item(parent_row, parent_epoch)
            finally:
                conn.close()
            return self._run_claimed_parent(parent, parent_lease_ms=parent_lease_ms)
        finally:
            self.offline()

    def run_dispatch(self, parent_dispatch_id: str) -> dict[str, Any]:
        """Run exactly one parent dispatch; never fall back to queue claiming."""
        parent_lease_ms = self._parent_lease_ms()
        try:
            self.heartbeat(lease_ms=parent_lease_ms)
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (parent_dispatch_id,)).fetchone()
                if not row:
                    return {"parent_dispatch_id": parent_dispatch_id, "status": "missing"}
                if row["receiver_profile"] != self.pm_profile:
                    return {"parent_dispatch_id": parent_dispatch_id, "status": "wrong_receiver", "receiver_profile": row["receiver_profile"]}
                if row["status"] in {"completed", "failed", "dead_letter"}:
                    return {"parent_dispatch_id": parent_dispatch_id, "status": row["status"], "already_terminal": True}
                ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=parent_dispatch_id, instance_id=self.pm_instance_id, lease_ms=parent_lease_ms)
                if not ok or epoch is None:
                    current = conn.execute("SELECT status, lease_instance_id, lease_epoch FROM cp_dispatches WHERE dispatch_id=?", (parent_dispatch_id,)).fetchone()
                    return {
                        "parent_dispatch_id": parent_dispatch_id,
                        "status": "not_claimed",
                        "current_status": current["status"] if current else "missing",
                        "lease_instance_id": current["lease_instance_id"] if current else None,
                        "lease_epoch": current["lease_epoch"] if current else None,
                    }
                parent_row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (parent_dispatch_id,)).fetchone()
                parent = _dispatch_item(parent_row, int(epoch))
            finally:
                conn.close()
            return self._run_claimed_parent(parent, parent_lease_ms=parent_lease_ms)
        finally:
            self.offline()

    def _run_claimed_parent(self, parent: DispatchWorkItem, *, parent_lease_ms: int) -> dict[str, Any]:
        try:
            parent_payload = validate_statute_dispatch_v1(parent.payload)
        except ContractError as exc:
            conn = self._connect()
            try:
                result = _result("failed", f"invalid parent dispatch: {exc}", blockers=[{"kind": "missing_context", "message": str(exc)}])
                cp.record_result(conn, dispatch_id=parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, result=result)
                cp.create_message_from_instance(conn, sender_instance_id=self.pm_instance_id, receiver_profile=self.admin_profile, kind="action_required", body=json.dumps(result, sort_keys=True))
                cp.advance_dispatch(conn, parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, status="failed", last_error=str(exc))
            finally:
                conn.close()
            return {"parent_dispatch_id": parent.dispatch_id, "status": "failed", "error": str(exc)}

        conn = self._connect()
        try:
            cp.advance_dispatch(conn, parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, status="running")
            child_payload = make_child_payload(parent_payload, parent_dispatch_id=parent.dispatch_id)
            child_payload["pm_context"] = {
                "schema": "statutepm_context_v1",
                "parent_dispatch_id": parent.dispatch_id,
                "pm_instance_id": self.pm_instance_id,
                "fresh_context": True,
            }
            child_id = cp.create_dispatch_from_instance(
                conn,
                sender_instance_id=self.pm_instance_id,
                receiver_profile=self.worker_profile,
                payload=child_payload,
                parent_dispatch_id=parent.dispatch_id,
                idempotency_key=f"pm-child:{parent.dispatch_id}:worker:0:0",
            )
        finally:
            conn.close()

        pid = self.spawn_child(child_id, child_payload, self.root, parent.dispatch_id) if self.spawn_child else None
        deadline = time.monotonic() + self.child_timeout_s
        child_row = None
        latest = None
        while True:
            conn = self._connect()
            try:
                self.heartbeat(lease_ms=parent_lease_ms)
                cp.extend_dispatch_lease(conn, parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, lease_ms=parent_lease_ms)
                cp.reap_expired_dispatches(conn)
                child_row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (child_id,)).fetchone()
                latest = cp.get_latest_dispatch_result(conn, child_id)
                if child_row and child_row["status"] in {"completed", "failed", "dead_letter"}:
                    break
            finally:
                conn.close()
            if time.monotonic() >= deadline:
                break
            time.sleep(self.poll_interval_s)

        conn = self._connect()
        try:
            child_status = child_row["status"] if child_row else "missing"
            if child_status != "completed" or not latest:
                blocker = {"kind": "runtime_error", "message": f"child dispatch {child_id} status={child_status}"}
                result = _result("action_required", "child dispatch did not complete", blockers=[blocker])
                cp.record_result(conn, dispatch_id=parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, result=result)
                cp.create_message_from_instance(conn, sender_instance_id=self.pm_instance_id, receiver_profile=self.admin_profile, kind="action_required", body=json.dumps(result, sort_keys=True))
                cp.advance_dispatch(conn, parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, status="failed", last_error=blocker["message"])
                return {"parent_dispatch_id": parent.dispatch_id, "child_dispatch_id": child_id, "status": "failed", "pid": pid}

            child_result = latest["result"]
            try:
                validate_control_result(child_result)
            except Exception as exc:
                blocker = {"kind": "runtime_error", "message": f"invalid child result contract: {exc}"}
                result = _result("action_required", "child result contract invalid", blockers=[blocker])
                cp.record_result(conn, dispatch_id=parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, result=result)
                cp.create_message_from_instance(conn, sender_instance_id=self.pm_instance_id, receiver_profile=self.admin_profile, kind="action_required", body=json.dumps(result, sort_keys=True))
                cp.advance_dispatch(conn, parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, status="failed", last_error=blocker["message"])
                return {"parent_dispatch_id": parent.dispatch_id, "child_dispatch_id": child_id, "status": "failed", "pid": pid}

            if child_result.get("status") not in CONTROL_RESULT_SUCCESS_STATUSES:
                blocker = {"kind": "runtime_error", "message": f"child dispatch {child_id} result status={child_result.get('status')}"}
                result = _result("action_required", "child result did not complete", blockers=[*child_result.get("blockers", []), blocker])
                cp.record_result(conn, dispatch_id=parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, result=result)
                cp.create_message_from_instance(conn, sender_instance_id=self.pm_instance_id, receiver_profile=self.admin_profile, kind="action_required", body=json.dumps(result, sort_keys=True))
                cp.advance_dispatch(conn, parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, status="failed", last_error=blocker["message"])
                return {"parent_dispatch_id": parent.dispatch_id, "child_dispatch_id": child_id, "status": "failed", "pid": pid}

            result = _result(
                child_result.get("status", "completed"),
                child_result.get("summary", "statutepm completed child dispatch"),
                artifacts=child_result.get("artifacts", []),
                tests=child_result.get("tests", []),
                blockers=child_result.get("blockers", []),
            )
            cp.record_result(conn, dispatch_id=parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, result=result)
            cp.create_message_from_instance(conn, sender_instance_id=self.pm_instance_id, receiver_profile=self.admin_profile, kind="status", body=json.dumps(result, sort_keys=True))
            cp.advance_dispatch(conn, parent.dispatch_id, instance_id=self.pm_instance_id, lease_epoch=parent.lease_epoch, status="completed")
            return {"parent_dispatch_id": parent.dispatch_id, "child_dispatch_id": child_id, "status": "completed", "pid": pid}
        finally:
            conn.close()


def _result(status: str, summary: str, *, artifacts: list[dict[str, Any]] | None = None, tests: list[dict[str, Any]] | None = None, blockers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": "control_result_v1",
        "status": status,
        "summary": summary,
        "artifacts": artifacts or [],
        "tests": tests or [],
        "blockers": blockers or [],
    }
