from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_cli import control_db as cp
from hermes_cli.control_contracts import validate_statute_dispatch_v1


CONTROL_RESULT_STATUSES = {"completed", "completed_with_warnings", "failed", "action_required"}
CONTROL_RESULT_SUCCESS_STATUSES = {"completed", "completed_with_warnings"}


@dataclass
class DispatchWorkItem:
    dispatch_id: str
    sender_profile: str
    receiver_profile: str
    lease_epoch: int
    payload: dict[str, Any]


Runner = Callable[..., dict[str, Any]]


def _row_to_item(row, lease_epoch: int) -> DispatchWorkItem:
    return DispatchWorkItem(
        dispatch_id=row["dispatch_id"],
        sender_profile=row["sender_profile"],
        receiver_profile=row["receiver_profile"],
        lease_epoch=lease_epoch,
        payload=json.loads(row["payload_json"]),
    )


class ControlDispatchWorker:
    def __init__(self, profile_id: str, instance_id: str, root: Path | None = None, *, lease_ms: int = 300_000):
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.root = root
        self.lease_ms = lease_ms

    def _connect(self):
        return cp.connect(root=self.root)

    def heartbeat_once(self) -> None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT role FROM cp_profiles WHERE profile_id=?", (self.profile_id,)).fetchone()
            if row and row["role"] != "worker":
                cp.heartbeat_instance(conn, self.instance_id, lease_ms=120_000)
            else:
                cp.register_instance(conn, self.profile_id, instance_id=self.instance_id, lease_ms=120_000)
        finally:
            conn.close()

    def claim_next(self) -> DispatchWorkItem | None:
        conn = self._connect()
        try:
            claimed = cp.claim_next_for_profile(conn, receiver_profile=self.profile_id, instance_id=self.instance_id, lease_ms=self.lease_ms)
            if not claimed:
                return None
            dispatch_id, epoch = claimed
            row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()
            cp.emit_status(conn, instance_id=self.instance_id, dispatch_id=dispatch_id, status="claimed", summary=f"claimed {dispatch_id}")
            return _row_to_item(row, epoch)
        finally:
            conn.close()

    def claim_dispatch(self, dispatch_id: str) -> DispatchWorkItem | None:
        conn = self._connect()
        try:
            ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=dispatch_id, instance_id=self.instance_id, lease_ms=self.lease_ms)
            if not ok or epoch is None:
                return None
            row = conn.execute("SELECT * FROM cp_dispatches WHERE dispatch_id=?", (dispatch_id,)).fetchone()
            cp.emit_status(conn, instance_id=self.instance_id, dispatch_id=dispatch_id, status="claimed", summary=f"claimed {dispatch_id}")
            return _row_to_item(row, epoch)
        finally:
            conn.close()

    def mark_running(self, item: DispatchWorkItem) -> bool:
        conn = self._connect()
        try:
            return cp.advance_dispatch(conn, item.dispatch_id, instance_id=self.instance_id, lease_epoch=item.lease_epoch, status="running") and bool(cp.emit_status(conn, instance_id=self.instance_id, dispatch_id=item.dispatch_id, status="running", summary=f"running {item.dispatch_id}"))
        finally:
            conn.close()

    def extend_lease(self, item: DispatchWorkItem, *, lease_ms: int | None = None) -> bool:
        conn = self._connect()
        try:
            cp.heartbeat_instance(conn, self.instance_id, lease_ms=lease_ms or self.lease_ms)
            return cp.extend_dispatch_lease(conn, item.dispatch_id, instance_id=self.instance_id, lease_epoch=item.lease_epoch, lease_ms=lease_ms or self.lease_ms)
        finally:
            conn.close()

    def record_artifacts(self, item: DispatchWorkItem, artifacts: list[dict[str, Any]]) -> None:
        conn = self._connect()
        try:
            for artifact in artifacts:
                cp.record_artifact(
                    conn,
                    dispatch_id=item.dispatch_id,
                    instance_id=self.instance_id,
                    lease_epoch=item.lease_epoch,
                    path=str(artifact.get("path") or ""),
                    summary=artifact.get("summary"),
                    metadata=artifact.get("metadata") or {},
                )
        finally:
            conn.close()

    def complete(self, item: DispatchWorkItem, result: dict[str, Any]) -> bool:
        conn = self._connect()
        try:
            cp.record_result(conn, dispatch_id=item.dispatch_id, instance_id=self.instance_id, lease_epoch=item.lease_epoch, result=result)
            ok = cp.advance_dispatch(conn, item.dispatch_id, instance_id=self.instance_id, lease_epoch=item.lease_epoch, status="completed")
            if ok:
                cp.emit_status(conn, instance_id=self.instance_id, dispatch_id=item.dispatch_id, status="completed", summary=result.get("summary") or f"completed {item.dispatch_id}")
            return ok
        finally:
            conn.close()

    def fail(self, item: DispatchWorkItem, error: str, result: dict[str, Any] | None = None) -> bool:
        failure_result = result or {
            "schema": "control_result_v1",
            "status": "failed",
            "summary": error,
            "artifacts": [],
            "tests": [],
            "blockers": [{"kind": "runtime_error", "message": error}],
        }
        conn = self._connect()
        try:
            cp.record_result(conn, dispatch_id=item.dispatch_id, instance_id=self.instance_id, lease_epoch=item.lease_epoch, result=failure_result)
            ok = cp.advance_dispatch(conn, item.dispatch_id, instance_id=self.instance_id, lease_epoch=item.lease_epoch, status="failed", last_error=error)
            if ok:
                cp.emit_status(conn, instance_id=self.instance_id, dispatch_id=item.dispatch_id, status="failed", summary=error)
            return ok
        finally:
            conn.close()


def deterministic_result(item: DispatchWorkItem) -> dict[str, Any]:
    payload = validate_statute_dispatch_v1(item.payload)
    artifact_path = Path(payload["repo_root"]) / ".hermes-control" / f"{item.dispatch_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "schema": "control_deterministic_artifact_v1",
        "dispatch_id": item.dispatch_id,
        "sender_profile": item.sender_profile,
        "receiver_profile": item.receiver_profile,
        "lease_epoch": item.lease_epoch,
        "status": "completed",
        "summary": f"deterministic worker completed {item.dispatch_id}",
    }
    artifact_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "control_result_v1",
        "status": "completed",
        "summary": f"deterministic worker completed {item.dispatch_id}",
        "artifacts": [{"path": str(artifact_path), "summary": "deterministic artifact"}],
        "tests": [{"command": "deterministic", "exit_code": 0, "summary": "no-op deterministic handler"}],
        "blockers": [],
    }


def run_deterministic_dispatch(*, root: Path | None, profile_id: str, instance_id: str, dispatch_id: str) -> dict[str, Any]:
    worker = ControlDispatchWorker(profile_id, instance_id, root)
    worker.heartbeat_once()
    try:
        item = worker.claim_dispatch(dispatch_id)
        if item is None:
            raise cp.ControlPlaneError(f"could not claim dispatch {dispatch_id}")
        validate_statute_dispatch_v1(item.payload, require_parent=bool(item.payload.get("parent_dispatch_id")))
        worker.mark_running(item)
        result = deterministic_result(item)
        worker.record_artifacts(item, result.get("artifacts", []))
        worker.complete(item, result)
        return {"dispatch_id": dispatch_id, "lease_epoch": item.lease_epoch, "result": result}
    finally:
        conn = worker._connect()
        try:
            cp.mark_instance_offline(conn, instance_id)
        finally:
            conn.close()


def _default_runner(cmd, *, env: dict[str, str], input_text: str, timeout_s: float, cwd: str | None) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd,
        timeout=timeout_s,
        check=False,
    )
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def build_agent_prompt(item: DispatchWorkItem) -> str:
    payload = validate_statute_dispatch_v1(item.payload, require_parent=bool(item.payload.get("parent_dispatch_id")))
    return (
        "You are a statute-worker Hermes agent running under the DB-centric control plane.\n"
        "Use only the dispatch payload below as operative scope. Do not use prior chat context.\n"
        "Do not push code, expose public network services, delete files, install packages, or perform other dangerous actions unless the Hermes approval system grants approval.\n"
        "When done, print exactly one line beginning with CONTROL_RESULT_JSON: followed by JSON with schema=control_result_v1, status, summary, artifacts, tests, blockers.\n\n"
        "Dispatch payload JSON:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


def _agent_command(profile_id: str) -> list[str]:
    hermes = shutil.which("hermes")
    base = [hermes] if hermes else [sys.executable, "-m", "hermes_cli.main"]
    return [*base, "-p", profile_id, "chat", "--quiet", "--source", "control-worker", "--query", "-"]


def _extract_control_result(stdout: str) -> dict[str, Any]:
    match = re.search(r"CONTROL_RESULT_JSON:\s*(\{.*\})", stdout, flags=re.DOTALL)
    if not match:
        raise ValueError("missing CONTROL_RESULT_JSON block")
    return json.loads(match.group(1))


def _parse_control_result(stdout: str) -> dict[str, Any]:
    result = _extract_control_result(stdout)
    return validate_control_result(result)


def validate_control_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("control result must be object")
    if result.get("schema") != "control_result_v1":
        raise ValueError("control result schema must be control_result_v1")
    if result.get("status") not in CONTROL_RESULT_STATUSES:
        raise ValueError("control result status invalid")
    if not isinstance(result.get("summary"), str) or not result.get("summary"):
        raise ValueError("control result summary required")
    for key in ("artifacts", "tests", "blockers"):
        if key not in result or not isinstance(result[key], list):
            raise ValueError(f"control result {key} list required")
    if result.get("status") in CONTROL_RESULT_SUCCESS_STATUSES and result.get("blockers"):
        raise ValueError("successful control result cannot include blockers")
    return result


def _runtime_artifact_path(item: DispatchWorkItem, root: Path | None) -> Path:
    control_root = root or Path(os.environ.get("HERMES_CONTROL_ROOT") or ".").resolve()
    path = control_root / "control-plane" / "agent-runs" / f"{item.dispatch_id}-{item.lease_epoch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_agent_dispatch(
    *,
    root: Path | None,
    profile_id: str,
    instance_id: str,
    dispatch_id: str,
    runner: Runner | None = None,
    timeout_s: float = 3600,
) -> dict[str, Any]:
    worker = ControlDispatchWorker(profile_id, instance_id, root, lease_ms=max(300_000, int(timeout_s * 1000) + 30_000))
    worker.heartbeat_once()
    try:
        item = worker.claim_dispatch(dispatch_id)
        if item is None:
            raise cp.ControlPlaneError(f"could not claim dispatch {dispatch_id}")
        validate_statute_dispatch_v1(item.payload, require_parent=bool(item.payload.get("parent_dispatch_id")))
        worker.mark_running(item)
        prompt = build_agent_prompt(item)
        env = os.environ.copy()
        control_root = root or Path(os.environ.get("HERMES_CONTROL_ROOT") or ".").resolve()
        env.update(
            {
                "HERMES_PROFILE": profile_id,
                "HERMES_PROFILE_ID": profile_id,
                "HERMES_CONTROL_ROOT": str(control_root),
                "HERMES_CONTROL_INSTANCE_ID": instance_id,
                "HERMES_CONTROL_DISPATCH_ID": dispatch_id,
                "HERMES_CONTROL_LEASE_EPOCH": str(item.lease_epoch),
                "HERMES_APPROVER_PROFILE": "default",
                "HERMES_CONTROL_ALLOWED_PATHS": json.dumps(item.payload.get("allowed_paths") or []),
            }
        )
        worker.extend_lease(item)
        cmd = _agent_command(profile_id)
        run = runner or _default_runner
        try:
            proc = run(cmd, env=env, input_text=prompt, timeout_s=timeout_s, cwd=item.payload.get("repo_root"))
            stdout = str(proc.get("stdout") or "")
            stderr = str(proc.get("stderr") or "")
            returncode = int(proc.get("returncode") if proc.get("returncode") is not None else 1)
            worker.extend_lease(item)
            run_artifact = {
                "schema": "control_agent_run_v1",
                "dispatch_id": dispatch_id,
                "lease_epoch": item.lease_epoch,
                "command": [shlex.quote(str(c)) for c in cmd],
                "returncode": returncode,
                "stdout_tail": cp.redact_text(stdout[-4000:]),
                "stderr_tail": cp.redact_text(stderr[-4000:]),
            }
            runtime_path = _runtime_artifact_path(item, root)
            runtime_path.write_text(json.dumps(run_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            runtime_path.chmod(0o600)
            if returncode != 0:
                raise RuntimeError(f"agent subprocess exited {returncode}: {stderr[-500:] or stdout[-500:]}")
            try:
                result = _parse_control_result(stdout)
            except Exception as exc:
                try:
                    run_artifact["invalid_control_result"] = cp.redact_jsonable(_extract_control_result(stdout))
                    run_artifact["stdout_tail"] = "[redacted: invalid CONTROL_RESULT_JSON captured separately]"
                    runtime_path.write_text(json.dumps(run_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                except Exception:
                    pass
                raise exc
            result.setdefault("artifacts", [])
            result["artifacts"] = [*result["artifacts"], {"path": str(runtime_path), "summary": "agent subprocess run log"}]
        except Exception as exc:
            result = {
                "schema": "control_result_v1",
                "status": "failed",
                "summary": str(exc),
                "artifacts": [],
                "tests": [],
                "blockers": [{"kind": "runtime_error", "message": str(exc)}],
            }
        worker.record_artifacts(item, result.get("artifacts", []))
        if result.get("status") in CONTROL_RESULT_SUCCESS_STATUSES:
            worker.complete(item, result)
        else:
            worker.fail(item, result.get("summary") or "agent worker failed", result=result)
        return {"dispatch_id": dispatch_id, "lease_epoch": item.lease_epoch, "result": result}
    finally:
        conn = worker._connect()
        try:
            cp.mark_instance_offline(conn, instance_id)
        finally:
            conn.close()


def agent_not_implemented_result() -> dict[str, Any]:
    return {"status": "not_implemented", "handler": "agent", "agent_worker_ready": False}
