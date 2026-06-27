from __future__ import annotations

import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_cli import control_db as cp
from hermes_cli.control_contracts import validate_statute_dispatch_v1


CONTROL_RESULT_STATUSES = {"completed", "completed_with_warnings", "failed", "action_required"}
CONTROL_RESULT_SUCCESS_STATUSES = {"completed", "completed_with_warnings"}
DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S = 600.0
DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S = 3000.0
DEFAULT_AGENT_WORKER_TIMEOUT_S = DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S
AGENT_WORKER_TERMINATE_GRACE_S = 10.0
AGENT_RUN_TAIL_CHARS = 4000
CONTROL_RESULT_STATUS_ALIASES = {
    "blocked": "action_required",
    "blocked_action_required": "action_required",
    "blocked_galt": "action_required",
    "gate_blocked": "action_required",
    "needs_action": "action_required",
    "needs_supervisor": "action_required",
    "requires_action": "action_required",
    "requires_human": "action_required",
    "requires_supervisor": "action_required",
}


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
                if not artifact.get("path"):
                    continue
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
            result_status = failure_result.get("status")
            blocker_kinds = {str(blocker.get("kind") or "") for blocker in failure_result.get("blockers", []) if isinstance(blocker, dict)}
            deliberate_action_required = result_status == "action_required" and "hard_timeout" not in blocker_kinds
            dispatch_status = "blocked" if deliberate_action_required else "failed"
            event_status = "blocked" if dispatch_status == "blocked" else "failed"
            last_error = failure_result.get("summary") if dispatch_status == "blocked" else error
            ok = cp.advance_dispatch(
                conn,
                item.dispatch_id,
                instance_id=self.instance_id,
                lease_epoch=item.lease_epoch,
                status=dispatch_status,
                last_error=str(last_error or error),
            )
            if ok:
                cp.emit_status(conn, instance_id=self.instance_id, dispatch_id=item.dispatch_id, status=event_status, summary=str(last_error or error))
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


def _read_temp_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _temp_stat(path: str) -> dict[str, Any]:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return {"bytes": 0, "mtime_ns": 0}
    return {"bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _terminate_process_group(proc: subprocess.Popen, *, grace_s: float = AGENT_WORKER_TERMINATE_GRACE_S) -> dict[str, Any]:
    terminated = False
    killed = False
    can_signal_group = os.name != "nt" and hasattr(os, "getpgid") and hasattr(os, "killpg")
    pgid = None
    if can_signal_group:
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None
    if proc.poll() is None:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)  # windows-footgun: ok
            else:
                proc.terminate()
            terminated = True
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.wait(timeout=max(grace_s, 0.0))
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            try:
                if pgid is not None:
                    os.killpg(pgid, getattr(signal, "SIGKILL", signal.SIGTERM))  # windows-footgun: ok
                else:
                    proc.kill()
                killed = True
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    return {"terminated": terminated, "killed": killed, "returncode": proc.poll()}


def _default_runner(
    cmd,
    *,
    env: dict[str, str],
    input_text: str,
    timeout_s: float,
    cwd: str | None,
    soft_timeout_s: float = DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S,
    lease_extender: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    stdout_tmp = tempfile.NamedTemporaryFile(prefix="hermes-agent-stdout-", delete=False)
    stderr_tmp = tempfile.NamedTemporaryFile(prefix="hermes-agent-stderr-", delete=False)
    stdout_path = stdout_tmp.name
    stderr_path = stderr_tmp.name
    stdout_tmp.close()
    stderr_tmp.close()
    soft_checks: list[dict[str, Any]] = []
    start_s = time.monotonic()
    hard_deadline_s = start_s + max(float(timeout_s), 0.0)
    next_soft_s = start_s + max(float(soft_timeout_s), 0.001)
    last_stdout = _temp_stat(stdout_path)
    last_stderr = _temp_stat(stderr_path)
    no_output_checks = 0
    proc: subprocess.Popen | None = None
    try:
        with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
            if proc.stdin is not None:
                proc.stdin.write(input_text)
                proc.stdin.close()
            while proc.poll() is None:
                now_s = time.monotonic()
                if now_s >= hard_deadline_s:
                    kill_info = _terminate_process_group(proc)
                    stdout = _read_temp_text(stdout_path)
                    stderr = _read_temp_text(stderr_path)
                    return {
                        "returncode": None,
                        "stdout": stdout,
                        "stderr": stderr,
                        "timed_out": True,
                        "hard_timed_out": True,
                        "timeout_s": timeout_s,
                        "hard_timeout_s": timeout_s,
                        "soft_timeout_s": soft_timeout_s,
                        "soft_timeout_checks": soft_checks,
                        "terminated": bool(kill_info["terminated"]),
                        "killed": bool(kill_info["killed"]),
                        "post_kill_returncode": kill_info["returncode"],
                    }
                if now_s >= next_soft_s:
                    if lease_extender is not None:
                        lease_extender()
                    stdout_stat = _temp_stat(stdout_path)
                    stderr_stat = _temp_stat(stderr_path)
                    output_changed = stdout_stat != last_stdout or stderr_stat != last_stderr
                    no_output_checks = 0 if output_changed else no_output_checks + 1
                    decision = "extend_progressing" if output_changed else "extend_alive_no_observed_output"
                    if no_output_checks >= 2:
                        decision = "soft_stall_suspected"
                    soft_checks.append(
                        {
                            "elapsed_s": round(now_s - start_s, 3),
                            "alive": True,
                            "stdout_bytes": stdout_stat["bytes"],
                            "stderr_bytes": stderr_stat["bytes"],
                            "stdout_mtime_ns": stdout_stat["mtime_ns"],
                            "stderr_mtime_ns": stderr_stat["mtime_ns"],
                            "output_changed": output_changed,
                            "decision": decision,
                        }
                    )
                    last_stdout = stdout_stat
                    last_stderr = stderr_stat
                    next_soft_s += max(float(soft_timeout_s), 0.001)
                time.sleep(min(0.1, max(0.001, hard_deadline_s - time.monotonic())))
            return {"returncode": proc.returncode, "stdout": _read_temp_text(stdout_path), "stderr": _read_temp_text(stderr_path), "soft_timeout_checks": soft_checks}
    finally:
        if proc is not None and proc.poll() is None:
            _terminate_process_group(proc)
        for path in (stdout_path, stderr_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _text_tail(value: Any, *, limit: int = AGENT_RUN_TAIL_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        text = bytes(value).decode("utf-8", errors="replace")
    else:
        text = str(value)
    return cp.redact_text(text[-limit:])


def _path_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _is_bounded_wave_payload(payload: dict[str, Any]) -> bool:
    constraints = payload.get("constraints") or {}
    if not isinstance(constraints, dict):
        return False
    if constraints.get("wave"):
        return True
    sprint_ids = constraints.get("sprint_ids")
    return isinstance(sprint_ids, list) and len(sprint_ids) > 1


def _post_wave_closeout_instructions(payload: dict[str, Any]) -> str:
    if not _is_bounded_wave_payload(payload):
        return ""
    return (
        "\nPost-wave closeout requirements for bounded NJ Statutes waves:\n"
        "- Execute the dispatched wave first. Do not start the next wave from inside this wave.\n"
        "- After the wave is otherwise complete, identify whether there were any control plane issues by inspecting parent/child dispatch rows, result rows, messages to default, open blockers, agent-run artifacts, handoff files, and relevant local logs for this wave.\n"
        "- If any control plane issue exists, handle it before final success using this exact loop: research -> diagnose -> plan -> write executable proposal -> looped oppositional review scoped only to the issue/proposal -> finalize proposal -> implement proposal -> review implementation -> fix any implementation errors found.\n"
        "- Identify the next wave from the executable contract ledger after closeout, starting from the current ready sprint; do not stop at a one-sprint prompt unless the next executable wave is genuinely one sprint. Run: /Users/johngalt/.hermes/hermes-agent/venv/bin/python /Users/johngalt/.hermes/profiles/nj-statutes-pm/scripts/autonomous_contract.py ready --db .contract-ledger/state.sqlite\n"
        "- Determine the next wave boundary from the executable contract and ledger state: include the ready sprint plus sequential dependent sprints that belong in the same bounded wave, respect stop conditions/gates/parallelSafety/scope, and hard-stop before the following wave. Do not infer the next wave from the just-completed sprint_ids list.\n"
        "- Write the full next-wave dispatch markdown file under docs/dispatches/ so Benjamin can provide it to Galt in a fresh Discord thread.\n"
        "- Include that markdown dispatch file in control_result_v1.artifacts as an existing .md path under docs/dispatches/.\n"
        "- If the next dispatch markdown cannot be produced because no next wave is ready, the repo/control plane is blocked, or permissions/scope prevent writing it, return status=action_required with blocker kind next_dispatch_prompt_missing.\n"
    )


def _bounded_wave_dispatch_artifact_problem(item: DispatchWorkItem, result: dict[str, Any]) -> str | None:
    if not _is_bounded_wave_payload(item.payload):
        return None
    if result.get("status") not in CONTROL_RESULT_SUCCESS_STATUSES:
        return None
    payload = validate_statute_dispatch_v1(item.payload, require_parent=bool(item.payload.get("parent_dispatch_id")))
    if "write" not in set(payload.get("task_permissions") or []):
        return "bounded wave requires write permission to produce docs/dispatches next-dispatch markdown"
    repo_root = Path(payload["repo_root"]).resolve(strict=False)
    dispatch_dir = (repo_root / "docs" / "dispatches").resolve(strict=False)
    allowed_paths = [Path(path).resolve(strict=False) for path in payload.get("allowed_paths") or []]
    if not any(_path_within(dispatch_dir, allowed) for allowed in allowed_paths):
        return "bounded wave requires docs/dispatches/ to be within dispatch allowed_paths"
    for artifact in result.get("artifacts") or []:
        if not isinstance(artifact, dict) or not artifact.get("path"):
            continue
        raw_artifact_path = Path(str(artifact["path"])).expanduser()
        artifact_path = (repo_root / raw_artifact_path if not raw_artifact_path.is_absolute() else raw_artifact_path).resolve(strict=False)
        if artifact_path.suffix != ".md":
            continue
        if not _path_within(artifact_path, dispatch_dir):
            continue
        if not any(_path_within(artifact_path, allowed) for allowed in allowed_paths):
            continue
        if artifact_path.is_file():
            return None
    return "bounded wave completed without required existing docs/dispatches/*.md next-dispatch artifact"


def _enforce_bounded_wave_postcondition(item: DispatchWorkItem, result: dict[str, Any]) -> dict[str, Any]:
    problem = _bounded_wave_dispatch_artifact_problem(item, result)
    if not problem:
        return result
    return {
        "schema": "control_result_v1",
        "status": "action_required",
        "summary": "bounded wave completed without required next-dispatch markdown artifact",
        "artifacts": list(result.get("artifacts") or []),
        "tests": list(result.get("tests") or []),
        "blockers": [
            *list(result.get("blockers") or []),
            {
                "kind": "next_dispatch_prompt_missing",
                "message": problem,
                "required_directory": "docs/dispatches/",
                "required_suffix": ".md",
            },
        ],
    }


def build_agent_prompt(item: DispatchWorkItem) -> str:
    payload = validate_statute_dispatch_v1(item.payload, require_parent=bool(item.payload.get("parent_dispatch_id")))
    return (
        "You are a statute-worker Hermes agent running under the DB-centric control plane.\n"
        "Use only the dispatch payload below as operative scope. Do not use prior chat context.\n"
        "Do not push code, expose public network services, delete files, install packages, or perform other dangerous actions unless the Hermes approval system grants approval.\n"
        "When done, print exactly one line beginning with CONTROL_RESULT_JSON: followed by JSON with schema=control_result_v1, status, summary, artifacts, tests, blockers.\n"
        "Allowed status values are exactly: completed, completed_with_warnings, failed, action_required.\n"
        "Use action_required for CodeRabbit, auth, push, CI, user-decision, or supervisor gates; do not invent blocked/custom statuses.\n"
        "artifacts must be an array of objects like {path, summary, metadata}; tests must be an array of objects like {command, exit_code, summary}; blockers must be an array of objects like {kind, message}.\n\n"
        "Dispatch payload JSON:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
        f"{_post_wave_closeout_instructions(payload)}"
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


def _coerce_artifact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value)
    return {"path": "", "summary": text, "metadata": {"raw_artifact": text, "coerced": True}}


def _coerce_test(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value)
    return {"command": text, "summary": text, "metadata": {"raw_test": text, "coerced": True}}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_blocker(value: Any, *, default_kind: str) -> dict[str, Any]:
    if isinstance(value, dict):
        blocker = dict(value)
        blocker.setdefault("kind", default_kind)
        if not blocker.get("message"):
            blocker["message"] = str(value)
        return blocker
    text = str(value)
    return {"kind": default_kind, "message": text, "metadata": {"raw_blocker": text, "coerced": True}}


def normalize_control_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("control result must be object")
    normalized = dict(result)
    raw_status = str(normalized.get("status") or "").strip()
    status = raw_status
    normalized_status = False
    raw_blockers = normalized.get("blockers")
    has_blockers = bool(_as_list(raw_blockers))
    if status not in CONTROL_RESULT_STATUSES:
        alias = CONTROL_RESULT_STATUS_ALIASES.get(status)
        if alias:
            status = alias
            normalized_status = True
        elif has_blockers:
            status = "action_required"
            normalized_status = True
        else:
            normalized["status"] = raw_status
            return normalized
    normalized["status"] = status
    normalized["artifacts"] = [_coerce_artifact(v) for v in _as_list(normalized.get("artifacts"))]
    normalized["tests"] = [_coerce_test(v) for v in _as_list(normalized.get("tests"))]
    default_blocker_kind = "runtime_error" if status == "failed" else "action_required"
    normalized["blockers"] = [_coerce_blocker(v, default_kind=default_blocker_kind) for v in _as_list(normalized.get("blockers"))]
    if normalized_status and raw_status != status:
        normalized["blockers"].append(
            {
                "kind": "control_result_status_normalized",
                "message": f"normalized non-contract control result status {raw_status!r} to {status!r}",
                "raw_status": raw_status,
                "normalized_status": status,
            }
        )
    return normalized


def validate_control_result(result: dict[str, Any]) -> dict[str, Any]:
    result = normalize_control_result(result)
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
    timeout_s: float | None = None,
    soft_timeout_s: float = DEFAULT_AGENT_WORKER_SOFT_TIMEOUT_S,
    hard_timeout_s: float | None = None,
) -> dict[str, Any]:
    if hard_timeout_s is None:
        hard_timeout_s = DEFAULT_AGENT_WORKER_HARD_TIMEOUT_S if timeout_s is None else timeout_s
    if timeout_s is None:
        timeout_s = hard_timeout_s
    worker = ControlDispatchWorker(profile_id, instance_id, root, lease_ms=max(300_000, int((hard_timeout_s + AGENT_WORKER_TERMINATE_GRACE_S) * 1000) + 30_000))
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
            if runner is None:
                proc = run(
                    cmd,
                    env=env,
                    input_text=prompt,
                    timeout_s=hard_timeout_s,
                    soft_timeout_s=soft_timeout_s,
                    cwd=item.payload.get("repo_root"),
                    lease_extender=lambda: worker.extend_lease(item),
                )
            else:
                proc = run(cmd, env=env, input_text=prompt, timeout_s=hard_timeout_s, cwd=item.payload.get("repo_root"))
            stdout = str(proc.get("stdout") or "")
            stderr = str(proc.get("stderr") or "")
            timed_out = bool(proc.get("hard_timed_out") or proc.get("timed_out"))
            returncode = int(proc.get("returncode") if proc.get("returncode") is not None else (1 if not timed_out else 0))
            worker.extend_lease(item)
            run_artifact = {
                "schema": "control_agent_run_v1",
                "dispatch_id": dispatch_id,
                "lease_epoch": item.lease_epoch,
                "command": [shlex.quote(str(c)) for c in cmd],
                "returncode": None if timed_out else returncode,
                "runner_kind": "default" if runner is None else "custom",
                "soft_timeout_s": soft_timeout_s,
                "hard_timeout_s": hard_timeout_s,
                "timeout_s": hard_timeout_s,
                "soft_timeout_checks": proc.get("soft_timeout_checks") or [],
                "stdout_tail": _text_tail(stdout),
                "stderr_tail": _text_tail(stderr),
            }
            if timed_out:
                run_artifact.update(
                    {
                        "timed_out": True,
                        "hard_timed_out": True,
                        "terminated": bool(proc.get("terminated")),
                        "killed": bool(proc.get("killed")),
                        "post_kill_returncode": proc.get("post_kill_returncode"),
                    }
                )
            runtime_path = _runtime_artifact_path(item, root)
            runtime_path.write_text(json.dumps(run_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            runtime_path.chmod(0o600)
            if timed_out:
                summary = f"agent subprocess hit hard timeout after {hard_timeout_s} seconds; partial work may exist and must be inspected before remediation/re-dispatch"
                result = {
                    "schema": "control_result_v1",
                    "status": "action_required",
                    "summary": summary,
                    "artifacts": [{"path": str(runtime_path), "summary": "agent subprocess hard-timeout run log"}],
                    "tests": [],
                    "blockers": [
                        {
                            "kind": "hard_timeout",
                            "message": summary,
                            "timeout_kind": "hard",
                            "timeout_s": hard_timeout_s,
                            "hard_timeout_s": hard_timeout_s,
                            "soft_timeout_s": soft_timeout_s,
                            "partial_work_may_exist": True,
                        }
                    ],
                }
                worker.record_artifacts(item, result.get("artifacts", []))
                worker.fail(item, result.get("summary") or "agent worker failed", result=result)
                return {"dispatch_id": dispatch_id, "lease_epoch": item.lease_epoch, "result": result}
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
            result = _enforce_bounded_wave_postcondition(item, result)
            result.setdefault("artifacts", [])
            result["artifacts"] = [*result["artifacts"], {"path": str(runtime_path), "summary": "agent subprocess run log"}]
        except subprocess.TimeoutExpired as exc:
            worker.extend_lease(item)
            runtime_path = _runtime_artifact_path(item, root)
            stdout_value = exc.stdout if exc.stdout is not None else getattr(exc, "output", None)
            run_artifact = {
                "schema": "control_agent_run_v1",
                "dispatch_id": dispatch_id,
                "lease_epoch": item.lease_epoch,
                "command": [shlex.quote(str(c)) for c in cmd],
                "returncode": None,
                "timed_out": True,
                "hard_timed_out": True,
                "runner_kind": "custom",
                "timeout_s": hard_timeout_s,
                "hard_timeout_s": hard_timeout_s,
                "soft_timeout_s": soft_timeout_s,
                "soft_timeout_checks": [],
                "stdout_tail": _text_tail(stdout_value),
                "stderr_tail": _text_tail(exc.stderr),
            }
            runtime_path.write_text(json.dumps(run_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            runtime_path.chmod(0o600)
            summary = f"agent subprocess hit hard timeout after {hard_timeout_s} seconds; partial work may exist and must be inspected before remediation/re-dispatch"
            result = {
                "schema": "control_result_v1",
                "status": "action_required",
                "summary": summary,
                "artifacts": [{"path": str(runtime_path), "summary": "agent subprocess timeout run log"}],
                "tests": [],
                "blockers": [{"kind": "hard_timeout", "message": summary, "timeout_kind": "hard", "timeout_s": hard_timeout_s, "hard_timeout_s": hard_timeout_s, "soft_timeout_s": soft_timeout_s, "partial_work_may_exist": True}],
            }
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
