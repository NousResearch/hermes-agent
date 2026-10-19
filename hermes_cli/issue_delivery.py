"""Execution evidence for the existing GitHub ingress and Kanban worker."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def is_github_task(task) -> bool:
    return bool(re.fullmatch(r"github:ISVGitHub/autonomous-entity#(?:22|27)", task.created_by or ""))


def current_github_worker() -> bool:
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not task_id:
        return False
    conn = kbc.connect(board=os.environ.get("HERMES_KANBAN_BOARD"))
    try:
        task = kb.get_task(conn, task_id)
        return bool(task and is_github_task(task))
    finally:
        conn.close()


def record_current_worker_delivery(session_id: str) -> None:
    if not current_github_worker():
        return
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect(board=os.environ.get("HERMES_KANBAN_BOARD"))
    try:
        record_worker_delivery(conn, os.environ["HERMES_KANBAN_TASK"], session_id)
    finally:
        conn.close()


def _binding(conn, task_id):
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='github_task_binding'",
        (task_id,),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("GitHub task requires exactly one canonical binding")
    return json.loads(rows[0]["payload"])


def record_worker_delivery(conn, task_id: str, session_id: str) -> None:
    """Called after the CLI acquired its normal owner lease, before the model turn."""
    from hermes_cli import active_sessions, kanban_db as kb
    task = kb.get_task(conn, task_id)
    if not task or not is_github_task(task):
        return
    binding = _binding(conn, task_id)
    if (binding["session_id"] != session_id or task.session_id != session_id
            or binding["task_sha256"] != hashlib.sha256((task.body or "").encode()).hexdigest()):
        raise ValueError("GitHub task/session/body binding mismatch")
    with active_sessions.session_owner_fence(session_id) as fence:
        entries = active_sessions._read_entries(active_sessions._state_path(), strict=True)
        if not any(row.get("session_id") == session_id and row.get("pid") == os.getpid() for row in entries):
            raise ValueError("GitHub session is owned by a different process")
        with kb.write_txn(conn):
            row = conn.execute("SELECT current_run_id,worker_pid,status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row["status"] != "running" or row["worker_pid"] != os.getpid():
                raise ValueError("GitHub task run is not owned by this worker")
            kb._append_event(conn, task_id, "github_task_running", {
                "session_id": session_id, "worker_pid": os.getpid(),
                "task_sha256": binding["task_sha256"],
            }, run_id=row["current_run_id"])
            fence.assert_current()


def completion_evidence(conn, task_id, expected_run_id, metadata):
    """Require actual outputs before the canonical completion transaction can commit."""
    from hermes_cli import kanban_db as kb
    task = kb.get_task(conn, task_id)
    if not task or not is_github_task(task):
        return metadata
    binding = _binding(conn, task_id)
    running = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='github_task_running' ORDER BY id DESC LIMIT 1",
        (task_id, expected_run_id),
    ).fetchone()
    if not expected_run_id or task.current_run_id != expected_run_id or not running:
        raise ValueError("GitHub completion requires a delivered, owned run")
    owner = json.loads(running["payload"])
    if (owner["session_id"] != binding["session_id"]
            or owner["task_sha256"] != binding["task_sha256"]
            or hashlib.sha256((task.body or "").encode()).hexdigest() != binding["task_sha256"]):
        raise ValueError("GitHub completion binding changed")
    expected = binding.get("result_files")
    if not expected or not task.workspace_path:
        raise ValueError("GitHub completion requires an explicit artifact contract")
    workspace = Path(task.workspace_path).resolve(strict=True)
    supplied = metadata.get("artifacts", []) if isinstance(metadata, dict) else []
    artifacts = {Path(value).name: Path(value).resolve(strict=True) for value in supplied}
    proof = {}
    canary_result = None
    for name in expected:
        path = artifacts.get(name)
        if path is None or not path.is_relative_to(workspace) or not path.is_file():
            raise ValueError(f"GitHub completion missing workspace artifact: {name}")
        if not 0 < path.stat().st_size <= 1024 * 1024:
            raise ValueError("GitHub artifact size invalid")
        data = path.read_bytes()
        if name.endswith(".json"):
            value = json.loads(data)
            if (not isinstance(value, dict) or value.get("task_id") != task_id
                    or value.get("session_id") != binding["session_id"]
                    or value.get("source_comment_id") != binding["comment_id"]):
                raise ValueError("GitHub artifact provenance mismatch")
            if name == "automation_evidence.json":
                findings = value.get("findings")
                if (not isinstance(findings, list) or not findings
                        or any(not isinstance(item, dict) or not item.get("source")
                               or item.get("status") not in {"VERIFIED_RUNNING", "IMPLEMENTED_NOT_RUNNING", "HISTORICAL_ONLY", "UNVERIFIED", "CONFLICT"}
                               for item in findings)):
                    raise ValueError("Automation evidence requires sourced, classified findings")
            if name == "CANARY_RESULT.json":
                canary_result = value.get("result")
                if type(canary_result) is not int or not -(2 ** 63) <= canary_result < 2 ** 63:
                    raise ValueError("Canary result must be a bounded integer")
        proof[name] = hashlib.sha256(data).hexdigest()
    if not any(name.endswith(".json") for name in expected):
        raise ValueError("GitHub completion requires a JSON provenance artifact")
    return {**(metadata or {}), "github_delivery": {
        "task_id": task_id, "run_id": expected_run_id,
        "session_id": binding["session_id"], "comment_id": binding["comment_id"],
        "task_sha256": binding["task_sha256"], "artifacts": proof,
        **({"canary_result": canary_result} if canary_result is not None else {}),
    }}


def evidence_summary(run_metadata, task_id, session_id, comment_id):
    proof = json.loads(run_metadata or "{}").get("github_delivery")
    if (not isinstance(proof, dict) or proof.get("task_id") != task_id
            or proof.get("session_id") != session_id or proof.get("comment_id") != comment_id
            or not proof.get("artifacts")):
        raise ValueError("GitHub RESULT has no verified artifact evidence")
    # Never publish arbitrary model prose, file content, or paths from the worker.
    return json.dumps(proof, sort_keys=True, ensure_ascii=True)
