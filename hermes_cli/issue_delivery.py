"""Execution and artifact evidence for Gateway-admitted GitHub Kanban claims."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def is_github_task(task) -> bool:
    return bool(re.fullmatch(r"github:ISVGitHub/autonomous-entity#(?:22|27)", task.created_by or ""))


def _binding(conn, task_id):
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='github_task_binding'",
        (task_id,),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("GitHub task requires exactly one canonical binding")
    return json.loads(rows[0]["payload"])


def record_worker_delivery(conn, task_id: str, *, context: dict, session_id: str,
                           runtime_identity: dict) -> None:
    """Record the exact Gateway-admitted attempt before its first model turn.

    The managed worker has no CLI active-session lease of its own: the session and
    claim are owned by Gateway. Callers must pass the frozen kanban frame context.
    """
    from hermes_cli import kanban_db as kb
    task = kb.get_task(conn, task_id)
    if not task or not is_github_task(task):
        return
    binding = _binding(conn, task_id)
    run_id = context.get("run_id")
    claim_lock = context.get("claim_lock")
    request_id = context.get("request_id")
    if (not isinstance(run_id, int) or not isinstance(claim_lock, str)
            or not isinstance(request_id, str) or task.session_id is not None
            or binding.get("task_sha256") != hashlib.sha256((task.body or "").encode()).hexdigest()
            or context.get("github_delivery") != {key: binding[key] for key in binding if key != "executor_profile"}):
        raise ValueError("GitHub task/session/body binding mismatch")
    admitted_rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='github_task_admitted'",
        (task_id, run_id),
    ).fetchall()
    if len(admitted_rows) != 1:
        raise ValueError("GitHub task requires one canonical Gateway admission")
    admitted = json.loads(admitted_rows[0]["payload"])
    owner_rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted'",
        (task_id, run_id),
    ).fetchall()
    owner = json.loads(owner_rows[0]["payload"]) if len(owner_rows) == 1 else {}
    required = {"task_id": task_id, "run_id": run_id, "claim_lock": claim_lock,
                "session_id": session_id, "request_id": request_id}
    if (len(owner_rows) != 1 or any(admitted.get(k) != v for k, v in required.items())
            or owner.get("session_id") != session_id or owner.get("request_id") != request_id
            or task.status != "running" or task.current_run_id != run_id or task.claim_lock != claim_lock):
        raise ValueError("GitHub task is not owned by the current Gateway claim")
    if any(admitted.get(k) != binding.get(k) for k in binding if k != "executor_profile"):
        raise ValueError("Gateway admission differs from authenticated Issue binding")
    if not isinstance(runtime_identity, dict):
        raise ValueError("Gateway runtime identity is required")
    runtime_root = runtime_identity.get("runtime_root")
    runtime_commit = runtime_identity.get("runtime_commit")
    if (not isinstance(runtime_root, str) or not Path(runtime_root).is_absolute()
            or Path(runtime_root).resolve() != Path(runtime_root)
            or not isinstance(runtime_commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", runtime_commit)):
        raise ValueError("Gateway runtime identity is invalid")
    with kb.write_txn(conn):
        existing = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='github_task_running'",
            (task_id, run_id),
        ).fetchall()
        payload = {**required, "task_sha256": binding["task_sha256"], **runtime_identity}
        if existing:
            if len(existing) != 1 or json.loads(existing[0]["payload"]) != payload:
                raise ValueError("GitHub running evidence is ambiguous or changed")
            return
        kb._append_event(conn, task_id, "github_task_running", payload, run_id=run_id)


def completion_evidence(conn, task_id, expected_run_id, metadata):
    """Validate the exact Gateway run and required files before task completion.

    The canonical complete_task hook must call this before its transaction commits.
    Non-GitHub tasks pass through unchanged.
    """
    from hermes_cli import kanban_db as kb
    task = kb.get_task(conn, task_id)
    if not task or not is_github_task(task):
        return metadata
    binding = _binding(conn, task_id)
    running_rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='github_task_running'",
        (task_id, expected_run_id),
    ).fetchall()
    admitted_rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='github_task_admitted'",
        (task_id, expected_run_id),
    ).fetchall()
    owner_rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted'",
        (task_id, expected_run_id),
    ).fetchall()
    if (not expected_run_id or task.current_run_id != expected_run_id
            or len(running_rows) != 1 or len(admitted_rows) != 1 or len(owner_rows) != 1):
        raise ValueError("GitHub completion requires a delivered, owned run")
    running = json.loads(running_rows[0]["payload"])
    admitted = json.loads(admitted_rows[0]["payload"])
    owner_admission = json.loads(owner_rows[0]["payload"])
    session_id = admitted.get("session_id")
    required = {"task_id": task_id, "run_id": expected_run_id,
                "claim_lock": task.claim_lock, "session_id": session_id}
    if (not isinstance(session_id, str) or not session_id
            or any(admitted.get(key) != value for key, value in required.items())
            or any(owner_admission.get(key) != admitted.get(key) for key in ("session_id", "request_id"))
            or any(running.get(key) != admitted.get(key) for key in ("task_id", "run_id", "claim_lock", "session_id", "request_id"))
            or any(admitted.get(key) != binding.get(key) for key in binding if key != "executor_profile")
            or running.get("task_sha256") != binding["task_sha256"]
            or hashlib.sha256((task.body or "").encode()).hexdigest() != binding["task_sha256"]):
        raise ValueError("GitHub completion binding changed")
    identity = {key: running.get(key) for key in ("runtime_root", "runtime_commit")}
    runtime_root = identity["runtime_root"]
    runtime_commit = identity["runtime_commit"]
    if (not isinstance(runtime_root, str) or not Path(runtime_root).is_absolute()
            or Path(runtime_root).resolve() != Path(runtime_root)
            or not isinstance(runtime_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", runtime_commit)):
        raise ValueError("completion runtime identity is missing")
    expected = binding.get("result_files")
    if not expected or not task.workspace_path:
        raise ValueError("GitHub completion requires an explicit artifact contract")
    workspace = Path(task.workspace_path).resolve(strict=True)
    supplied = metadata.get("artifacts", []) if isinstance(metadata, dict) else []
    if not isinstance(supplied, list) or any(not isinstance(value, str) for value in supplied):
        raise ValueError("GitHub completion artifact paths are invalid")
    candidates = [Path(value) for value in supplied]
    if (len(candidates) != len(expected)
            or any(candidate.is_symlink() for candidate in candidates)
            or len({candidate.name for candidate in candidates}) != len(candidates)):
        raise ValueError("GitHub completion artifact set is invalid")
    artifacts = {candidate.name: candidate.resolve(strict=True) for candidate in candidates}
    if set(artifacts) != set(expected):
        raise ValueError("GitHub completion artifact contract differs")
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
                    or value.get("session_id") != session_id
                    or value.get("source_comment_id") != binding["comment_id"]):
                raise ValueError("GitHub artifact provenance mismatch")
            if name == "automation_evidence.json":
                findings = value.get("findings")
                if (not isinstance(findings, list) or not findings
                        or any(not isinstance(item, dict) or not isinstance(item.get("source"), str) or not item["source"].strip()
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
        "session_id": session_id, "comment_id": binding["comment_id"],
        "task_sha256": binding["task_sha256"], "artifacts": proof, **identity,
        **({"canary_result": canary_result} if canary_result is not None else {}),
    }}


def evidence_summary(run_metadata, task_id, session_id, comment_id):
    proof = json.loads(run_metadata or "{}").get("github_delivery")
    if (not isinstance(proof, dict) or proof.get("task_id") != task_id
            or proof.get("session_id") != session_id or proof.get("comment_id") != comment_id
            or not proof.get("artifacts") or not proof.get("runtime_root")
            or not re.fullmatch(r"[0-9a-f]{40}", str(proof.get("runtime_commit", "")))):
        raise ValueError("GitHub RESULT has no verified artifact evidence")
    # Never publish arbitrary model prose, file content, or paths from the worker.
    return json.dumps(proof, sort_keys=True, ensure_ascii=True)


def verified_run_summary(conn, task_id, run, binding):
    """Check durable owner/runtime event and persisted output bytes before App RESULT."""
    admitted_rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='github_task_admitted'",
        (task_id, run['id']),
    ).fetchall()
    if len(admitted_rows) != 1:
        raise ValueError('completion has no unique canonical admission')
    admitted = json.loads(admitted_rows[0]['payload'])
    summary = evidence_summary(run['metadata'], task_id, admitted['session_id'], binding['comment_id'])
    proof = json.loads(summary)
    events = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='github_task_running'",
        (task_id, run['id']),
    ).fetchall()
    if len(events) != 1 or proof.get('run_id') != run['id']:
        raise ValueError('completion has no unique fenced running event')
    owner = json.loads(events[0]['payload'])
    task = conn.execute('SELECT body FROM tasks WHERE id=?', (task_id,)).fetchone()
    if (not task or hashlib.sha256((task['body'] or '').encode()).hexdigest() != binding['task_sha256']
            or proof.get('task_sha256') != binding['task_sha256']
            or owner.get('session_id') != admitted['session_id']
            or owner.get('task_sha256') != binding['task_sha256']
            or any(owner.get(key) != proof.get(key) for key in ('runtime_root', 'runtime_commit'))):
        raise ValueError('persisted delivery binding differs from owner/runtime evidence')
    _verify_persisted_artifacts(conn, task_id, binding, proof)
    return summary


def _verify_persisted_artifacts(conn, task_id, binding, proof):
    paths = {}
    for row in conn.execute("SELECT filename,stored_path FROM task_attachments WHERE task_id=? AND uploaded_by='kanban_complete'", (task_id,)):
        if row['filename'] in paths:
            raise ValueError('ambiguous persisted artifact')
        paths[row['filename']] = Path(row['stored_path'])
    if set(proof['artifacts']) != set(binding['result_files']):
        raise ValueError('persisted artifact contract differs')
    for name, expected in proof['artifacts'].items():
        path = paths.get(name)
        if path is None or path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1024 * 1024:
            raise ValueError('persisted artifact missing or unsafe')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('persisted artifact hash mismatch')
