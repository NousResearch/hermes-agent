"""Canonical owner and artifact evidence for GitHub sourced Kanban tasks."""
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
    try:
        binding = json.loads(rows[0]["payload"] or "{}")
    except (TypeError, ValueError) as exc:
        raise ValueError("GitHub task binding is corrupt") from exc
    return binding


def canonical_owner_admission(conn, task_id: str, run_id: int, binding: dict) -> dict:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted'",
        (task_id, run_id),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("GitHub run requires exactly one canonical owner admission")
    try:
        owner = json.loads(rows[0]["payload"] or "{}")
    except (TypeError, ValueError) as exc:
        raise ValueError("GitHub owner admission is corrupt") from exc
    request_id = "kanban:" + json.dumps(
        [binding.get("board"), task_id, run_id], separators=(",", ":"),
    )
    if (owner.get("request_id") != request_id or not isinstance(owner.get("session_id"), str)
            or not owner["session_id"]):
        raise ValueError("GitHub run is not bound to its canonical owner session")
    return owner


def append_worker_context(conn, task, context: str, *, board: str, session_id: str) -> str:
    """Add trusted artifact instructions after Gateway has derived the session id."""
    if not is_github_task(task):
        return context
    binding = _binding(conn, task.id)
    if (binding.get("board") != board or binding.get("executor_profile") != task.assignee
            or binding.get("task_sha256") != hashlib.sha256((task.body or "").encode()).hexdigest()):
        raise ValueError("GitHub task binding changed before owner admission")
    files = binding.get("result_files")
    if (not isinstance(files, list) or not files
            or any(not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."} for name in files)
            or len(set(files)) != len(files)):
        raise ValueError("GitHub task artifact contract is invalid")
    lines = [
        "## Required GitHub delivery evidence",
        "Write every required artifact inside this task workspace and pass its absolute path in metadata.artifacts to kanban_complete.",
        "Required files: " + json.dumps(files, ensure_ascii=True),
        "Each JSON artifact must contain task_id=" + task.id + ", session_id=" + session_id
        + ", source_comment_id=" + str(binding["comment_id"]) + ".",
        "automation_evidence.json must contain a nonempty findings list; each finding needs source and status (VERIFIED_RUNNING, IMPLEMENTED_NOT_RUNNING, HISTORICAL_ONLY, UNVERIFIED, CONFLICT).",
        "A diagnostic summary alone cannot complete this task. If useful work is blocked, use kanban_block with the exact blocker.",
    ]
    return context.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def completion_evidence(conn, task_id, expected_run_id, metadata):
    """Require canonical owner admission and contracted outputs before completion commits."""
    from hermes_cli import kanban_db as kb
    task = kb.get_task(conn, task_id)
    if not task or not is_github_task(task):
        return metadata
    if type(expected_run_id) is not int or task.current_run_id != expected_run_id:
        raise ValueError("GitHub completion requires the current admitted run")
    binding = _binding(conn, task_id)
    if (binding.get("executor_profile") != task.assignee
            or binding.get("task_sha256") != hashlib.sha256((task.body or "").encode()).hexdigest()):
        raise ValueError("GitHub completion binding changed")
    owner = canonical_owner_admission(conn, task_id, expected_run_id, binding)
    expected = binding.get("result_files")
    if not expected or not task.workspace_path:
        raise ValueError("GitHub completion requires an explicit artifact contract")
    workspace = Path(task.workspace_path).resolve(strict=True)
    raw_artifacts = metadata.get("artifacts", []) if isinstance(metadata, dict) else []
    if not isinstance(raw_artifacts, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in raw_artifacts):
        raise ValueError("GitHub completion artifact paths are invalid")
    artifacts = {}
    for value in raw_artifacts:
        path = Path(value).resolve(strict=True)
        if path.name in artifacts:
            raise ValueError("GitHub completion contains duplicate artifact names")
        artifacts[path.name] = path
    if set(artifacts) != set(expected):
        raise ValueError("GitHub completion artifact set does not match its contract")
    proof = {}
    canary_result = None
    for name in expected:
        path = artifacts[name]
        if not path.is_relative_to(workspace) or not path.is_file():
            raise ValueError(f"GitHub completion artifact is outside the task workspace: {name}")
        if not 0 < path.stat().st_size <= 1024 * 1024:
            raise ValueError("GitHub artifact size invalid")
        data = path.read_bytes()
        if name.endswith(".json"):
            value = json.loads(data)
            if (not isinstance(value, dict) or value.get("task_id") != task_id
                    or value.get("session_id") != owner["session_id"]
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
        "session_id": owner["session_id"], "comment_id": binding["comment_id"],
        "task_sha256": binding["task_sha256"], "artifacts": proof,
        **({"canary_result": canary_result} if canary_result is not None else {}),
    }}


def evidence_summary(run_metadata, task_id, session_id, comment_id):
    proof = json.loads(run_metadata or "{}").get("github_delivery")
    if (not isinstance(proof, dict) or proof.get("task_id") != task_id
            or proof.get("session_id") != session_id or proof.get("comment_id") != comment_id
            or not proof.get("artifacts")):
        raise ValueError("GitHub RESULT has no verified artifact evidence")
    return json.dumps(proof, sort_keys=True, ensure_ascii=True)
