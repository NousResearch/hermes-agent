"""Canonical Gateway admission and artifact proof for GitHub tasks."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import issue_delivery


def _proof_fixture(tmp_path, monkeypatch, *, include_running=True):
    task_id, run_id, session_id, comment_id = "task-1", 7, "gateway-session", 81
    body = "Prepare the requested index"
    workspace = tmp_path.resolve()
    artifact = workspace / "ARCH001_H1_REUSE_INDEX.json"
    artifact.write_text(json.dumps({
        "task_id": task_id, "session_id": session_id,
        "source_comment_id": comment_id, "entries": [],
    }))
    binding = {
        "repository": "ISVGitHub/autonomous-entity", "repository_id": 1,
        "issue_number": 22, "comment_id": comment_id, "app_id": 2,
        "installation_id": 3, "actor_id": 4, "nonce": "nonce_123456789012",
        "executor_profile": "grandloop", "recorded_at": 100,
        "task_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "result_files": [artifact.name],
    }
    identity = {"runtime_root": str(Path(__file__).resolve().parents[2]), "runtime_commit": "a" * 40}
    admitted = {
        "task_id": task_id, "run_id": run_id, "claim_lock": "claim-1",
        "session_id": session_id, "request_id": "kanban:[missions,task-1,7]",
        **{key: value for key, value in binding.items() if key != "executor_profile"},
    }
    running = {
        "task_id": task_id, "run_id": run_id, "claim_lock": "claim-1",
        "session_id": session_id, "request_id": admitted["request_id"],
        "task_sha256": binding["task_sha256"], **identity,
    }
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE task_events(task_id TEXT,run_id INTEGER,kind TEXT,payload TEXT)")
    for run, kind, payload in [
        (None, "github_task_binding", binding),
        (run_id, "owner_admitted", {"session_id": session_id, "request_id": admitted["request_id"]}),
        (run_id, "github_task_admitted", admitted),
    ]:
        conn.execute("INSERT INTO task_events VALUES (?,?,?,?)", (task_id, run, kind, json.dumps(payload)))
    if include_running:
        conn.execute("INSERT INTO task_events VALUES (?,?,?,?)", (task_id, run_id, "github_task_running", json.dumps(running)))
    task = SimpleNamespace(
        id=task_id, created_by="github:ISVGitHub/autonomous-entity#22", body=body,
        status="running", current_run_id=run_id, claim_lock="claim-1",
        session_id=None, workspace_path=str(workspace),
    )
    monkeypatch.setattr("hermes_cli.kanban_db.get_task", lambda _conn, _task_id: task)
    return conn, task, artifact, admitted, identity


def test_completion_evidence_uses_owner_derived_session_and_exact_artifact_contract(tmp_path, monkeypatch):
    conn, task, artifact, admitted, identity = _proof_fixture(tmp_path, monkeypatch)
    metadata = issue_delivery.completion_evidence(
        conn, task.id, task.current_run_id, {"artifacts": [str(artifact)]},
    )
    proof = metadata["github_delivery"]
    assert proof["session_id"] == admitted["session_id"]
    assert proof["run_id"] == task.current_run_id
    assert proof["runtime_commit"] == identity["runtime_commit"]
    assert proof["artifacts"] == {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}
    conn.close()


def test_completion_evidence_refuses_missing_gateway_running_marker(tmp_path, monkeypatch):
    conn, task, artifact, _admitted, _identity = _proof_fixture(tmp_path, monkeypatch, include_running=False)
    with pytest.raises(ValueError, match="delivered, owned run"):
        issue_delivery.completion_evidence(
            conn, task.id, task.current_run_id, {"artifacts": [str(artifact)]},
        )
    conn.close()


def test_completion_evidence_refuses_artifact_outside_contract(tmp_path, monkeypatch):
    conn, task, _artifact, _admitted, _identity = _proof_fixture(tmp_path, monkeypatch)
    extra = tmp_path / "other.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="artifact (set|contract)"):
        issue_delivery.completion_evidence(
            conn, task.id, task.current_run_id, {"artifacts": [str(extra)]},
        )
    conn.close()


def test_complete_task_cannot_mark_github_task_done_without_gateway_evidence(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env",
        lambda _name: str(tmp_path),
    )
    kb.create_board("missions")
    with connect(board="missions") as conn:
        task_id = kb.create_task(
            conn, title="Issue delivery", body="Prepare result",
            created_by="github:ISVGitHub/autonomous-entity#22", assignee="default",
            workspace_kind="scratch", project_id="", board="missions",
        )
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
        with pytest.raises(ValueError, match="canonical binding"):
            kb.complete_task(
                conn, task_id, summary="Done", expected_run_id=task.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_complete_task_persists_verified_gateway_artifact_and_run_metadata(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_workspace as kbw

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.create_board("missions")
    body = "Prepare the requested index"
    session_id, comment_id = "gateway-session", 81
    runtime = {"runtime_root": str(Path(__file__).resolve().parents[2]), "runtime_commit": "a" * 40}
    with kbc.connect(board="missions") as conn:
        task_id = kb.create_task(
            conn, title="Issue delivery", body=body,
            created_by="github:ISVGitHub/autonomous-entity#22", assignee="default",
            workspace_kind="scratch", project_id="", board="missions",
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        task = kb.claim_task(conn, task_id, claimer="gateway-claim")
        assert task is not None
        workspace = kb.workspaces_root(board="missions") / task_id
        workspace.mkdir(parents=True)
        artifact = workspace / "ARCH001_H1_REUSE_INDEX.json"
        artifact.write_text(json.dumps({
            "task_id": task_id, "session_id": session_id,
            "source_comment_id": comment_id, "entries": [],
        }))
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        kbw.set_workspace_path(conn, task_id, workspace)
        binding = {
            "repository": "ISVGitHub/autonomous-entity", "repository_id": 1,
            "issue_number": 22, "comment_id": comment_id, "app_id": 2,
            "installation_id": 3, "actor_id": 4, "nonce": "nonce_123456789012",
            "executor_profile": "grandloop", "recorded_at": 100,
            "task_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "result_files": [artifact.name],
        }
        kb._append_event(conn, task_id, "github_task_binding", binding)
        request_id = f"kanban:[missions,{task_id},{task.current_run_id}]"
        kb._append_event(conn, task_id, "owner_admitted", {"session_id": session_id, "request_id": request_id}, run_id=task.current_run_id)
        kb._append_event(conn, task_id, "github_task_admitted", {
            "task_id": task_id, "run_id": task.current_run_id, "claim_lock": task.claim_lock,
            "session_id": session_id, "request_id": request_id,
            **{key: value for key, value in binding.items() if key != "executor_profile"},
        }, run_id=task.current_run_id)
        issue_delivery.record_worker_delivery(
            conn, task_id,
            context={
                "run_id": task.current_run_id, "claim_lock": task.claim_lock,
                "request_id": request_id,
                "github_delivery": {key: value for key, value in binding.items() if key != "executor_profile"},
            },
            session_id=session_id, runtime_identity=runtime,
        )
        assert kb.complete_task(
            conn, task_id, summary="Index prepared", expected_run_id=task.current_run_id,
            metadata={"artifacts": [str(artifact)]},
        )
        completed = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        attachments = kb.list_attachments(conn, task_id)

    assert completed.status == "done"
    assert run.metadata["github_delivery"]["artifacts"] == {
        artifact.name: artifact_sha256,
    }
    assert run.metadata["github_delivery"]["session_id"] == session_id
    assert len(attachments) == 1
    assert attachments[0].filename == artifact.name
    assert hashlib.sha256(Path(attachments[0].stored_path).read_bytes()).hexdigest() == artifact_sha256
