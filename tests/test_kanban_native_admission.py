import json

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_native_admission as native
from hermes_cli.kanban import run_slash


def _req(**overrides):
    data = {
        "title": "Implement native ledger pilot",
        "tenant": "hermes",
        "repo_full_name": "NousResearch/hermes-agent",
        "profile": "yuuka",
        "executor": "hermes-direct",
        "closeout_policy": "pr_review_handoff_then_done_closeout",
        "base_branch": "main",
        "worktree_branch": "ch417-kanban-native-create",
        "skills": ("hermes-agent",),
    }
    data.update(overrides)
    return native.NativeAdmissionRequest(**data)


def test_native_create_run_slash_json_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()

    out = run_slash(
        "native-create 'CLI native work' "
        "--tenant hermes --repo NousResearch/hermes-agent --profile yuuka "
        "--executor hermes-direct --closeout-policy pr_review_handoff_then_done_closeout "
        "--base-branch main --worktree-branch ch417-kanban-native-create --json"
    )
    data = json.loads(out)

    assert data["status"] == "created"
    assert data["public_id"] == "HL-001"
    assert data["task"]["assignee"] is None
    assert data["side_effects"]["executor_spawned"] is False
    assert data["authority"]["routing_verdict"]["verdict"] == "Hermes direct"
    assert data["authority"]["admission_snapshot"]["executor_dispatch"] == "forbidden_during_admission"


def test_native_admission_dry_run_is_linear_free_and_no_write(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()

    with kb.connect() as conn:
        result = native.create_native_work(conn, _req(), dry_run=True)
        count = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]

    assert result["status"] == "would_create"
    assert result["public_id"] == "HL-001"
    assert result["task"]["idempotency_key"] == "kanban:HL-001"
    assert result["side_effects"] == {
        "kanban_task_written": False,
        "executor_spawned": False,
        "linear_required": False,
        "linear_mutated": False,
    }
    assert count == 0


def test_native_admission_creates_triage_task_with_separate_public_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()

    with kb.connect() as conn:
        result = native.create_native_work(conn, _req(public_id="HL-009"))
        task = kb.get_task(conn, result["task_id"])

    assert result["status"] == "created"
    assert result["created"] is True
    assert task is not None
    assert task.id != "HL-009"
    assert task.public_id == "HL-009"
    assert task.idempotency_key == "kanban:HL-009"
    assert task.status == "triage"
    assert task.assignee is None
    assert task.workspace_kind == "worktree"
    assert task.tenant == "hermes"
    assert task.skills == ["hermes-agent"]
    payload_text = task.body.split("```json source_payload\n", 1)[1].split("\n```", 1)[0]
    payload = json.loads(payload_text)
    assert payload["source"] == "kanban_native"
    assert payload["public_id"] == "HL-009"
    assert payload["admission"]["linear_required"] is False
    assert payload["admission"]["executor_dispatch"] == "forbidden_during_admission"
    assert payload["routing"]["verdict"] == "hermes-direct"
    assert payload["routing"]["approval_boundary"] == "human_approval_required"
    assert payload["closeout"]["worker_done_review_ready_closed_are_distinct"] is True
    assert task.routing_verdict["verdict"] == "Hermes direct"
    assert task.routing_verdict["reason"]
    assert task.admission_snapshot["linear_required"] is False
    assert task.closeout_evidence["evidence_status"] == "not_started"


def test_native_admission_reuses_existing_public_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()

    with kb.connect() as conn:
        first = native.create_native_work(conn, _req(public_id="HL-002"))
        second = native.create_native_work(conn, _req(public_id="HL-002", title="Retry title"))
        rows = conn.execute("SELECT id FROM tasks WHERE public_id = ?", ("HL-002",)).fetchall()

    assert first["created"] is True
    assert second["status"] == "reused"
    assert second["task_id"] == first["task_id"]
    assert len(rows) == 1


def test_native_admission_fails_closed_without_required_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()

    with kb.connect() as conn:
        result = native.create_native_work(conn, _req(repo_full_name="", profile="", closeout_policy=""))
        count = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]

    assert result["status"] == "blocked"
    assert result["reason"] == "native_admission_missing_required_fields"
    assert result["missing"] == ["repo_full_name", "profile", "closeout_policy"]
    assert count == 0


def test_native_admission_rejects_ch_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()

    with kb.connect() as conn:
        try:
            native.create_native_work(conn, _req(namespace="CH"), dry_run=True)
        except ValueError as exc:
            assert "CH is reserved" in str(exc)
        else:
            raise AssertionError("expected CH namespace rejection")


def test_native_admission_parent_validation_is_no_write(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()

    with kb.connect() as conn:
        try:
            native.create_native_work(conn, _req(parents=("missing-task",)))
        except ValueError as exc:
            assert "unknown parent task" in str(exc)
        else:
            raise AssertionError("expected missing parent rejection")
        count = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]

    assert count == 0
