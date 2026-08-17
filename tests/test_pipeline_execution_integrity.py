import argparse
import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import feature as feature_mod
from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli.feature_pipeline import (
    validate_decompose_artifact,
    validate_execute_artifact,
    validate_pr_qa_artifact,
)


GOOD_DECOMPOSE = """# Decomposition

## Child Tasks

### T1 — Implement core
- **Owner/profile:** octacon
- **Acceptance criteria:** works

### T6 — Quan QA
- **Owner/profile:** quan
- **Acceptance criteria:** tests

### T7 — Kensei Review
- **Owner/profile:** kensei-review
- **Acceptance criteria:** audit

## Acceptance Criteria
All tasks complete.

## Test Plan
Run tests.
"""


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_is_profile_spawnable", lambda _name: True)
    kb.init_db()
    return home


def _manifest(parent_id):
    return {
        "schema_version": 1,
        "parent_task_id": parent_id,
        "tasks": [
            {
                "key": "T1",
                "title": "Implement core",
                "owner": "octacon",
                "role": "implementation",
                "dependencies": [],
                "body": "## Problem\nImplement core.\n\n## Acceptance Criteria\nWorks.\n\n## Test Plan\npytest.",
                "workspace_kind": "worktree",
            },
            {
                "key": "T6",
                "title": "Quan QA",
                "owner": "quan",
                "role": "qa",
                "dependencies": ["T1"],
                "body": "## Problem\nIndependently test T1.\n\n## Acceptance Criteria\nGreen.\n\n## Test Plan\npytest.",
                "workspace_kind": "scratch",
            },
            {
                "key": "T7",
                "title": "Kensei Review",
                "owner": "kensei-review",
                "role": "audit",
                "dependencies": ["T6"],
                "body": "## Problem\nAudit evidence.\n\n## Acceptance Criteria\nHonest verdict.\n\n## Test Plan\nReview evidence.",
                "workspace_kind": "scratch",
            },
        ],
    }


def _seed_decompose(home):
    with kb.connect() as conn:
        parent_id = kb.create_task(
            conn,
            title="Feature parent",
            body="## Problem\nBuild.\n## Success Criteria\nWorks.",
            assignee="octacon",
            triage=True,
            tier="full",
            pipeline_mode="full",
        )
        conn.execute(
            "UPDATE tasks SET status='decompose', pipeline_stage='decompose' WHERE id=?",
            (parent_id,),
        )
        conn.commit()
    artifact_dir = home / "feature-artifacts" / parent_id
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "decompose-output.md").write_text(GOOD_DECOMPOSE)
    return parent_id, artifact_dir


def _write_manifest(artifact_dir, parent_id, manifest=None):
    payload = manifest or _manifest(parent_id)
    (artifact_dir / "decompose-tasks.json").write_text(json.dumps(payload))


def _mapping(conn, parent_id):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='decompose_children_created' ORDER BY id DESC LIMIT 1",
        (parent_id,),
    ).fetchone()
    assert row is not None
    return json.loads(row[0])["children"]


def _execution_evidence(parent_id, mapping):
    return {
        "schema_version": 1,
        "parent_task_id": parent_id,
        "children": [
            {
                "key": child["key"],
                "task_id": child["task_id"],
                "status": "done",
                "result_digest": hashlib.sha256(child["task_id"].encode()).hexdigest(),
            }
            for child in mapping
            if child["role"] == "implementation"
        ],
    }


def _pr_qa_evidence(parent_id, artifact_dir):
    hermaguard_digest = hashlib.sha256(
        (artifact_dir / "hermaguard-report.json").read_bytes()
    ).hexdigest()
    simplify_digest = hashlib.sha256(
        (artifact_dir / "simplify-swarm-report.json").read_bytes()
    ).hexdigest()
    return {
        "schema_version": 1,
        "parent_task_id": parent_id,
        "commit_sha": "a" * 40,
        "pull_request": {
            "url": "https://github.com/Sahil-SS9/KenseiAgent/pull/1",
            "state": "open",
        },
        "tests": [{"command": "pytest -q", "exit_code": 0, "summary": "green"}],
        "quality_gates": {
            "hermaguard": {
                "version": "2.1.0",
                "status": "pass",
                "report": "hermaguard-report.json",
                "report_sha256": hermaguard_digest,
            },
            "simplify_swarm": {
                "version": "2.0.0",
                "status": "pass",
                "report": "simplify-swarm-report.json",
                "report_sha256": simplify_digest,
            },
        },
    }


def _audit_report():
    return """## quan-fleet
code: PASS
arch: PASS
perf: PASS
security: PASS
ux: PASS

## kensei-review
Independent review found the implementation evidence complete and the tested change safe to present for sign-off.

## verdict
**Verdict: PASS**
"""


class TestDecomposeManifestGate:
    def test_markdown_without_manifest_fails(self, kanban_home):
        parent_id, artifact_dir = _seed_decompose(kanban_home)
        reason = validate_decompose_artifact(str(artifact_dir))
        assert reason == "Missing decompose-tasks.json artifact"

    def test_unknown_dependency_fails(self, kanban_home):
        parent_id, artifact_dir = _seed_decompose(kanban_home)
        manifest = _manifest(parent_id)
        manifest["tasks"][1]["dependencies"] = ["MISSING"]
        _write_manifest(artifact_dir, parent_id, manifest)
        assert "unknown dependency" in validate_decompose_artifact(str(artifact_dir)).lower()

    def test_cycle_fails(self, kanban_home):
        parent_id, artifact_dir = _seed_decompose(kanban_home)
        manifest = _manifest(parent_id)
        manifest["tasks"][0]["dependencies"] = ["T7"]
        _write_manifest(artifact_dir, parent_id, manifest)
        assert "cycle" in validate_decompose_artifact(str(artifact_dir)).lower()

    def test_valid_manifest_materialises_exact_dag_once(self, kanban_home):
        parent_id, artifact_dir = _seed_decompose(kanban_home)
        _write_manifest(artifact_dir, parent_id)
        with kb.connect() as conn:
            result = kb.dispatch_once(conn, dry_run=False)
            assert (parent_id, "decompose", "execute") in result.pipeline_advanced
            assert kb.get_task(conn, parent_id).pipeline_stage == "execute"
            mapping = _mapping(conn, parent_id)
            assert [(x["key"], x["role"]) for x in mapping] == [
                ("T1", "implementation"), ("T6", "qa"), ("T7", "audit")
            ]
            by_key = {x["key"]: kb.get_task(conn, x["task_id"]) for x in mapping}
            assert by_key["T1"].assignee == "octacon"
            assert by_key["T1"].status == "ready"
            assert by_key["T1"].workspace_kind == "worktree"
            assert "## Acceptance Criteria" in by_key["T1"].body
            assert by_key["T6"].status == "todo"
            assert by_key["T7"].status == "todo"
            assert kb.parent_ids(conn, by_key["T6"].id) == [by_key["T1"].id]
            assert kb.parent_ids(conn, by_key["T7"].id) == [by_key["T6"].id]
            count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            kb._create_decompose_child_tasks(conn, parent_id, str(artifact_dir))
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == count

    def test_materialisation_failure_does_not_advance(self, kanban_home, monkeypatch):
        parent_id, artifact_dir = _seed_decompose(kanban_home)
        _write_manifest(artifact_dir, parent_id)
        monkeypatch.setattr(kb, "_create_decompose_child_tasks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with kb.connect() as conn:
            result = kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "decompose"
            assert not any(x[0] == parent_id for x in result.pipeline_advanced)

    def test_nonspawnable_owner_does_not_advance(self, kanban_home, monkeypatch):
        parent_id, artifact_dir = _seed_decompose(kanban_home)
        _write_manifest(artifact_dir, parent_id)
        monkeypatch.setattr(
            kb,
            "_is_profile_spawnable",
            lambda name: name != "octacon",
        )
        with kb.connect() as conn:
            result = kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "decompose"
            assert not any(x[0] == parent_id for x in result.pipeline_advanced)
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


class TestExecutionAndQaGates:
    def _materialised(self, home):
        parent_id, artifact_dir = _seed_decompose(home)
        _write_manifest(artifact_dir, parent_id)
        with kb.connect() as conn:
            kb.dispatch_once(conn, dry_run=False)
            mapping = _mapping(conn, parent_id)
        return parent_id, artifact_dir, mapping

    def test_execute_requires_artifact_and_completed_implementation_children(self, kanban_home):
        parent_id, artifact_dir, mapping = self._materialised(kanban_home)
        assert "Missing execution-evidence.json" in validate_execute_artifact(str(artifact_dir))
        (artifact_dir / "execution-evidence.json").write_text(
            json.dumps(_execution_evidence(parent_id, mapping))
        )
        with kb.connect() as conn:
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "execute"
            impl = next(x for x in mapping if x["role"] == "implementation")
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (impl["task_id"],))
            conn.commit()
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "pr+qa"

    def test_pr_qa_requires_quan_done_tests_pr_and_current_quality_gates(self, kanban_home):
        parent_id, artifact_dir, mapping = self._materialised(kanban_home)
        impl = next(x for x in mapping if x["role"] == "implementation")
        qa = next(x for x in mapping if x["role"] == "qa")
        (artifact_dir / "execution-evidence.json").write_text(json.dumps(_execution_evidence(parent_id, mapping)))
        with kb.connect() as conn:
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (impl["task_id"],))
            conn.commit()
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "pr+qa"

        (artifact_dir / "hermaguard-report.json").write_text('{"status":"pass"}')
        (artifact_dir / "simplify-swarm-report.json").write_text('{"status":"pass"}')
        (artifact_dir / "pr-qa-evidence.json").write_text(json.dumps(_pr_qa_evidence(parent_id, artifact_dir)))
        assert validate_pr_qa_artifact(str(artifact_dir)) is None
        with kb.connect() as conn:
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "pr+qa"
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (qa["task_id"],))
            conn.commit()
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "audit"

    def test_pr_qa_rejects_tampered_quality_report(self, kanban_home):
        parent_id, artifact_dir, _mapping_rows = self._materialised(kanban_home)
        (artifact_dir / "hermaguard-report.json").write_text('{"status":"pass"}')
        (artifact_dir / "simplify-swarm-report.json").write_text('{"status":"pass"}')
        evidence = _pr_qa_evidence(parent_id, artifact_dir)
        (artifact_dir / "pr-qa-evidence.json").write_text(json.dumps(evidence))
        assert validate_pr_qa_artifact(str(artifact_dir)) is None

        (artifact_dir / "hermaguard-report.json").write_text('{"status":"tampered"}')
        assert "SHA-256 mismatch" in validate_pr_qa_artifact(str(artifact_dir))

    def test_pr_qa_evidence_is_bound_to_live_parent_task(self, kanban_home):
        parent_id, artifact_dir, mapping = self._materialised(kanban_home)
        impl = next(x for x in mapping if x["role"] == "implementation")
        qa = next(x for x in mapping if x["role"] == "qa")
        (artifact_dir / "execution-evidence.json").write_text(
            json.dumps(_execution_evidence(parent_id, mapping))
        )
        (artifact_dir / "hermaguard-report.json").write_text('{"status":"pass"}')
        (artifact_dir / "simplify-swarm-report.json").write_text('{"status":"pass"}')
        evidence = _pr_qa_evidence(parent_id, artifact_dir)
        evidence["parent_task_id"] = "t_wrong_parent"
        (artifact_dir / "pr-qa-evidence.json").write_text(json.dumps(evidence))
        with kb.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='done' WHERE id IN (?, ?)",
                (impl["task_id"], qa["task_id"]),
            )
            conn.commit()
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "pr+qa"

    def test_audit_requires_audit_child_done(self, kanban_home):
        parent_id, artifact_dir, mapping = self._materialised(kanban_home)
        impl = next(x for x in mapping if x["role"] == "implementation")
        qa = next(x for x in mapping if x["role"] == "qa")
        audit = next(x for x in mapping if x["role"] == "audit")
        (artifact_dir / "execution-evidence.json").write_text(json.dumps(_execution_evidence(parent_id, mapping)))
        (artifact_dir / "hermaguard-report.json").write_text('{"status":"pass"}')
        (artifact_dir / "simplify-swarm-report.json").write_text('{"status":"pass"}')
        (artifact_dir / "pr-qa-evidence.json").write_text(json.dumps(_pr_qa_evidence(parent_id, artifact_dir)))
        (artifact_dir / "audit-report.md").write_text(_audit_report())
        with kb.connect() as conn:
            conn.execute("UPDATE tasks SET status='done' WHERE id IN (?, ?)", (impl["task_id"], qa["task_id"]))
            conn.commit()
            kb.dispatch_once(conn, dry_run=False)
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "audit"
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (audit["task_id"],))
            conn.commit()
            kb.dispatch_once(conn, dry_run=False)
            assert kb.get_task(conn, parent_id).pipeline_stage == "final_sign_off"

    def test_manual_force_cannot_bypass_integrity_stages(self, kanban_home, capsys):
        parent_id, artifact_dir = _seed_decompose(kanban_home)
        rc = feature_mod.cmd_feature_advance(argparse.Namespace(task_id=parent_id, force=True))
        assert rc == 1
        assert "cannot be force-bypassed" in capsys.readouterr().err.lower()


class TestKanbanBlockCommentContract:
    def test_block_cli_persists_reason_comment_and_state(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="Block me", initial_status="running")
            conn.commit()
        args = argparse.Namespace(task_id=tid, ids=[], reason=["verified", "defect"], kind="capability")
        assert kanban_cli._cmd_block(args) == 0
        with kb.connect() as conn:
            assert kb.get_task(conn, tid).status == "blocked"
            comments = kb.list_comments(conn, tid)
            assert comments[-1].body == "BLOCKED: verified defect"

    def test_failed_block_does_not_leave_misleading_comment(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="Already done", initial_status="running")
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
            conn.commit()
        args = argparse.Namespace(task_id=tid, ids=[], reason=["not", "blockable"], kind="capability")
        assert kanban_cli._cmd_block(args) == 1
        with kb.connect() as conn:
            assert kb.list_comments(conn, tid) == []
