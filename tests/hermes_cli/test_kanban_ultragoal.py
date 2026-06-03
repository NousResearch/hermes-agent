"""Durable Kanban-Ultragoal controller tests.

These tests encode RALPLAN v2: Kanban remains authority, a controller run
checkpoints/resumes across bounded ticks, and PR readiness is impossible before
verifier + reviewer + PR/CI evidence exists.
"""

from __future__ import annotations

import json

import pytest


def _authority(task_id="BO-203", *, snapshot_hash="sha256:a", done_hash="sha256:d", status="triage"):
    return {
        "authority": "kanban",
        "taskId": task_id,
        "publicId": task_id,
        "status": status,
        "routingVerdict": "direct-kanban",
        "executionApproved": True,
        "snapshotHash": snapshot_hash,
        "doneCriteriaHash": done_hash,
    }


def test_start_creates_canonical_run_root_with_subordinate_ultragoal_root(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    run = store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")

    root = tmp_path / ".hermes" / "goal-runs" / "BO-203"
    assert run.run_id == "BO-203"
    assert (root / "run.json").exists()
    assert (root / "authority.json").exists()
    assert (root / "ledger.jsonl").exists()
    assert (root / "ultragoal" / "goals.json").exists()
    assert json.loads((root / "authority.json").read_text())["snapshotHash"] == "sha256:a"


def test_run_id_rejects_path_traversal(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        store.start("../../escape", authority=_authority("../../escape"), root_objective="bad")


def test_mutating_tick_requires_fresh_kanban_authority_match(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")

    with pytest.raises(ValueError, match="snapshotHash"):
        store.tick("BO-203", authority=_authority(snapshot_hash="sha256:changed"))

    with pytest.raises(ValueError, match="taskId"):
        store.tick("BO-203", authority=_authority("BO-999"))


def test_budget_exhaustion_writes_resumable_pending_action(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")
    run = store.tick("BO-203", authority=_authority(), budget_remaining=0)

    assert run.state == "running"
    assert run.resumable is True
    assert run.pending_action is not None
    assert run.pending_action["phase"] == "prepared"
    assert run.pending_action["stepId"].startswith("BO-203:tick-")
    events = [json.loads(line)["event"] for line in store.ledger_path("BO-203").read_text().splitlines()]
    assert events[-1] == "checkpoint_budget_near_limit"


def test_controller_transitions_block_pr_until_verifier_and_reviewer_pass(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")

    run = store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": []})
    assert run.state == "worker_done"

    with pytest.raises(ValueError, match="reviewed PR gate"):
        store.record_pr_created("BO-203", authority=_authority(), pr={"url": "https://example.invalid/pr/1"})

    run = store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": False, "missing": [{"criterionId": "DC-1", "reason": "no test"}]},
    )
    assert run.state == "verification_failed"
    assert run.current_goal_id is not None

    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    run = store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]},
    )
    assert run.state == "verification_passed"

    run = store.record_reviewer_result(
        "BO-203",
        authority=_authority(),
        result={"recommendation": "REQUEST_CHANGES", "securityConcerns": [], "logicErrors": ["missing resume test"]},
    )
    assert run.state == "review_failed"
    assert run.current_goal_id is not None

    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]},
    )
    run = store.record_reviewer_result(
        "BO-203",
        authority=_authority(),
        result={"recommendation": "APPROVE", "securityConcerns": [], "logicErrors": []},
    )
    assert run.state == "review_passed"

    run = store.record_pr_created(
        "BO-203",
        authority=_authority(),
        pr={"url": "https://github.com/chriskim12/hermes-agent/pull/1", "number": 1, "headSha": "abc"},
    )
    assert run.state == "pr_created"


def test_pr_ready_requires_complete_artifact_package(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")
    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]},
    )
    store.record_reviewer_result(
        "BO-203",
        authority=_authority(),
        result={"recommendation": "APPROVE", "securityConcerns": [], "logicErrors": []},
    )
    store.record_pr_created(
        "BO-203",
        authority=_authority(),
        pr={"url": "https://github.com/chriskim12/hermes-agent/pull/1", "number": 1, "headSha": "abc"},
    )

    with pytest.raises(ValueError, match="CI"):
        store.mark_review_ready("BO-203", authority=_authority())

    run = store.record_ci_result("BO-203", authority=_authority(), ci={"state": "success", "headSha": "abc"})
    assert run.state == "ci_passed"
    store.record_cleanup_proof("BO-203", authority=_authority(), proof={"status": "passed", "retained": []})
    run = store.mark_review_ready("BO-203", authority=_authority())
    assert run.state == "review_ready"
    package = json.loads((tmp_path / ".hermes" / "goal-runs" / "BO-203" / "pr.json").read_text())
    assert package["url"].endswith("/1")


def test_force_start_clears_stale_ledger_and_evidence(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="first")
    stale = tmp_path / ".hermes" / "goal-runs" / "BO-203" / "evidence" / "old.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": []})

    store.start("BO-203", authority=_authority(), root_objective="second", force=True)

    assert not stale.exists()
    events = [json.loads(line)["event"] for line in store.ledger_path("BO-203").read_text().splitlines()]
    assert events == ["run_started"]


def test_state_regressions_are_rejected_after_later_phases(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")
    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]},
    )

    with pytest.raises(ValueError, match="worker_done transition"):
        store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": []})


def test_reviewer_approval_requires_explicit_empty_blocker_lists(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")
    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]},
    )

    with pytest.raises(ValueError, match="securityConcerns"):
        store.record_reviewer_result("BO-203", authority=_authority(), result={"recommendation": "APPROVE"})


def test_start_with_stale_root_without_run_json_is_fail_closed_and_force_cleans(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    root = tmp_path / ".hermes" / "goal-runs" / "BO-203"
    stale = root / "evidence" / "old.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    with pytest.raises(FileExistsError):
        store.start("BO-203", authority=_authority(), root_objective="first")

    store.start("BO-203", authority=_authority(), root_objective="clean", force=True)
    assert not stale.exists()
    assert (root / "run.json").exists()


def test_rejected_reviewer_payload_is_not_persisted(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")
    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]},
    )

    with pytest.raises(ValueError, match="securityConcerns"):
        store.record_reviewer_result("BO-203", authority=_authority(), result={"recommendation": "APPROVE"})
    assert not (tmp_path / ".hermes" / "goal-runs" / "BO-203" / "reviews" / "final.json").exists()


def test_authority_requires_exact_direct_kanban_routing(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    bad = _authority()
    bad["routingVerdict"] = "kanban-ultragoal"
    with pytest.raises(ValueError, match="routing"):
        store.start("BO-203", authority=bad, root_objective="Bring one reviewed PR")


def test_successful_resume_clears_stale_pending_action(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")
    run = store.tick("BO-203", authority=_authority(), budget_remaining=0)
    assert run.pending_action is not None

    run = store.tick("BO-203", authority=_authority(), budget_remaining=20)
    assert run.pending_action is not None
    assert run.pending_action["executor"] == "hermes-direct-goal-loop"
    assert run.pending_action["dispatcherUsed"] is False
    assert run.resumable is False


def test_ci_success_must_match_pr_head_sha_and_success_clears_repair_goal(tmp_path):
    from hermes_cli.kanban_ultragoal import KanbanUltragoalStore

    store = KanbanUltragoalStore(tmp_path)
    store.start("BO-203", authority=_authority(), root_objective="Bring one reviewed PR")
    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": False, "missing": [{"criterionId": "DC-1", "reason": "first fail"}]},
    )
    assert store.load_run("BO-203").current_goal_id is not None
    store.record_worker_done("BO-203", authority=_authority(), evidence={"commandsRun": ["pytest"]})
    run = store.record_verifier_result(
        "BO-203",
        authority=_authority(),
        result={"passed": True, "doneCriteriaEvidence": [{"criterionId": "DC-1", "evidence": "pytest"}]},
    )
    assert run.current_goal_id is None
    store.record_reviewer_result(
        "BO-203",
        authority=_authority(),
        result={"recommendation": "APPROVE", "securityConcerns": [], "logicErrors": []},
    )
    store.record_pr_created(
        "BO-203",
        authority=_authority(),
        pr={"url": "https://github.com/chriskim12/hermes-agent/pull/1", "number": 1, "headSha": "abc"},
    )

    with pytest.raises(ValueError, match="headSha"):
        store.record_ci_result("BO-203", authority=_authority(), ci={"state": "success"})
    with pytest.raises(ValueError, match="headSha"):
        store.record_ci_result("BO-203", authority=_authority(), ci={"state": "success", "headSha": "other"})

    run = store.record_ci_result("BO-203", authority=_authority(), ci={"state": "success", "headSha": "abc"})
    assert run.state == "ci_passed"
    assert run.current_goal_id is None


def test_cli_start_status_and_tick_json(tmp_path, capsys):
    from hermes_cli.kanban_ultragoal import build_parser, kanban_ultragoal_command
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_parser(sub)
    authority = json.dumps(_authority())

    args = parser.parse_args([
        "kanban-ultragoal",
        "--workdir",
        str(tmp_path),
        "--json",
        "start",
        "BO-203",
        "--authority-json",
        authority,
        "--root-objective",
        "Bring one reviewed PR",
    ])
    assert kanban_ultragoal_command(args) == 0
    start_out = json.loads(capsys.readouterr().out)
    assert start_out["state"] == "admitted"

    args = parser.parse_args([
        "kanban-ultragoal",
        "--workdir",
        str(tmp_path),
        "--json",
        "tick",
        "BO-203",
        "--authority-json",
        authority,
        "--budget-remaining",
        "0",
    ])
    assert kanban_ultragoal_command(args) == 0
    tick_out = json.loads(capsys.readouterr().out)
    assert tick_out["resumable"] is True


def _seed_live_task(db_path, *, task_id="t_parent", public_id="BO-217", status="triage", execution_approved=True):
    from hermes_cli import kanban_db as kb

    kb.init_db(db_path)
    conn = kb.connect(db_path)
    for ddl in (
        "ALTER TABLE tasks ADD COLUMN public_id TEXT",
        "ALTER TABLE tasks ADD COLUMN routing_verdict TEXT",
        "ALTER TABLE tasks ADD COLUMN admission_snapshot TEXT",
        "ALTER TABLE tasks ADD COLUMN goal_mode INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass
    now = 1780366203
    body = """
Goal
Run a live Kanban card like Ultragoal until a reviewed PR or terminal blocker.

Done criteria:
- Authority snapshot command reads canonical Kanban state.
- Lifecycle transition commands expose existing controller transitions.
- pilot-check is strict read-only.
"""
    snapshot = {
        "execution_approved": execution_approved,
        "source": "test",
        "doneCriteria": [
            "Authority snapshot command reads canonical Kanban state.",
            "Lifecycle transition commands expose existing controller transitions.",
            "pilot-check is strict read-only.",
        ],
    }
    conn.execute(
        """
        insert into tasks(id,title,body,assignee,status,priority,tenant,workspace_kind,created_by,created_at,public_id,routing_verdict,admission_snapshot,goal_mode)
        values(?,?,?,?,?,?,?,?,?,?,?,?,?,0)
        """,
        (task_id, f"{public_id} — parent", body, None, status, 0, "BO", "scratch", "yuuka", now, public_id, "direct-kanban", json.dumps(snapshot)),
    )
    conn.commit()
    conn.close()


def test_authority_snapshot_command_reads_live_kanban_and_hashes_done_criteria(tmp_path, monkeypatch, capsys):
    from hermes_cli.kanban_ultragoal import build_parser, kanban_ultragoal_command
    import argparse

    db_path = tmp_path / "kanban.db"
    _seed_live_task(db_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_parser(sub)

    args = parser.parse_args(["kanban-ultragoal", "--json", "authority-snapshot", "BO-217"])
    assert kanban_ultragoal_command(args) == 0
    out = json.loads(capsys.readouterr().out)

    assert out["authority"] == "kanban"
    assert out["taskId"] == "t_parent"
    assert out["publicId"] == "BO-217"
    assert out["routingVerdict"] == "direct-kanban"
    assert out["executionApproved"] is True
    assert out["snapshotHash"].startswith("sha256:")
    assert out["doneCriteriaHash"].startswith("sha256:")
    assert out["doneCriteria"]


def test_pilot_check_is_strict_read_only_and_blocks_unapproved_authority(tmp_path, monkeypatch, capsys):
    from hermes_cli.kanban_ultragoal import build_parser, kanban_ultragoal_command
    import argparse
    import sqlite3

    db_path = tmp_path / "kanban.db"
    _seed_live_task(db_path, execution_approved=False)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    before = sqlite3.connect(db_path).execute("select count(*) from task_events").fetchone()[0]
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_parser(sub)

    args = parser.parse_args(["kanban-ultragoal", "--json", "pilot-check", "BO-217"])
    rc = kanban_ultragoal_command(args)
    after = sqlite3.connect(db_path).execute("select count(*) from task_events").fetchone()[0]
    out = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert before == after
    assert out["eligible"] is False
    assert "executionApproved" in out["blockers"]


def test_main_cli_propagates_kanban_ultragoal_nonzero_without_global_returncode_side_effects(tmp_path):
    import os
    import sqlite3
    import subprocess
    import sys

    db_path = tmp_path / "kanban.db"
    _seed_live_task(db_path, execution_approved=False)
    before = sqlite3.connect(db_path).execute("select count(*) from task_events").fetchone()[0]
    env = os.environ.copy()
    env["HERMES_KANBAN_DB"] = str(db_path)
    env["HERMES_HOME"] = str(tmp_path / ".hermes")
    env["PYTHONPATH"] = os.getcwd()

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban-ultragoal", "--json", "pilot-check", "BO-217"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    after = sqlite3.connect(db_path).execute("select count(*) from task_events").fetchone()[0]
    out = json.loads(result.stdout)

    assert result.returncode == 2, result.stderr + result.stdout
    assert before == after
    assert out["eligible"] is False
    assert "executionApproved" in out["blockers"]


def test_authority_snapshot_is_read_only_and_fails_closed_when_db_missing(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import build_authority_snapshot

    missing = tmp_path / "missing.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(missing))

    with pytest.raises(ValueError, match="does not exist"):
        build_authority_snapshot("BO-217")

    assert not missing.exists()
    assert not (tmp_path / "missing.db.init.lock").exists()


def test_authority_snapshot_fails_closed_on_ambiguous_task_reference(tmp_path, monkeypatch):
    from hermes_cli.kanban_ultragoal import build_authority_snapshot
    import sqlite3

    db_path = tmp_path / "kanban.db"
    _seed_live_task(db_path, task_id="same", public_id="BO-217")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        insert into tasks(id,title,body,assignee,status,priority,tenant,workspace_kind,created_by,created_at,public_id,routing_verdict,admission_snapshot,goal_mode)
        values(?,?,?,?,?,?,?,?,?,?,?,?,?,0)
        """,
        ("other", "ambiguous", "Done criteria:\n- one", None, "triage", 0, "BO", "scratch", "yuuka", 1780366204, "same", "direct-kanban", json.dumps({"execution_approved": True, "doneCriteria": ["one"]})),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    with pytest.raises(ValueError, match="ambiguous"):
        build_authority_snapshot("same")


def test_cli_transition_subcommands_expose_existing_store_state_machine(tmp_path, monkeypatch, capsys):
    from hermes_cli.kanban_ultragoal import build_parser, kanban_ultragoal_command
    import argparse

    db_path = tmp_path / "kanban.db"
    _seed_live_task(db_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    build_parser(sub)

    def run(argv):
        args = parser.parse_args(["kanban-ultragoal", "--workdir", str(tmp_path), "--json", *argv])
        assert kanban_ultragoal_command(args) == 0
        return json.loads(capsys.readouterr().out)

    authority = run(["authority-snapshot", "BO-217"])
    authority_json = json.dumps(authority)
    run(["start", "t_parent", "--authority-json", authority_json, "--root-objective", "one reviewed PR"])
    assert run(["record-worker-done", "t_parent", "--authority-json", authority_json, "--evidence-json", '{"commandsRun":["pytest"]}'])["state"] == "worker_done"
    assert run(["record-verifier-result", "t_parent", "--authority-json", authority_json, "--result-json", '{"passed":true,"doneCriteriaEvidence":[{"criterionId":"DC-1","evidence":"pytest"}]}'])["state"] == "verification_passed"
    assert run(["record-reviewer-result", "t_parent", "--authority-json", authority_json, "--result-json", '{"recommendation":"APPROVE","securityConcerns":[],"logicErrors":[]}'])["state"] == "review_passed"
    assert run(["record-pr-created", "t_parent", "--authority-json", authority_json, "--pr-json", '{"url":"https://github.com/chriskim12/hermes-agent/pull/92","number":92,"headSha":"abc"}'])["state"] == "pr_created"
    assert run(["record-ci-result", "t_parent", "--authority-json", authority_json, "--ci-json", '{"state":"success","headSha":"abc"}'])["state"] == "ci_passed"
    assert run(["record-cleanup-proof", "t_parent", "--authority-json", authority_json, "--proof-json", '{"status":"passed","retained":[]}'])["state"] == "ci_passed"
    assert run(["mark-review-ready", "t_parent", "--authority-json", authority_json])["state"] == "review_ready"
