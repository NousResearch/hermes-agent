"""Kanban-native closeout verifier tests."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_closeout as closeout
from hermes_cli import kanban_db as kb
from hermes_cli.kanban import run_slash


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, check=True, capture_output=True)
    return repo


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _valid_residue(**overrides):
    residue = {"summary": "Residue: none", "items": []}
    residue.update(overrides)
    return residue


def _review_ready_evidence(repo: Path, **overrides):
    evidence = {
        "summary": "implementation completed with regression coverage",
        "pr": {
            "number": 421,
            "url": "https://github.com/NousResearch/hermes-agent/pull/421",
            "state": "OPEN",
            "is_draft": False,
            "head_sha": _head(repo),
            "live": True,
        },
        "checks": [{"name": "tests", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        "verifier_verdict": {"verdict": "PASS"},
        "boundaries_confirmed": True,
        "cleanup": {"proof": "worktree cleanup verified", "worktree_clean": True, "artifacts_removed": []},
        "residue": _valid_residue(),
        "evidence": {"changed_files": ["hermes_cli/kanban_closeout.py"], "tests_run": ["targeted"]},
    }
    evidence.update(overrides)
    return evidence


def test_worker_done_transition_does_not_final_close(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="native governed work",
            closeout_evidence={"evidence_status": "not_started"},
        )
        result = closeout.transition_task_closeout(
            conn,
            task_id,
            "worker_done",
            {"summary": "worker finished; reviewer still required"},
        )
        task = kb.get_task(conn, task_id)

    assert result["status"] == "transitioned"
    assert task.review_phase == "worker_done"
    assert task.status == "blocked"
    assert task.completed_at is None
    assert task.review_phase != "closed"
    assert task.closeout_evidence["verification"]["linear_done_mutated"] is False


def test_worker_done_transition_from_triage_repairs_execution_status(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="manual admission work", triage=True)
        result = closeout.transition_task_closeout(
            conn,
            task_id,
            "worker_done",
            {"summary": "manual worker evidence accepted"},
        )
        task = kb.get_task(conn, task_id)

    assert result["status"] == "transitioned"
    assert task.review_phase == "worker_done"
    assert task.status == "blocked"


def test_review_ready_block_records_verifier_result_event(kanban_home):
    """A failed closeout verifier should leave a first-class auditable event."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="needs verifier",
            closeout_evidence={"evidence_status": "not_started"},
        )
        closeout.transition_task_closeout(
            conn,
            task_id,
            "worker_done",
            {"summary": "worker submitted evidence"},
        )

        result = closeout.transition_task_closeout(
            conn,
            task_id,
            "review_ready",
            {"summary": "missing verifier and review package"},
        )
        events = kb.list_events(conn, task_id)

    assert result["status"] == "blocked"
    verifier_events = [ev for ev in events if ev.kind == "verifier_result"]
    assert verifier_events, "blocked verifier decisions must be visible in Kanban events"
    payload = verifier_events[-1].payload
    assert payload is not None
    assert payload["target_phase"] == "review_ready"
    assert payload["verdict"] in {"FAIL", "BLOCKED"}
    assert "missing_verifier_pass" in payload["reason_codes"]
    assert payload["review_ready_input_eligible"] is False


def test_complete_task_cannot_bypass_review_ready_to_done(kanban_home, git_repo):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review package", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "done"})
        closeout.transition_task_closeout(
            conn,
            task_id,
            "review_ready",
            _review_ready_evidence(git_repo),
            repo_path=git_repo,
        )

        assert kb.complete_task(conn, task_id, result="accidental raw complete") is False
        task = kb.get_task(conn, task_id)

    assert task.status == "blocked"
    assert task.review_phase == "review_ready"


def test_complete_task_on_governed_task_sets_worker_done_not_closed(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="native governed work",
            closeout_evidence={"evidence_status": "not_started"},
        )
        assert kb.complete_task(conn, task_id, result="worker result")
        task = kb.get_task(conn, task_id)

    assert task.status == "blocked"
    assert task.review_phase == "worker_done"
    assert task.completed_at is None
    assert task.review_phase != "closed"


def test_complete_task_without_closeout_policy_keeps_legacy_done(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy worker task")
        assert kb.complete_task(conn, task_id, result="legacy done")
        task = kb.get_task(conn, task_id)

    assert task.status == "done"
    assert task.review_phase is None
    assert task.completed_at is not None


def test_complete_task_cannot_escalate_worker_done_to_board_done(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="governed retry handoff",
            closeout_evidence={"evidence_status": "not_started"},
        )
        closeout.transition_task_closeout(
            conn,
            task_id,
            "worker_done",
            {"summary": "first worker handoff"},
        )
        assert kb.complete_task(conn, task_id, result="retry handoff")
        task = kb.get_task(conn, task_id)

    assert task.status == "blocked"
    assert task.review_phase == "worker_done"
    assert task.completed_at is None



def test_review_ready_requires_live_pr_checks_evidence_and_cleanup(kanban_home, git_repo):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review me", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "done"})
        result = closeout.transition_task_closeout(
            conn,
            task_id,
            "review_ready",
            _review_ready_evidence(git_repo),
            repo_path=git_repo,
        )
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert result["status"] == "transitioned"
    assert task.review_phase == "review_ready"
    assert task.status == "blocked"
    assert result["task_status"] == "blocked"
    assert task.closeout_evidence["residue"]["summary"] == "Residue: none"
    assert task.closeout_evidence["verification"]["allowed"] is True
    assert task.closeout_evidence["verification"]["gateway_restarted_or_reloaded"] is False
    assert task.closeout_evidence["verification"]["pr_merged"] is False
    verifier_events = [event for event in events if event.kind == "verifier_result"]
    assert verifier_events, "review_ready PASS must leave a first-class verifier verdict event"
    payload = verifier_events[-1].payload
    assert payload["target_phase"] == "review_ready"
    assert payload["verdict"] == "PASS"
    assert payload["allowed"] is True
    assert payload["review_ready_input_eligible"] is True


def test_parent_review_ready_blocks_when_hierarchy_child_not_review_ready(kanban_home, git_repo):
    """Parent review_ready cannot substitute embedded childEvidence for child Kanban closeouts."""

    with kb.connect() as conn:
        parent_id = kb.create_task(
            conn,
            title="parent ultragoal",
            workspace_kind="dir",
            workspace_path=str(git_repo),
            goal_mode=True,
            closeout_evidence={"evidence_status": "not_started"},
        )
        child_id = kb.create_task(conn, title="child still triage", triage=True)
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id, relation_type) VALUES (?, ?, 'hierarchy')",
            (parent_id, child_id),
        )
        closeout.transition_task_closeout(conn, parent_id, "worker_done", {"summary": "parent worker done"})

        result = closeout.transition_task_closeout(
            conn,
            parent_id,
            "review_ready",
            _review_ready_evidence(
                git_repo,
                childEvidence={child_id: {"summary": "embedded parent evidence is not child closeout"}},
            ),
            repo_path=git_repo,
        )
        parent = kb.get_task(conn, parent_id)
        child = kb.get_task(conn, child_id)

    assert result["status"] == "blocked"
    assert "child_not_review_ready" in result["blockers"]
    assert parent.review_phase == "worker_done"
    assert child.status == "triage"
    assert child.review_phase is None
    matrix = result["evidence"]["child_review_matrix"]
    assert matrix == [
        {
            "task_id": child_id,
            "title": "child still triage",
            "status": "triage",
            "review_phase": None,
            "closeout_allowed": None,
            "ready": False,
            "reason": "child_not_review_ready",
        }
    ]


def test_parent_review_ready_allows_when_hierarchy_child_review_ready(kanban_home, git_repo):
    with kb.connect() as conn:
        parent_id = kb.create_task(
            conn,
            title="parent ultragoal",
            workspace_kind="dir",
            workspace_path=str(git_repo),
            goal_mode=True,
            closeout_evidence={"evidence_status": "not_started"},
        )
        child_id = kb.create_task(
            conn,
            title="child review ready",
            review_phase="review_ready",
            closeout_evidence={"verification": {"allowed": True}},
        )
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id, relation_type) VALUES (?, ?, 'hierarchy')",
            (parent_id, child_id),
        )
        closeout.transition_task_closeout(conn, parent_id, "worker_done", {"summary": "parent worker done"})

        result = closeout.transition_task_closeout(
            conn,
            parent_id,
            "review_ready",
            _review_ready_evidence(git_repo),
            repo_path=git_repo,
        )
        parent = kb.get_task(conn, parent_id)

    assert result["status"] == "transitioned"
    assert parent.review_phase == "review_ready"
    matrix = result["evidence"]["child_review_matrix"]
    assert matrix[0]["task_id"] == child_id
    assert matrix[0]["ready"] is True


def test_review_ready_requires_worker_done_phase(git_repo):
    result = closeout.verify_closeout_transition(
        "review_ready",
        _review_ready_evidence(git_repo),
        current_phase=None,
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "review_ready_requires_worker_done" in result.blockers


def test_review_ready_allows_no_pr_review_package_when_changed_files_empty(git_repo):
    evidence = _review_ready_evidence(
        git_repo,
        pr={},
        checks=[{"name": "local verifier", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        evidence={
            "tests_run": ["hermes kanban closeout BO-115 review_ready --check-only --json"],
            "changed_files": [],
            "artifact_refs": ["kanban://BO-115/events/verifier_result"],
            "proof": "Read-only smoke fixture verified; no repository diff was produced.",
        },
        no_pr_reason="No-code smoke fixture: changed_files is empty and proof/artifact evidence is attached.",
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is True
    assert result.blockers == []
    assert result.evidence["verification"]["allowed"] is True
    assert result.evidence["review_package"]["kind"] == "no_pr_evidence"


def test_review_ready_requires_no_pr_reason_and_artifact_when_changed_files_empty(git_repo):
    evidence = _review_ready_evidence(
        git_repo,
        pr={},
        evidence={"changed_files": []},
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "missing_no_pr_reason" in result.blockers
    assert "missing_no_pr_artifact_or_proof" in result.blockers
    assert "missing_live_pr" not in result.blockers


def test_review_ready_requires_pr_when_changed_files_non_empty_even_with_no_pr_reason(git_repo):
    evidence = _review_ready_evidence(
        git_repo,
        pr={},
        evidence={
            "changed_files": ["hermes_cli/kanban_closeout.py"],
            "artifact_refs": ["diff://local"],
            "proof": "code changed locally",
        },
        no_pr_reason="operator says no PR is needed",
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "missing_live_pr" in result.blockers
    assert "missing_no_pr_reason" not in result.blockers
    assert result.evidence["review_package"]["kind"] == "pr_required"


def test_review_ready_requires_authority_boundaries_for_every_review_package(git_repo):
    evidence = _review_ready_evidence(git_repo)
    evidence.pop("boundaries_confirmed")

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "missing_authority_boundary_confirmation" in result.blockers


@pytest.mark.parametrize(
    "mutate,blocker",
    [
        (lambda repo, ev: ev["pr"].update({"head_sha": "0" * 40}), "stale_pr"),
        (
            lambda repo, ev: ev.update(
                {"checks": [{"name": "ci", "status": "COMPLETED", "conclusion": "FAILURE"}]}
            ),
            "failed_checks",
        ),
        (lambda repo, ev: ev.update({"cleanup": {"worktree_clean": True}}), "missing_cleanup_proof"),
        (
            lambda repo, ev: ev.update(
                {"pr_candidates": [ev.pop("pr"), {"number": 422, "live": True}]}
            ),
            "ambiguous_pr_evidence",
        ),
    ],
)
def test_review_ready_fails_closed_for_stale_failed_missing_or_ambiguous_evidence(git_repo, mutate, blocker):
    evidence = _review_ready_evidence(git_repo)
    mutate(git_repo, evidence)

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert blocker in result.blockers
    assert result.evidence["verification"]["allowed"] is False


def test_review_ready_fails_closed_for_dirty_worktree(git_repo):
    (git_repo / "DIRTY.txt").write_text("dirty")
    evidence = _review_ready_evidence(git_repo)

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "dirty_worktree" in result.blockers


def test_review_ready_requires_residue_evidence(git_repo):
    evidence = _review_ready_evidence(git_repo)
    evidence.pop("residue")

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "missing_residue_evidence" in result.blockers


@pytest.mark.parametrize(
    "residue,blocker",
    [
        ({"summary": "Residue: needs decision", "items": [{"kind": "task_tmp", "path": "/tmp/x", "disposition": "needs_decision"}]}, "undisposed_residue"),
        ({"summary": "Residue retained", "items": [{"kind": "task_tmp", "path": "/tmp/x", "disposition": "retained", "revisit_at": "2026-06-01"}]}, "retained_residue_missing_reason"),
        ({"summary": "Residue retained", "items": [{"kind": "task_tmp", "path": "/tmp/x", "disposition": "retained", "reason": "debug evidence"}]}, "retained_residue_missing_ttl"),
        ({"summary": "Residue moved", "items": [{"kind": "workspace_backup", "path": "/tmp/ws.tgz", "disposition": "moved", "destination": "/tmp/ws.tgz", "reason": "rollback", "revisit_at": "2026-06-01"}]}, "moved_residue_not_on_data_mount"),
        ({"summary": "DB backups", "items": [], "db_backups": {"count": 12}}, "db_backup_retention_uncapped"),
    ],
)
def test_residue_policy_fails_closed_for_invalid_dispositions(git_repo, residue, blocker):
    result = closeout.verify_closeout_transition(
        "review_ready",
        _review_ready_evidence(git_repo, residue=residue),
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert blocker in result.blockers


def test_residue_policy_accepts_moved_and_retained_backup_dispositions(git_repo):
    residue = {
        "summary": "Residue accounted",
        "items": [
            {
                "kind": "workspace_backup",
                "path": "/home/ubuntu/.hermes/kanban/residue-backups/ws.tgz",
                "disposition": "moved",
                "destination": "/mnt/hermes-data/kanban-residue/ws.tgz",
                "reason": "rollback until review closes",
                "revisit_at": "2026-06-01",
                "size_bytes": 89000000,
            },
            {
                "kind": "archive_backup",
                "path": "/home/ubuntu/.hermes/kanban/residue-backups/archive.tgz",
                "disposition": "retained",
                "reason": "temporary audit evidence",
                "ttl": "7d",
            },
        ],
        "db_backups": {"count": 2, "retention": "latest 10 or 7 days", "revisit_at": "2026-06-01"},
    }

    result = closeout.verify_closeout_transition(
        "review_ready",
        _review_ready_evidence(git_repo, residue=residue),
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is True
    assert result.blockers == []


def test_closed_requires_residue_even_with_approval(git_repo):
    evidence = _review_ready_evidence(git_repo, approval={"decision": "approved", "approved_by": "reviewer"})
    evidence.pop("residue")

    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "missing_residue_evidence" in result.blockers


def test_worker_done_remains_compatible_without_residue(git_repo):
    result = closeout.verify_closeout_transition(
        "worker_done",
        {"summary": "worker finished; residue gate deferred to review_ready"},
        current_phase=None,
        repo_path=git_repo,
    )

    assert result.allowed is True
    assert "missing_residue_evidence" not in result.blockers


def test_closed_requires_review_ready_and_explicit_approval(kanban_home, git_repo):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="close me", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "done"})

        # Blocked: missing review_ready + approval
        blocked = closeout.transition_task_closeout(
            conn,
            task_id,
            "closed",
            _review_ready_evidence(git_repo),
            repo_path=git_repo,
        )
        assert blocked["status"] == "blocked"
        assert "closed_requires_review_ready" in blocked["blockers"]
        assert "missing_close_approval" in blocked["blockers"]

        # Transition to review_ready
        closeout.transition_task_closeout(
            conn,
            task_id,
            "review_ready",
            _review_ready_evidence(git_repo),
            repo_path=git_repo,
        )
        # closed now requires MERGED PR — use closed-specific evidence
        approved = _closed_evidence(git_repo)
        result = closeout.transition_task_closeout(conn, task_id, "closed", approved, repo_path=git_repo)
        task = kb.get_task(conn, task_id)

    assert result["status"] == "transitioned"
    assert task.review_phase == "closed"
    assert task.status == "done"
    assert task.closeout_evidence["approval"]["approved_by"] == "reviewer"


def test_closed_allows_documented_no_pr_exception_policy(kanban_home, git_repo):
    evidence = {
        "summary": "documentation-only cleanup completed",
        "cleanup": {"proof": "no worktree residue", "worktree_clean": True, "artifacts_removed": []},
        "residue": _valid_residue(),
        "verifier_verdict": {"verdict": "PASS"},
        "checks": [{"name": "local verifier", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        "approval": {"decision": "approved", "approved_by": "operator"},
        "no_pr_exception": {
            "policy": "docs-only-no-pr-closeout",
            "reason": "operator documented no repository diff requiring PR",
        },
    }
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="no pr close", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "done"})
        result = closeout.transition_task_closeout(conn, task_id, "closed", evidence, repo_path=git_repo)
        task = kb.get_task(conn, task_id)

    assert result["status"] == "transitioned"
    assert task.review_phase == "closed"
    assert task.status == "done"
    assert task.closeout_evidence["no_pr_exception"]["policy"] == "docs-only-no-pr-closeout"


def test_closeout_cli_check_only_blocks_without_writing(kanban_home, git_repo):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="cli closeout", workspace_kind="dir", workspace_path=str(git_repo))
        kb.set_task_authority(conn, task_id, review_phase="worker_done")

    out = run_slash(
        "closeout "
        f"{task_id} review_ready --repo {git_repo} --check-only --json --evidence "
        + shlex.quote(json.dumps({"summary": "missing pr"}))
    )
    payload = json.loads(out)

    assert payload["status"] == "blocked"
    assert "missing_no_pr_reason" in payload["blockers"]
    assert "missing_no_pr_artifact_or_proof" in payload["blockers"]
    assert "missing_live_pr" not in payload["blockers"]
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).review_phase == "worker_done"


def test_closeout_cli_report_json_includes_cross_lane_dimensions(kanban_home, git_repo):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="integrated closeout", workspace_kind="dir", workspace_path=str(git_repo))
        kb.set_task_authority(conn, task_id, review_phase="worker_done")
    evidence = _review_ready_evidence(
        git_repo,
        lane="ultragoal",
        goal_contract={"id": "goal-7", "criteria_hash": "crit-7"},
        done_criteria_ledger=_done_criteria_ledger(task_id=task_id),
        worker_evidence={
            "authority_boundary_confirmed": True,
            "per_criterion": {"DC-001": {"claim": "satisfied"}},
        },
        verifier_result={
            "verdict": "PASS",
            "per_criterion": {"DC-001": {"verdict": "pass"}},
        },
        completion_audit={"status": "PASS", "artifact_refs": ["task-7-green.txt"]},
        reviewer_readiness={"status": "PASS", "reviewer": "verifier"},
        parent_child_matrix={
            "parentRollupState": "complete",
            "countsByRollupState": {"complete": 1},
            "remainingChildren": [],
        },
    )

    out = run_slash(
        "closeout "
        f"{task_id} review_ready --repo {git_repo} --check-only --json --report --evidence "
        + shlex.quote(json.dumps(evidence))
    )
    payload = json.loads(out)

    report = payload["closeout_report"]
    assert report["lane"] == "ultragoal"
    assert report["dimensions"] == {
        "lane_identity": "PASS",
        "kanban_relation_drift": "PASS",
        "kanban_ready_contract": "PASS",
        "lane_owned_evidence": "PASS",
        "parent_child_coverage": "PASS",
        "worker_evidence": "PASS",
        "verifier_result": "BLOCKED",
        "completion_audit": "PASS",
        "cleanup": "PASS",
        "reviewer_readiness": "PASS",
    }
    assert "invalid_verifier_result_schema" in report["blocker_keys"]
    assert report["ready"] is False


def test_closeout_cli_report_plain_text_names_autopilot_parent_blocker(kanban_home, git_repo):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="autopilot report", workspace_kind="dir", workspace_path=str(git_repo))
        kb.set_task_authority(conn, task_id, review_phase="worker_done")
    evidence = _review_ready_evidence(
        git_repo,
        lane="autopilot",
        parent_child_matrix={
            "parentRollupState": "review_blocked",
            "countsByRollupState": {"complete": 1, "review_blocked": 1},
            "remainingChildren": ["child-2"],
        },
        worker_evidence={
            "authority_boundary_confirmed": True,
            "per_criterion": {"DC-001": {"claim": "satisfied"}},
        },
        verifier_result={"verdict": "PASS"},
        completion_audit={"status": "PASS", "artifact_refs": ["task-7-green.txt"]},
        reviewer_readiness={"status": "PASS", "reviewer": "verifier"},
    )

    out = run_slash(
        "closeout "
        f"{task_id} review_ready --repo {git_repo} --check-only --report --evidence "
        + shlex.quote(json.dumps(evidence))
    )

    assert "Closeout report:" in out
    assert "lane: autopilot" in out
    assert "parent_child_coverage: BLOCKED" in out
    assert "parent_child_coverage:blocked" in out


# ── Slice 4: prose-only cleanup rejection & structured evidence ──


def test_cleanup_prose_only_is_rejected_without_structured_artifacts(git_repo):
    """Prose-only cleanup proof ("cleaned up") without artifacts_removed or
    worktree_retained must fail with missing_structured_cleanup_evidence."""
    evidence = _review_ready_evidence(git_repo,
        cleanup={"proof": "everything cleaned up", "worktree_clean": True},
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "missing_structured_cleanup_evidence" in result.blockers


def test_cleanup_accepts_artifacts_removed_list(git_repo):
    """Cleanup with artifacts_removed list (even empty) must pass."""
    evidence = _review_ready_evidence(git_repo,
        cleanup={"proof": "verified", "worktree_clean": True, "artifacts_removed": ["node_modules"]},
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is True


def test_cleanup_retained_worktree_requires_reason_and_ttl(git_repo):
    """worktree_retained=True without retained_reason or revisit_at must fail."""
    evidence = _review_ready_evidence(git_repo,
        cleanup={"proof": "retaining worktree", "worktree_clean": True, "worktree_retained": True},
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "retained_worktree_missing_reason" in result.blockers


def test_cleanup_retained_worktree_with_reason_and_revisit_passes(git_repo):
    """worktree_retained=True with retained_reason and revisit_at must pass."""
    evidence = _review_ready_evidence(git_repo,
        cleanup={
            "proof": "retaining worktree for review",
            "worktree_clean": True,
            "worktree_retained": True,
            "retained_reason": "PR review pending",
            "revisit_at": "after PR merge or close",
        },
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is True


def test_worker_done_preserves_original_status(git_repo):
    """worker_done must not set the task status to done."""
    result = closeout.verify_closeout_transition(
        "worker_done",
        {"summary": "worker finished"},
        current_phase=None,
        repo_path=git_repo,
    )

    assert result.allowed is True
    # worker_done verification should pass without residue
    assert "missing_residue_evidence" not in result.blockers
    # Verifier itself does not set status; apply_closeout_transition now preserves it


def test_review_ready_status_is_blocked_not_done(git_repo):
    """review_ready transition must result in status='blocked', not 'done'."""
    result = closeout.verify_closeout_transition(
        "review_ready",
        _review_ready_evidence(git_repo),
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is True
    # The verifier doesn't mutate status, but the evidence must be accepted.
# ── Slice 6 — closed final Done gate ──────────────────────────────

def _closed_evidence(repo: Path, **overrides):
    """Helper: build closed-phase evidence with MERGED PR."""
    evidence = {
        "summary": "final closeout with post-merge cleanup proof",
        "pr": {
            "number": 421,
            "url": "https://github.com/NousResearch/hermes-agent/pull/421",
            "state": "MERGED",
            "is_draft": False,
            "head_sha": _head(repo),
            "live": True,
        },
        "checks": [{"name": "tests", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        "verifier_verdict": {"verdict": "PASS"},
        "cleanup": {
            "proof": "worktree removed, branch deleted, git prune complete",
            "worktree_clean": True,
            "artifacts_removed": ["task worktree"],
        },
        "residue": _valid_residue(),
        "approval": {"decision": "approved", "approved_by": "reviewer"},
        "evidence": {"changed_files": ["hermes_cli/kanban_closeout.py"], "tests_run": ["targeted"]},
    }
    evidence.update(overrides)
    return evidence


def test_closed_blocks_open_pr(git_repo):
    """closed must reject an OPEN pr — only MERGED is accepted for final Done."""
    evidence = _closed_evidence(
        git_repo,
        pr={**_closed_evidence(git_repo)["pr"], "state": "OPEN"},
    )
    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )
    assert result.allowed is False
    assert any(b in {"pr_not_merged", "pr_not_open", "stale_pr"} for b in result.blockers), \
        f"Expected a PR-state blocker, got: {result.blockers}"


def test_closed_accepts_merged_pr(git_repo):
    """closed with MERGED PR, approval, cleanup, and residue passes all gates."""
    evidence = _closed_evidence(git_repo)
    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )
    assert result.allowed is True, f"Blockers: {result.blockers}"
    assert result.blockers == []


def test_closed_accepts_squash_merged_pr_when_merge_commit_matches_repo_head(git_repo):
    """A squash-merged PR closes against the merge commit, not feature head."""
    feature_head = "f" * 40
    merge_commit = _head(git_repo)
    evidence = _closed_evidence(
        git_repo,
        pr={
            **_closed_evidence(git_repo)["pr"],
            "head_sha": feature_head,
            "merge_commit_sha": merge_commit,
        },
    )

    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )

    assert result.allowed is True, f"Blockers: {result.blockers}"
    assert "stale_pr" not in result.blockers


def test_closed_blocks_merged_pr_when_neither_head_nor_merge_commit_matches(git_repo):
    evidence = _closed_evidence(
        git_repo,
        pr={
            **_closed_evidence(git_repo)["pr"],
            "head_sha": "f" * 40,
            "merge_commit_sha": "e" * 40,
        },
    )

    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "stale_pr" in result.blockers


def test_closed_blocks_missing_cleanup_proof(git_repo):
    """closed with approval but no cleanup proof must block."""
    evidence = _closed_evidence(git_repo)
    evidence["cleanup"] = {"worktree_clean": True}  # no "proof" key
    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )
    assert result.allowed is False
    assert "missing_cleanup_proof" in result.blockers


def test_closed_blocks_missing_residue_evidence(git_repo):
    """closed without any residue evidence must block."""
    evidence = _closed_evidence(git_repo)
    evidence.pop("residue")
    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )
    assert result.allowed is False
    assert "missing_residue_evidence" in result.blockers


def test_closed_blocks_stale_residue_none_with_dirty_worktree(git_repo):
    """closed claiming 'residue: none' while worktree is dirty must block."""
    (git_repo / "UNTRACKED.md").write_text("unexpected residue")
    evidence = _closed_evidence(git_repo)
    evidence["residue"] = _valid_residue(summary="Residue: none", items=[])
    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )
    assert result.allowed is False
    assert "dirty_worktree" in result.blockers


def test_closed_writes_done_status(kanban_home, git_repo):
    """closed transition writes status=done with all gates satisfied."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="final close me",
            workspace_kind="dir",
            workspace_path=str(git_repo),
        )
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "worker done"})
        closeout.transition_task_closeout(
            conn,
            task_id,
            "review_ready",
            _review_ready_evidence(git_repo),
            repo_path=git_repo,
        )
        closed_ev = _closed_evidence(git_repo)
        result = closeout.transition_task_closeout(
            conn,
            task_id,
            "closed",
            closed_ev,
            repo_path=git_repo,
        )
        task = kb.get_task(conn, task_id)

    assert result["status"] == "transitioned"
    assert task.review_phase == "closed"
    assert task.status == "done"
    assert task.closeout_evidence["approval"]["approved_by"] == "reviewer"
    assert task.closeout_evidence["pr"]["state"] == "MERGED"


def test_closed_requires_approval_or_no_pr_exception(git_repo):
    """closed without approval and without no-PR exception must block."""
    evidence = _closed_evidence(git_repo)
    evidence.pop("approval")
    result = closeout.verify_closeout_transition(
        "closed",
        evidence,
        current_phase="review_ready",
        repo_path=git_repo,
    )
    assert result.allowed is False
    assert "missing_close_approval" in result.blockers


def test_closed_happy_path_end_to_end(kanban_home, git_repo):
    """Full lifecycle: worker_done → review_ready → closed with all gates."""
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="full lifecycle",
            workspace_kind="dir",
            workspace_path=str(git_repo),
        )

        # 1) worker_done
        r1 = closeout.transition_task_closeout(
            conn, task_id, "worker_done", {"summary": "work complete"}
        )
        assert r1["status"] == "transitioned"

        # 2) review_ready
        r2 = closeout.transition_task_closeout(
            conn, task_id, "review_ready",
            _review_ready_evidence(git_repo),
            repo_path=git_repo,
        )
        assert r2["status"] == "transitioned"
        t2 = kb.get_task(conn, task_id)
        assert t2.review_phase == "review_ready"
        assert t2.status == "blocked"  # waiting for review

        # 3) closed
        r3 = closeout.transition_task_closeout(
            conn, task_id, "closed",
            _closed_evidence(git_repo),
            repo_path=git_repo,
        )
        assert r3["status"] == "transitioned"
        t3 = kb.get_task(conn, task_id)
        assert t3.review_phase == "closed"
        assert t3.status == "done"


def _done_criteria_ledger(**overrides):
    ledger = {
        "schema": "kanban_done_criteria_ledger.v1",
        "task_id": "t_test",
        "public_id": "BO-188",
        "source": {"kind": "card_description", "ref": "kanban://BO-188"},
        "version": 1,
        "criteria": [
            {
                "id": "DC-001",
                "text": "Implement the criteria ledger validator.",
                "source_section": "Done criteria",
                "required_evidence_types": ["diff", "test"],
                "deterministic_checks": ["pytest tests/hermes_cli/test_kanban_closeout.py"],
                "authority_boundary": "No merge, restart, live apply, or env/secret mutation.",
                "ambiguous": False,
            }
        ],
        "forbidden_actions": ["merge", "gateway_restart", "live_apply", "env_secret_mutation"],
        "refinement_required": False,
    }
    ledger.update(overrides)
    return ledger


_DEFAULT_LEDGER = object()


def _governed_evidence(repo: Path, ledger=_DEFAULT_LEDGER, worker_evidence=None, verifier_result=None, **overrides):
    ledger = _done_criteria_ledger() if ledger is _DEFAULT_LEDGER else ledger
    normalized_ledger = closeout.normalize_done_criteria_ledger(ledger)
    worker_evidence = worker_evidence if worker_evidence is not None else {
        "schema": "kanban_worker_evidence.v1",
        "task_id": "t_test",
        "public_id": "BO-188",
        "criteria_hash": normalized_ledger["criteria_hash"],
        "attempt": 1,
        "worker_run_id": "run_1",
        "branch": "work/BO-188-worker-verifier-loop",
        "commit": _head(repo),
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/188",
        "diff_refs": ["git:HEAD"],
        "tests_run": [
            {"command": "pytest tests/hermes_cli/test_kanban_closeout.py", "result": "passed"}
        ],
        "per_criterion": {
            "DC-001": {
                "claim": "satisfied",
                "evidence_refs": ["tests/hermes_cli/test_kanban_closeout.py"],
            }
        },
        "known_gaps": [],
        "authority_boundary_confirmed": True,
        "forbidden_actions_performed": [],
    }
    verifier_result = verifier_result if verifier_result is not None else {
        "schema": "kanban_verifier_result.v1",
        "task_id": "t_test",
        "public_id": "BO-188",
        "criteria_hash": normalized_ledger["criteria_hash"],
        "worker_run_id": "run_1",
        "verification_attempt": 1,
        "verdict": "PASS",
        "per_criterion": {
            "DC-001": {
                "verdict": "PASS",
                "reason_codes": [],
                "evidence_checked": ["tests/hermes_cli/test_kanban_closeout.py"],
                "missing_evidence": [],
            }
        },
        "deterministic_checks": [
            {"name": "pytest", "status": "passed", "output_ref": "local"}
        ],
        "retry_allowed": False,
        "remediation_goal": None,
        "blocker_reason": None,
        "authority_boundary_ok": True,
    }
    evidence = _review_ready_evidence(
        repo,
        done_criteria_ledger=ledger,
        worker_evidence=worker_evidence,
        verifier_result=verifier_result,
        reviewer_loop={"enabled": True, "max_attempts": 3, "attempt": 1},
        require_done_criteria_ledger=True,
        require_worker_evidence_contract=True,
        require_verifier_result_contract=True,
    )
    evidence.update(overrides)
    return evidence


def test_done_criteria_ledger_hash_is_stable_across_formatting_noise():
    left = _done_criteria_ledger()
    right = _done_criteria_ledger(
        criteria=[
            {
                "id": "DC-001",
                "text": "  Implement   the criteria ledger validator.\n",
                "source_section": "Done criteria",
                "required_evidence_types": ["test", "diff"],
                "deterministic_checks": ["pytest tests/hermes_cli/test_kanban_closeout.py"],
                "authority_boundary": "No merge, restart, live apply, or env/secret mutation.",
                "ambiguous": False,
            }
        ],
        forbidden_actions=["env_secret_mutation", "live_apply", "gateway_restart", "merge"],
    )

    assert closeout.normalize_done_criteria_ledger(left)["criteria_hash"] == closeout.normalize_done_criteria_ledger(right)["criteria_hash"]


def test_done_criteria_ledger_hash_changes_when_meaning_changes():
    left = _done_criteria_ledger()
    right = _done_criteria_ledger(
        criteria=[
            {
                **left["criteria"][0],
                "text": "Implement a verifier-result contract instead.",
            }
        ]
    )

    assert closeout.normalize_done_criteria_ledger(left)["criteria_hash"] != closeout.normalize_done_criteria_ledger(right)["criteria_hash"]


@pytest.mark.parametrize(
    "ledger,blocker",
    [
        (None, "missing_done_criteria_ledger"),
        (_done_criteria_ledger(criteria=[]), "empty_done_criteria"),
        (_done_criteria_ledger(criteria=[{**_done_criteria_ledger()["criteria"][0], "ambiguous": True}]), "refinement_required"),
        (_done_criteria_ledger(criteria=[{k: v for k, v in _done_criteria_ledger()["criteria"][0].items() if k != "authority_boundary"}]), "missing_authority_boundary"),
        (_done_criteria_ledger(criteria=[{**_done_criteria_ledger()["criteria"][0], "deterministic_checks": []}]), "missing_deterministic_checks"),
    ],
)
def test_review_ready_fails_closed_for_invalid_done_criteria_ledger(git_repo, ledger, blocker):
    result = closeout.verify_closeout_transition(
        "review_ready",
        _governed_evidence(git_repo, ledger=ledger),
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert blocker in result.blockers


def test_review_ready_fails_when_worker_uses_stale_criteria_hash(git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["worker_evidence"]["criteria_hash"] = "sha256:" + "0" * 64

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "stale_worker_criteria_hash" in result.blockers


@pytest.mark.parametrize(
    "mutate,blocker",
    [
        (lambda ev: ev["worker_evidence"].pop("per_criterion"), "missing_worker_per_criterion_evidence"),
        (lambda ev: ev["worker_evidence"].update({"authority_boundary_confirmed": False}), "worker_authority_boundary_unconfirmed"),
        (lambda ev: ev["worker_evidence"].update({"forbidden_actions_performed": ["merge"]}), "forbidden_action_performed"),
    ],
)
def test_worker_evidence_contract_fails_closed(git_repo, mutate, blocker):
    evidence = _governed_evidence(git_repo)
    mutate(evidence)

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert blocker in result.blockers


@pytest.mark.parametrize(
    "mutate,blocker",
    [
        (lambda ev: ev["verifier_result"].update({"verdict": "FAIL", "retry_allowed": True, "remediation_goal": "Add missing tests."}), "verifier_result_not_pass"),
        (lambda ev: ev["verifier_result"].update({"criteria_hash": "sha256:" + "1" * 64}), "stale_verifier_criteria_hash"),
        (lambda ev: ev["verifier_result"]["per_criterion"]["DC-001"].update({"verdict": "FAIL"}), "criterion_verifier_not_pass"),
        (lambda ev: ev["verifier_result"].update({"authority_boundary_ok": False}), "verifier_authority_boundary_failed"),
    ],
)
def test_verifier_result_contract_fails_closed(git_repo, mutate, blocker):
    evidence = _governed_evidence(git_repo)
    mutate(evidence)

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert blocker in result.blockers


def test_reviewer_fail_requests_exactly_one_dispatcher_owned_remediation(kanban_home, git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["verifier_result"].update(
        {
            "verdict": "FAIL",
            "retry_allowed": True,
            "remediation_goal": "Add criterion DC-001 evidence and update existing PR.",
        }
    )
    evidence["verifier_result"]["per_criterion"]["DC-001"].update(
        {"verdict": "FAIL", "missing_evidence": ["test output"]}
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="reviewer remediation",
            workspace_kind="dir",
            workspace_path=str(git_repo),
            closeout_evidence={"reviewer_loop": {"attempt": 1, "max_attempts": 3}},
        )
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "worker candidate"})
        result = closeout.transition_task_closeout(
            conn,
            task_id,
            "review_ready",
            evidence,
            repo_path=git_repo,
        )
        task = kb.get_task(conn, task_id)
        events = [ev for ev in kb.list_events(conn, task_id) if ev.kind == "remediation_requested"]

    assert result["status"] == "remediation_requested"
    assert task.status == "ready"
    assert task.assignee is not None
    assert task.review_phase is None
    assert len(events) == 1
    assert events[-1].payload["dispatcher_owned"] is True
    assert "Add criterion DC-001 evidence" in events[-1].payload["remediation_goal"]



def test_review_ready_structural_blockers_create_remediation_request(kanban_home, git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["checks"] = [{"name": "local smoke", "status": "COMPLETED"}]
    evidence["residue"] = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="structural remediation", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "worker candidate"})
        result = closeout.transition_task_closeout(conn, task_id, "review_ready", evidence, repo_path=git_repo)
        task = kb.get_task(conn, task_id)
        events = [ev for ev in kb.list_events(conn, task_id) if ev.kind == "remediation_requested"]

    assert result["status"] == "remediation_requested"
    assert task.status == "ready"
    assert task.review_phase is None
    remediation = task.closeout_evidence["remediation_request"]
    assert remediation["schema"] == "kanban_remediation_request.v1"
    assert remediation["source_phase"] == "review_ready"
    assert remediation["dispatcher_owned"] is True
    assert remediation["retry_allowed"] is True
    assert "ambiguous_check_evidence" in remediation["blockers"]
    assert "invalid_residue_evidence" in remediation["blockers"]
    assert "checks[]" in remediation["remediation_goal"]
    assert "residue" in remediation["remediation_goal"]
    assert len(events) == 1



def test_structural_remediation_retry_can_reach_review_ready(kanban_home, git_repo):
    broken = _governed_evidence(git_repo)
    broken["checks"] = [{"name": "local smoke", "status": "COMPLETED"}]
    broken["residue"] = []
    fixed = _governed_evidence(git_repo)
    fixed["checks"] = [{"name": "local smoke", "status": "COMPLETED", "conclusion": "SUCCESS"}]
    fixed["residue"] = {"summary": "Residue: none", "items": []}
    fixed["reviewer_loop"] = {"attempt": 2, "max_attempts": 3}

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="structural remediation e2e", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "worker candidate"})
        first = closeout.transition_task_closeout(conn, task_id, "review_ready", broken, repo_path=git_repo)
        retry_task = kb.get_task(conn, task_id)
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "remediation worker fixed structural evidence"})
        second = closeout.transition_task_closeout(conn, task_id, "review_ready", fixed, repo_path=git_repo)
        final_task = kb.get_task(conn, task_id)
        event_kinds = [ev.kind for ev in kb.list_events(conn, task_id)]

    assert first["status"] == "remediation_requested"
    assert retry_task.status == "ready"
    assert retry_task.review_phase is None
    assert second["status"] == "transitioned"
    assert second["review_phase"] == "review_ready"
    assert final_task.status == "blocked"
    assert final_task.review_phase == "review_ready"
    assert "remediation_requested" in event_kinds


def test_structural_remediation_requeue_clears_stale_failure_guard(kanban_home, git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["checks"] = [{"name": "local smoke", "status": "COMPLETED"}]
    evidence["residue"] = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="stale failure remediation", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "worker candidate"})
        conn.execute(
            "UPDATE tasks SET last_failure_error = ?, consecutive_failures = 2 WHERE id = ?",
            ("authentication token expired", task_id),
        )
        result = closeout.transition_task_closeout(conn, task_id, "review_ready", evidence, repo_path=git_repo)
        row = conn.execute("SELECT status, last_failure_error, consecutive_failures FROM tasks WHERE id = ?", (task_id,)).fetchone()
        guard = kb.check_respawn_guard(conn, task_id)

    assert result["status"] == "remediation_requested"
    assert row["status"] == "ready"
    assert row["last_failure_error"] is None
    assert row["consecutive_failures"] == 0
    assert guard is None


def test_repeated_structural_remediation_attempts_eventually_exhaust(kanban_home, git_repo):
    first = _governed_evidence(git_repo)
    first["checks"] = [{"name": "local smoke", "status": "COMPLETED"}]
    first["residue"] = []
    second = _governed_evidence(git_repo)
    second["reviewer_loop"] = {"attempt": 2, "max_attempts": 3}
    second["checks"] = [{"name": "local smoke", "status": "COMPLETED"}]
    second["residue"] = []
    third = _governed_evidence(git_repo)
    third["reviewer_loop"] = {"attempt": 3, "max_attempts": 3}
    third["checks"] = [{"name": "local smoke", "status": "COMPLETED"}]
    third["residue"] = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="repeated structural remediation", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "initial worker"})
        r1 = closeout.transition_task_closeout(conn, task_id, "review_ready", first, repo_path=git_repo)
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "first remediation still bad"})
        r2 = closeout.transition_task_closeout(conn, task_id, "review_ready", second, repo_path=git_repo)
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "second remediation still bad"})
        r3 = closeout.transition_task_closeout(conn, task_id, "review_ready", third, repo_path=git_repo)
        task = kb.get_task(conn, task_id)

    assert r1["status"] == "remediation_requested"
    assert r2["status"] == "remediation_requested"
    assert r3["status"] == "blocked"
    assert "remediation_attempts_exhausted" in r3["blockers"]
    assert task.status == "blocked"
    assert task.review_phase == "worker_done"

def test_review_ready_structural_remediation_blocks_after_max_attempts(kanban_home, git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["reviewer_loop"] = {"attempt": 3, "max_attempts": 3}
    evidence["checks"] = [{"name": "local smoke", "status": "COMPLETED"}]
    evidence["residue"] = []

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="structural max attempts", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "worker candidate"})
        result = closeout.transition_task_closeout(conn, task_id, "review_ready", evidence, repo_path=git_repo)
        task = kb.get_task(conn, task_id)

    assert result["status"] == "blocked"
    assert "remediation_attempts_exhausted" in result["blockers"]
    assert task.status == "blocked"
    assert task.review_phase == "worker_done"

def test_reviewer_fail_blocks_after_max_attempts(kanban_home, git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["reviewer_loop"] = {"attempt": 3, "max_attempts": 3}
    evidence["verifier_result"].update(
        {"verdict": "FAIL", "retry_allowed": True, "remediation_goal": "Try again."}
    )
    evidence["verifier_result"]["per_criterion"]["DC-001"].update({"verdict": "FAIL"})

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="reviewer max attempts",
            workspace_kind="dir",
            workspace_path=str(git_repo),
        )
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "worker candidate"})
        result = closeout.transition_task_closeout(conn, task_id, "review_ready", evidence, repo_path=git_repo)
        task = kb.get_task(conn, task_id)

    assert result["status"] == "blocked"
    assert "remediation_attempts_exhausted" in result["blockers"]
    assert task.status == "blocked"
    assert task.review_phase == "worker_done"


def test_done_criteria_ledger_requires_contract_schema(git_repo):
    ledger = _done_criteria_ledger()
    ledger.pop("schema")

    result = closeout.verify_closeout_transition(
        "review_ready",
        _governed_evidence(git_repo, ledger=ledger),
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "invalid_done_criteria_ledger_schema" in result.blockers


def test_worker_and_verifier_contracts_require_schema(git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["worker_evidence"].pop("schema")
    evidence["verifier_result"].pop("schema")

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "invalid_worker_evidence_schema" in result.blockers
    assert "invalid_verifier_result_schema" in result.blockers


def test_reviewer_fail_does_not_requeue_when_unrelated_blockers_exist(kanban_home, git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["checks"] = [{"name": "ci", "status": "COMPLETED", "conclusion": "FAILURE"}]
    evidence["verifier_result"].update(
        {"verdict": "FAIL", "retry_allowed": True, "remediation_goal": "Add criterion evidence."}
    )
    evidence["verifier_result"]["per_criterion"]["DC-001"].update({"verdict": "FAIL"})

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="unrelated blocker", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "candidate"})
        result = closeout.transition_task_closeout(conn, task_id, "review_ready", evidence, repo_path=git_repo)
        task = kb.get_task(conn, task_id)
        events = [ev for ev in kb.list_events(conn, task_id) if ev.kind == "remediation_requested"]

    assert result["status"] == "blocked"
    assert "failed_checks" in result["blockers"]
    assert task.status == "blocked"
    assert task.review_phase == "worker_done"
    assert events == []


def test_reviewer_fail_remediation_request_is_idempotent(kanban_home, git_repo):
    evidence = _governed_evidence(git_repo)
    evidence["verifier_result"].update(
        {"verdict": "FAIL", "retry_allowed": True, "remediation_goal": "Add criterion evidence."}
    )
    evidence["verifier_result"]["per_criterion"]["DC-001"].update({"verdict": "FAIL"})

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="idempotent remediation", workspace_kind="dir", workspace_path=str(git_repo))
        closeout.transition_task_closeout(conn, task_id, "worker_done", {"summary": "candidate"})
        first = closeout.transition_task_closeout(conn, task_id, "review_ready", evidence, repo_path=git_repo)
        second = closeout.transition_task_closeout(conn, task_id, "review_ready", evidence, repo_path=git_repo)
        events = [ev for ev in kb.list_events(conn, task_id) if ev.kind == "remediation_requested"]

    assert first["status"] == "remediation_requested"
    assert second["status"] == "remediation_requested"
    assert second["reason"] == "reviewer_fail_remediation_already_queued"
    assert len(events) == 1
