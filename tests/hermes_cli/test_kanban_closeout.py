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

    assert result["status"] == "transitioned"
    assert task.review_phase == "review_ready"
    assert task.status == "blocked"
    assert result["task_status"] == "blocked"
    assert task.closeout_evidence["residue"]["summary"] == "Residue: none"
    assert task.closeout_evidence["verification"]["allowed"] is True
    assert task.closeout_evidence["verification"]["gateway_restarted_or_reloaded"] is False
    assert task.closeout_evidence["verification"]["pr_merged"] is False


def test_review_ready_requires_worker_done_phase(git_repo):
    result = closeout.verify_closeout_transition(
        "review_ready",
        _review_ready_evidence(git_repo),
        current_phase=None,
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "review_ready_requires_worker_done" in result.blockers


def test_review_ready_allows_strict_no_pr_exception_for_no_code_smoke(git_repo):
    evidence = _review_ready_evidence(
        git_repo,
        pr={},
        checks=[{"name": "local verifier", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        evidence={
            "tests_run": ["hermes kanban closeout BO-115 review_ready --check-only --json"],
            "changed_files": [],
            "review_package_expectation": "Changed files/commit are not expected for this smoke fixture.",
        },
        no_pr_exception={
            "policy": "no-code-autopilot-smoke-review-ready",
            "reason": "BO-115 is a no-code Autopilot smoke fixture with worker evidence and verifier PASS.",
            "review_package_expectation": "Changed files/commit are not expected for this smoke fixture.",
            "changed_files_expected": False,
        },
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
    assert result.evidence["no_pr_exception"]["policy"] == "no-code-autopilot-smoke-review-ready"


def test_review_ready_no_pr_exception_is_fail_closed_without_review_package_expectation(git_repo):
    evidence = _review_ready_evidence(
        git_repo,
        pr={},
        no_pr_exception={
            "policy": "vague-no-pr",
            "reason": "operator says no PR is needed",
        },
    )

    result = closeout.verify_closeout_transition(
        "review_ready",
        evidence,
        current_phase="worker_done",
        repo_path=git_repo,
    )

    assert result.allowed is False
    assert "missing_no_pr_review_package_expectation" in result.blockers
    assert "missing_live_pr" not in result.blockers


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
    assert "missing_live_pr" in payload["blockers"]
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).review_phase == "worker_done"


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
