"""Behavioral contracts for git-backed kanban completion facts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hermes_cli import kanban_completion_facts as facts
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    )
    return proc.stdout.strip()


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "base")
    return path


def _running_worktree(conn, repo: Path, title: str):
    tid = kb.create_task(
        conn, title=title, workspace_kind="worktree", workspace_path=str(repo),
        branch_name=_git(repo, "branch", "--show-current"),
    )
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    conn.commit()
    task = kb.claim_task(conn, tid, claimer="test")
    assert task is not None and task.current_run_id is not None
    kbw.set_workspace_path(conn, tid, repo)
    assert facts.record_workspace_baseline(conn, tid, repo)
    return tid, task.current_run_id


def test_worktree_completion_rejects_zero_commit_and_dirty_claims_but_audits_override(
    kanban_home, tmp_path,
):
    with kbc.connect() as conn:
        zero_repo = _repo(tmp_path / "zero")
        zero, zero_run = _running_worktree(conn, zero_repo, "fix(parser): handle empty input")
        assert not kb.complete_task(
            conn, zero, summary="Implemented the parser fix", expected_run_id=zero_run,
        )
        assert kb.get_task(conn, zero).status == "running"
        assert "zero commits" in kb.get_task(conn, zero).last_failure_error

        dirty_repo = _repo(tmp_path / "dirty")
        dirty, dirty_run = _running_worktree(conn, dirty_repo, "Investigate parser behavior")
        (dirty_repo / "tracked.txt").write_text("uncommitted\n", encoding="utf-8")
        assert not kb.complete_task(
            conn, dirty, summary="Investigation complete", expected_run_id=dirty_run,
        )
        assert "uncommitted file" in kb.get_task(conn, dirty).last_failure_error
        assert kb.complete_task(
            conn, dirty, summary="Intentional local-only investigation",
            expected_run_id=dirty_run, override_git_facts="operator accepted an uncommitted probe",
        )
        run = kb.latest_run(conn, dirty)
        receipt = run.metadata["completion_facts"]
        assert receipt["dirty_file_count"] == 1
        assert receipt["override_reason"] == "operator accepted an uncommitted probe"


def test_push_claim_requires_exact_remote_ref_and_persists_receipt(kanban_home, tmp_path):
    repo = _repo(tmp_path / "repo")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
    _git(repo, "remote", "add", "origin", str(remote))

    with kbc.connect() as conn:
        tid, run_id = _running_worktree(conn, repo, "feat(kanban): persist completion facts")
        (repo / "tracked.txt").write_text("implemented\n", encoding="utf-8")
        _git(repo, "add", "tracked.txt")
        _git(repo, "commit", "-m", "implement")
        head = _git(repo, "rev-parse", "HEAD")
        branch = _git(repo, "branch", "--show-current")
        summary = "Implemented and pushed completion fact validation"

        assert not kb.complete_task(conn, tid, summary=summary, expected_run_id=run_id)
        assert "git_refs_pushed" in kb.get_task(conn, tid).last_failure_error

        _git(repo, "push", "origin", f"HEAD:refs/heads/{branch}")
        metadata = {"git_refs_pushed": [{
            "remote": "origin", "ref": f"refs/heads/{branch}", "sha": head,
        }]}
        assert kb.complete_task(
            conn, tid, summary=summary, metadata=metadata, expected_run_id=run_id,
        )
        run = kb.latest_run(conn, tid)
        receipt = run.metadata["completion_facts"]
        assert receipt["commits_ahead"] == 1
        assert receipt["dirty_file_count"] == 0
        assert receipt["pushed_refs"] == [{
            "remote": "origin", "ref": f"refs/heads/{branch}",
            "claimed_sha": head, "remote_sha": head,
        }]
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='completion_facts' "
            "ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert json.loads(event["payload"])["ok"] is True
