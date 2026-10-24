"""Tests for worktree workspace teardown at task completion/archive.

Covers the ownership gap where kanban ``worktree`` workspaces were never
reaped by anything: ``_cleanup_workspace`` preserved them by design, the CLI
startup pruner explicitly skips ``t_*`` worktrees ("dispatcher-driven
lifecycle"), and ``kanban gc`` only swept scratch. A completed or archived
task's linked worktree is now removed when — and only when — it provably
holds no work: clean working tree and every commit reachable from a
remote-tracking ref. Any doubt preserves the worktree.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_db_connect as kbc


def _git(*args: str, cwd: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A project repo with a remote whose history is fully pushed."""
    origin = tmp_path / "origin.git"
    _git("init", "--bare", str(origin))
    project = tmp_path / "project"
    _git("clone", str(origin), str(project))
    _git("-C", str(project), "config", "user.email", "t@example.com")
    _git("-C", str(project), "config", "user.name", "t")
    (project / "README.md").write_text("hello\n", encoding="utf-8")
    _git("-C", str(project), "add", "README.md")
    _git("-C", str(project), "commit", "-m", "init")
    _git("-C", str(project), "push", "origin", "HEAD")
    return project


def _make_worktree(repo: Path, task_id: str, branch: str | None = None) -> Path:
    target = repo / ".worktrees" / task_id
    kbw._ensure_git_worktree(repo, target, branch or f"wt/{task_id}")
    return target


def _branch_exists(repo: Path, branch: str) -> bool:
    out = _git("-C", str(repo), "branch", "--list", branch)
    return bool(out.strip())


# ---------------------------------------------------------------------------
# _cleanup_worktree_workspace unit behavior
# ---------------------------------------------------------------------------


def test_clean_pushed_worktree_removed(repo: Path) -> None:
    wt = _make_worktree(repo, "t_aaaa1111")
    kbw._cleanup_worktree_workspace("t_aaaa1111", str(wt))
    assert not wt.exists()
    # auto-generated task branch goes with it
    assert not _branch_exists(repo, "wt/t_aaaa1111")
    # main checkout untouched
    assert (repo / "README.md").exists()


def test_cleanup_leaves_a_worktree_cwd_before_removal(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows cannot remove a worktree that is the process's current directory."""
    wt = _make_worktree(repo, "t_cwd112425")
    real_git = kbw._git

    def windows_git(repo_root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
        if args[:2] == ("worktree", "remove") and Path.cwd().is_relative_to(wt):
            return subprocess.CompletedProcess(
                ["git", *args], 1, stderr="Permission denied: current directory"
            )
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", windows_git)
    monkeypatch.chdir(wt)
    kbw._cleanup_worktree_workspace("t_cwd112425", str(wt))

    assert Path.cwd() == repo
    assert not wt.exists()


def test_cleanup_proceeds_when_cwd_was_deleted(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deferred parent cleanup (#33774) runs after the child's scratch cwd was
    rmtree'd; a dead cwd must not preserve a clean, pushed worktree."""
    wt = _make_worktree(repo, "t_deadcwd113073")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    scratch.rmdir()

    kbw._cleanup_worktree_workspace("t_deadcwd113073", str(wt))

    assert not wt.exists()
    assert not _branch_exists(repo, "wt/t_deadcwd113073")


def test_cleanup_retries_worktree_removal_once(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brief Windows directory-handle delay gets one safe retry."""
    wt = _make_worktree(repo, "t_retry112425")
    real_git = kbw._git
    attempts = 0

    def delayed_remove(repo_root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
        nonlocal attempts
        if args[:2] == ("worktree", "remove"):
            attempts += 1
            if attempts == 1:
                return subprocess.CompletedProcess(
                    ["git", *args], 1, stderr="Permission denied: handle pending"
                )
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", delayed_remove)
    monkeypatch.setattr(kbw.time, "sleep", lambda _delay: None)
    kbw._cleanup_worktree_workspace("t_retry112425", str(wt))

    assert attempts == 2
    assert not wt.exists()


def test_dirty_worktree_preserved(repo: Path) -> None:
    wt = _make_worktree(repo, "t_bbbb2222")
    (wt / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
    kbw._cleanup_worktree_workspace("t_bbbb2222", str(wt))
    assert wt.is_dir()
    assert (wt / "wip.txt").exists()


def test_unpushed_commits_preserved(repo: Path) -> None:
    wt = _make_worktree(repo, "t_cccc3333")
    (wt / "work.txt").write_text("committed but not pushed\n", encoding="utf-8")
    _git("-C", str(wt), "add", "work.txt")
    _git("-C", str(wt), "commit", "-m", "local work")
    kbw._cleanup_worktree_workspace("t_cccc3333", str(wt))
    assert wt.is_dir()


def test_custom_branch_survives_worktree_removal(repo: Path) -> None:
    wt = _make_worktree(repo, "t_dddd4444", branch="feature/custom")
    kbw._cleanup_worktree_workspace("t_dddd4444", str(wt), "feature/custom")
    assert not wt.exists()
    # only auto-generated wt/* branches are deleted
    assert _branch_exists(repo, "feature/custom")


def test_main_checkout_never_removed(repo: Path) -> None:
    kbw._cleanup_worktree_workspace("t_eeee5555", str(repo))
    assert repo.is_dir()
    assert (repo / "README.md").exists()


def test_non_git_dir_preserved(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-worktree"
    plain.mkdir()
    kbw._cleanup_worktree_workspace("t_ffff6666", str(plain))
    assert plain.is_dir()


def test_tree_dirtied_between_check_and_removal_preserved(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOCTOU: a tree that becomes dirty after the pre-check is NOT removed.

    Simulates the race by making the pre-check see a clean tree while the
    tree is actually dirty when ``git worktree remove`` runs. Without
    ``--force``, git's own dirty guard re-verifies at removal time and the
    removal fails safe.
    """
    wt = _make_worktree(repo, "t_gggg7777")
    (wt / "late-wip.txt").write_text("dirtied after the check\n", encoding="utf-8")
    # Pre-check lies (as if the file appeared just after it ran) — real git
    # must still refuse the removal.
    from hermes_cli import worktree_ops

    monkeypatch.setattr(worktree_ops, "_worktree_is_dirty", lambda _p, *_a, **_k: False)
    kbw._cleanup_worktree_workspace("t_gggg7777", str(wt))
    assert wt.is_dir()
    assert (wt / "late-wip.txt").exists()


# ---------------------------------------------------------------------------
# Lifecycle integration: complete / archive / deferred parents
# ---------------------------------------------------------------------------


def _worktree_task(conn, repo: Path, title: str = "wt-task") -> tuple[str, Path]:
    tid = kb.create_task(conn, title=title, assignee="worker")
    wt = _make_worktree(repo, tid)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', workspace_path=?, "
            "branch_name=? WHERE id=?",
            (str(wt), f"wt/{tid}", tid),
        )
    return tid, wt


def test_complete_task_reaps_clean_worktree(kanban_home: Path, repo: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.complete_task(conn, tid, summary="done")
    assert not wt.exists()
    assert not _branch_exists(repo, f"wt/{tid}")


def test_complete_task_preserves_dirty_worktree(kanban_home: Path, repo: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        (wt / "wip.txt").write_text("unsaved\n", encoding="utf-8")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.complete_task(conn, tid, summary="done")
    assert wt.is_dir()
    assert (wt / "wip.txt").exists()


def test_archive_task_reaps_clean_worktree(kanban_home: Path, repo: Path) -> None:
    with kbc.connect_closing() as conn:
        tid, wt = _worktree_task(conn, repo)
        assert kb.archive_task(conn, tid)
    assert not wt.exists()


def test_parent_worktree_deferred_until_children_done(
    kanban_home: Path, repo: Path
) -> None:
    with kbc.connect_closing() as conn:
        parent, parent_wt = _worktree_task(conn, repo, title="parent")
        child = kb.create_task(conn, title="child", assignee="worker")
        kb.link_tasks(conn, parent, child)

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        assert kb.claim_task(conn, parent, claimer="worker") is not None
        assert kb.complete_task(conn, parent, summary="parent done")
        # child still active -> parent worktree must survive for handoff
        assert parent_wt.is_dir()

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        assert kb.claim_task(conn, child, claimer="worker") is not None
        assert kb.complete_task(conn, child, summary="child done")
    # last child terminal -> deferred parent worktree reaped
    assert not parent_wt.exists()


def _running_scratch_parent(conn, title: str = "parent") -> tuple[str, Path]:
    tid = kb.create_task(conn, title=title)
    assert kb.claim_task(conn, tid) is not None  # -> running
    ws = Path(kbw.resolve_workspace(kb.get_task(conn, tid)))
    kbw.set_workspace_path(conn, tid, ws)
    return tid, ws


@pytest.mark.parametrize("live_status", ["running", "blocked", "review"])
def test_live_parent_scratch_survives_last_child_archived(
    kanban_home: Path, live_status: str
) -> None:
    """#133501: the deferred sweep must not reap a still-live parent's scratch.

    ``_try_cleanup_parent_workspaces`` (#33774) exists for a parent that is
    already terminal while children still need its handoff files. A parent
    still running/blocked/review owns its workspace until its own terminal
    transition, so archiving its last active child must not delete it.
    """
    with kbc.connect_closing() as conn:
        parent, ws = _running_scratch_parent(conn)
        (ws / "work.txt").write_text("in progress", encoding="utf-8")
        child = kb.create_task(conn, title="probe")
        kb.link_tasks(conn, parent, child)
        if live_status == "blocked":
            assert kb.block_task(conn, parent, reason="waiting")
        elif live_status == "review":
            assert kb.request_review(conn, parent, summary="please review", force=True)
        assert kb.archive_task(conn, child)
        assert kb.get_task(conn, parent).status == live_status
    assert (ws / "work.txt").is_file()


def test_live_parent_worktree_survives_last_child_archived(
    kanban_home: Path, repo: Path
) -> None:
    """#133501, worktree flavour: a running parent's clean linked worktree —
    its worker's cwd — must not be removed because a child got archived."""
    with kbc.connect_closing() as conn:
        parent, parent_wt = _worktree_task(conn, repo, title="live parent")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        assert kb.claim_task(conn, parent, claimer="worker") is not None
        child = kb.create_task(conn, title="probe")
        kb.link_tasks(conn, parent, child)
        assert kb.archive_task(conn, child)
        assert kb.get_task(conn, parent).status == "running"
    assert parent_wt.is_dir()


def test_terminal_scratch_parent_still_swept_after_last_child(
    kanban_home: Path,
) -> None:
    """#33774 unchanged by the #133501 guard: a parent that finished while a
    child was still active still gets its scratch reaped by the last child's
    completion."""
    with kbc.connect_closing() as conn:
        parent, ws = _running_scratch_parent(conn)
        (ws / "handoff.txt").write_text("x", encoding="utf-8")
        child = kb.create_task(conn, title="probe")
        kb.link_tasks(conn, parent, child)
        assert kb.complete_task(conn, parent, summary="parent done")
        assert ws.is_dir()  # live child still needs the handoff files
        assert kb.archive_task(conn, child)
    assert not ws.exists()


# ---------------------------------------------------------------------------
# Merged-PR reconciliation
# ---------------------------------------------------------------------------


def _stale_pr_worktree(conn, repo: Path) -> tuple[str, Path, str, str]:
    """Terminal task whose branch head is no longer remote-tracked."""
    tid, wt = _worktree_task(conn, repo, title="merged-pr cleanup")
    branch = f"wt/{tid}"
    (wt / "change.txt").write_text("merged work\n", encoding="utf-8")
    _git("-C", str(wt), "add", "change.txt")
    _git("-C", str(wt), "commit", "-m", "PR change")
    head = _git("-C", str(wt), "rev-parse", "HEAD").strip()
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='done', completed_at=?, completion_contract=? WHERE id=?",
            (
                int(__import__("time").time()),
                "https://github.com/acme/repo/pull/42",
                tid,
            ),
        )
    return tid, wt, branch, head


def _merged_pr(branch: str, head: str) -> dict:
    return {
        "number": 42,
        "url": "https://github.com/acme/repo/pull/42",
        "mergedAt": "2026-10-24T00:00:00Z",
        "state": "MERGED",
        "headRefName": branch,
        "headRefOid": head,
        "headRepository": {"nameWithOwner": "acme/repo"},
    }


def _configure_pr_reaper(monkeypatch: pytest.MonkeyPatch, proof) -> None:
    monkeypatch.setattr(kbw, "_terminal_worker_settled", lambda *_a, **_k: True)
    monkeypatch.setattr(kbw, "_origin_github_repo", lambda *_a, **_k: "acme/repo")
    monkeypatch.setattr(kbw, "_view_pr", lambda *_a, **_k: proof)


def test_merged_pr_exact_head_reaps_terminal_worktree(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        tid, wt, branch, head = _stale_pr_worktree(conn, repo)
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, head))

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == [tid]

        assert not wt.exists()
        assert not _branch_exists(repo, branch)
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='workspace_cleanup_pr_merged'",
            (tid,),
        ).fetchone()
        assert event is not None


@pytest.mark.parametrize("proof", [{"state": "OPEN"}, None])
def test_open_pr_or_github_failure_preserves_worktree(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, proof
) -> None:
    with kbc.connect_closing() as conn:
        tid, wt, _branch, _head = _stale_pr_worktree(conn, repo)
        _configure_pr_reaper(monkeypatch, proof)

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()
        assert kb.get_task(conn, tid).status == "done"


def test_merged_pr_head_mismatch_preserves_reused_branch(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _tid, wt, branch, _head = _stale_pr_worktree(conn, repo)
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, "0" * 40))

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()


def test_merged_pr_dirty_worktree_is_still_preserved(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _tid, wt, branch, head = _stale_pr_worktree(conn, repo)
        (wt / "late-wip.txt").write_text("not in the PR\n", encoding="utf-8")
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, head))

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert (wt / "late-wip.txt").is_file()


def test_merged_pr_shared_worktree_is_preserved(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _tid, wt, branch, head = _stale_pr_worktree(conn, repo)
        sharer = kb.create_task(conn, title="still using it", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', workspace_kind='worktree', workspace_path=? WHERE id=?",
                (str(wt), sharer),
            )
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, head))

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()


def test_conflicting_open_pr_schedules_idempotent_alignment_task(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        tid, wt, branch, head = _stale_pr_worktree(conn, repo)
        proof = _merged_pr(branch, head)
        proof.update({"state": "OPEN", "mergedAt": None, "mergeable": "CONFLICTING"})
        _configure_pr_reaper(monkeypatch, proof)

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()
        child = conn.execute(
            "SELECT id, status, workspace_path, branch_name, completion_contract "
            "FROM tasks WHERE created_by='kanban-pr-controller'"
        ).fetchone()
        assert child is not None
        assert child["status"] == "ready"
        assert child["workspace_path"] == str(wt)
        assert child["branch_name"] == branch
        assert child["completion_contract"] == "https://github.com/acme/repo/pull/42"

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by='kanban-pr-controller'"
        ).fetchone()[0] == 1
        assert kb.get_task(conn, tid).status == "done"


def test_terminal_worker_must_be_settled_before_pr_reconciliation(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _tid, wt, branch, head = _stale_pr_worktree(conn, repo)
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, head))
        monkeypatch.setattr(kbw, "_terminal_worker_settled", lambda *_a, **_k: False)

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()


def test_terminal_sharer_worker_must_also_be_settled(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        source, wt, branch, head = _stale_pr_worktree(conn, repo)
        sharer = kb.create_task(conn, title="terminal alignment", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='worktree', workspace_path=? WHERE id=?",
                (str(wt), sharer),
            )
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, head))
        monkeypatch.setattr(
            kbw,
            "_terminal_worker_settled",
            lambda _conn, task_id, _now: task_id == source,
        )

        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()


def test_identity_change_during_pr_query_preserves_worktree(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _tid, wt, branch, head = _stale_pr_worktree(conn, repo)
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, head))

        def advance_after_snapshot(*_args, **_kwargs):
            (wt / "later.txt").write_text("later commit\n", encoding="utf-8")
            _git("-C", str(wt), "add", "later.txt")
            _git("-C", str(wt), "commit", "-m", "advance after PR snapshot")
            return _merged_pr(branch, head)

        monkeypatch.setattr(kbw, "_view_pr", advance_after_snapshot)
        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()
        assert (wt / "later.txt").is_file()


def test_child_added_after_cleanup_intent_blocks_removal(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        source, wt, branch, head = _stale_pr_worktree(conn, repo)
        _configure_pr_reaper(monkeypatch, _merged_pr(branch, head))
        added: list[str] = []

        def add_child_after_intent(_conn, _task_id):
            if not added:
                added.append(
                    kb.create_task(
                        conn,
                        title="late child",
                        assignee="worker",
                        parents=(source,),
                    )
                )

        monkeypatch.setattr(kbw, "_cleanup_worker_tmux", add_child_after_intent)
        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert wt.is_dir()
        assert added and kb.get_task(conn, added[0]).status == "ready"


def test_ref_lock_blocks_clean_commit_during_worktree_remove(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        tid, wt, branch, head = _stale_pr_worktree(conn, repo)
    real_git = kbw._git
    attempted: list[int] = []

    def advance_at_remove(repo_root: Path, *args: str, timeout: int):
        if args[:2] == ("worktree", "remove") and not attempted:
            result = subprocess.run(
                ["git", "-C", str(wt), "commit", "--allow-empty", "-m", "racing commit"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            attempted.append(result.returncode)
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", advance_at_remove)
    assert kbw._cleanup_worktree_workspace(
        tid, str(wt), branch, merged_pr_head_oid=head,
    )
    assert attempted and attempted[0] != 0
    assert not wt.exists()


def test_concurrent_conflict_reconcilers_create_one_alignment_task(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _source, _wt, branch, head = _stale_pr_worktree(conn, repo)
        row = conn.execute(
            "SELECT id, workspace_path, branch_name, completion_contract, assignee, "
            "project_id, priority FROM tasks LIMIT 1"
        ).fetchone()
    proof = _merged_pr(branch, head)
    proof.update({"state": "OPEN", "mergedAt": None, "mergeable": "CONFLICTING"})

    def schedule() -> str | None:
        with kbc.connect_closing() as other:
            return kbw._schedule_pr_alignment(other, row, proof, head)

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda _n: schedule(), range(2)))
    assert ids[0] == ids[1]
    with kbc.connect_closing() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by='kanban-pr-controller'"
        ).fetchone()[0] == 1


def test_board_write_fences_use_one_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "boards"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.create_board("alpha")
    kb.create_board("beta")
    barrier = threading.Barrier(2)

    def fence(board: str) -> tuple[str, int]:
        with kbc.connect_closing(board=board) as conn:
            barrier.wait(timeout=5)
            with kbw._all_board_write_fences(conn) as (current, siblings):
                assert current.execute("SELECT 1").fetchone()[0] == 1
                return board, len(siblings)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fence, ("alpha", "beta")))
    assert sorted(results) == [("alpha", 1), ("beta", 1)]


def test_topology_fence_blocks_new_board_sharer_until_snapshot_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "boards"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.create_board("alpha")
    started = threading.Event()

    def create_live_sharer() -> str:
        started.set()
        kb.create_board("late")
        with kbc.connect_closing(board="late") as late:
            return kb.create_task(
                late,
                title="late sharer",
                workspace_kind="worktree",
                workspace_path=str(tmp_path / "shared-worktree"),
                assignee="worker",
            )

    with kbc.connect_closing(board="alpha") as conn:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with kbw._all_board_write_fences(conn):
                with pytest.raises(RuntimeError, match="topology is frozen"):
                    kb.create_board("inline")
                future = pool.submit(create_live_sharer)
                assert started.wait(timeout=5)
                with pytest.raises(FutureTimeoutError):
                    future.result(timeout=2.0)
            task_id = future.result(timeout=10)
    with kbc.connect_closing(board="late") as late:
        task = kb.get_task(late, task_id)
        assert task is not None and task.status == "ready"


def test_crash_recovery_requires_git_registry_absence(
    kanban_home: Path, repo: Path
) -> None:
    with kbc.connect_closing() as conn:
        tid, wt, branch, head = _stale_pr_worktree(conn, repo)
        common = kbw._git_common_dir(wt)
        assert common is not None
        intent = {
            "path": str(wt),
            "repo_root": str(repo.resolve()),
            "common_dir": str(common.resolve()),
            "branch": branch,
            "head": head,
        }
        with kb.write_txn(conn):
            kb._append_event(conn, tid, kbw._PR_INTENT_EVENT, intent)
        shutil.rmtree(wt)
        assert not kbw._recover_finished_pr_cleanup(conn, tid, str(wt))
        recovered = kb.get_task(conn, tid)
        assert recovered is not None and recovered.workspace_path == str(wt)
        _git("-C", str(repo), "worktree", "prune")
        assert kbw._recover_finished_pr_cleanup(conn, tid, str(wt))
        recovered = kb.get_task(conn, tid)
        assert recovered is not None and recovered.workspace_path is None


def test_abandoned_controller_ref_lock_is_recovered(
    kanban_home: Path, repo: Path
) -> None:
    with kbc.connect_closing() as conn:
        tid, wt, branch, head = _stale_pr_worktree(conn, repo)
    common = kbw._git_common_dir(wt)
    assert common is not None
    lock_path = common / "refs" / "heads" / f"{branch}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "owner": "hermes-kanban-pr-cleanup",
                "pid": 999_999_999,
                "started_at": 1,
                "branch": branch,
                "expected_head": head,
            }
        ),
        encoding="utf-8",
    )
    assert kbw._cleanup_worktree_workspace(
        tid, str(wt), branch, merged_pr_head_oid=head,
    )
    assert not lock_path.exists()


def test_ref_lock_post_link_verification_fault_cleans_published_inode(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _tid, wt, branch, head = _stale_pr_worktree(conn, repo)
    common = kbw._git_common_dir(wt)
    assert common is not None
    lock_path = common / "refs" / "heads" / f"{branch}.lock"
    real_stat = Path.stat
    injected = False

    def fail_first_lock_stat(path: Path, *args, **kwargs):
        nonlocal injected
        if path == lock_path and not injected:
            injected = True
            raise OSError("simulated post-link verification fault")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_first_lock_stat)
    with pytest.raises(OSError, match="post-link verification"):
        with kbw._branch_ref_write_lock(common, branch, head):
            pass
    assert injected
    assert not lock_path.exists()
    with kbw._branch_ref_write_lock(common, branch, head):
        pass
    assert not lock_path.exists()


def test_ref_recovery_guard_blocks_replacement_cleanup(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _tid, wt, branch, head = _stale_pr_worktree(conn, repo)
    common = kbw._git_common_dir(wt)
    assert common is not None
    lock_path = common / "refs" / "heads" / f"{branch}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "owner": "hermes-kanban-pr-cleanup",
                "pid": 999_999_999,
                "started_at": 1,
                "branch": branch,
                "expected_head": head,
            }
        ),
        encoding="utf-8",
    )
    real_remove = kbw._remove_abandoned_ref_lock_under_guard
    stale_check_entered = threading.Event()
    continue_stale_removal = threading.Event()
    first_holds_ref = threading.Event()
    allow_first_exit = threading.Event()

    def paused_remove(path: Path, guard: object) -> bool:
        stale_check_entered.set()
        assert continue_stale_removal.wait(timeout=5)
        return real_remove(path, guard)

    monkeypatch.setattr(kbw, "_remove_abandoned_ref_lock_under_guard", paused_remove)

    def acquire_first() -> None:
        with kbw._branch_ref_write_lock(common, branch, head):
            first_holds_ref.set()
            assert allow_first_exit.wait(timeout=5)

    def acquire_second() -> None:
        with kbw._branch_ref_write_lock(common, branch, head):
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(acquire_first)
        assert stale_check_entered.wait(timeout=5)
        second = pool.submit(acquire_second)
        with pytest.raises(FutureTimeoutError):
            second.result(timeout=2.0)
        continue_stale_removal.set()
        assert first_holds_ref.wait(timeout=5)
        with pytest.raises(FutureTimeoutError):
            second.result(timeout=2.0)
        allow_first_exit.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert not lock_path.exists()


def test_conflict_alignment_from_failed_source_is_ready_and_archived_key_is_durable(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        source, wt, branch, head = _stale_pr_worktree(conn, repo)
        conn.execute("UPDATE tasks SET status='failed' WHERE id=?", (source,))
        conn.commit()
        proof = _merged_pr(branch, head)
        proof.update({"state": "OPEN", "mergedAt": None, "mergeable": "CONFLICTING"})
        _configure_pr_reaper(monkeypatch, proof)
        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        alignment = conn.execute(
            "SELECT id, status FROM tasks WHERE created_by='kanban-pr-controller'"
        ).fetchone()
        assert alignment and alignment["status"] == "ready"
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (alignment["id"],))
        conn.commit()
        assert kbw.reap_merged_pr_worktrees(conn, probe_interval_seconds=0) == []
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE created_by='kanban-pr-controller'"
        ).fetchone()[0] == 1
        assert wt.is_dir()


def test_reaper_caps_filesystem_prechecks_before_network(
    kanban_home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with kbc.connect_closing() as conn:
        _stale_pr_worktree(conn, repo)
        _stale_pr_worktree(conn, repo)
        calls: list[Path] = []
        monkeypatch.setattr(
            kbw,
            "_is_linked_worktree_checkout",
            lambda path: calls.append(path) or False,
        )
        monkeypatch.setattr(kbw, "_view_pr", lambda *_args, **_kwargs: pytest.fail("network"))
        assert kbw.reap_merged_pr_worktrees(
            conn, max_checks=1, probe_interval_seconds=0,
        ) == []
        assert len(calls) == 1


def test_dispatch_tick_never_runs_pr_maintenance_inline(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("PR maintenance must not block dispatch_once")

    monkeypatch.setattr(kbw, "reap_merged_pr_worktrees", must_not_run)
    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, max_spawn=0)
        kbd.dispatch_once(conn, max_spawn=0, dry_run=True)
