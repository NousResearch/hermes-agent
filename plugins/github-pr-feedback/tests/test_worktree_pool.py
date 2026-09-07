from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from github_pr_feedback.controller import (
    LocalGitRepository,
    PooledLocalGitRepository,
    WorktreePoolExhausted,
    _prepare_receipt_worktree_with_overflow,
)
from github_pr_feedback.ledger import FeedbackLedger
from github_pr_feedback.policy import FeedbackReceipt


def initialized_repository(tmp_path: Path) -> Path:
    path = tmp_path / "repository"
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "--allow-empty", "-m", "root"],
        check=True,
    )
    return path


def commit(path: Path, filename: str, content: str) -> str:
    (path / filename).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", filename], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", filename],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def receipt(
    sha: str, *, pr_number: int = 17, repository: str = "acme/widgets"
) -> FeedbackReceipt:
    return FeedbackReceipt(repository, pr_number, "pr_local_ci", "local-ci-audit-v2", sha)


class MutableClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def make_governed_venv(repo: Path) -> None:
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("#!/bin/sh\n", encoding="utf-8")


def test_pool_reuses_the_same_slot_directory_after_release(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    prepared_a = pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    assert prepared_a.expected_sha == sha_a
    assert (prepared_a.path / "a.txt").is_file()

    from github_pr_feedback.ledger import WorktreeSlotLease

    lease_row = ledger._connection.execute(
        "SELECT slot_id, lease_version, owner_pid FROM worktree_pool_slots"
    ).fetchone()
    pool.release(WorktreeSlotLease(*lease_row))

    prepared_b = pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    assert prepared_b.expected_sha == sha_b
    assert prepared_b.path == prepared_a.path  # same physical slot-0 directory reused
    assert (prepared_b.path / "a.txt").is_file()  # still tracked at the new head
    assert (prepared_b.path / "b.txt").is_file()
    ledger.close()


def test_pool_uses_distinct_slot_directories_for_distinct_repositories(
    tmp_path: Path,
) -> None:
    repo_a = initialized_repository(tmp_path / "repo-a")
    sha_a = commit(repo_a, "a.txt", "a")
    repo_b = initialized_repository(tmp_path / "repo-b")
    sha_b = commit(repo_b, "other.txt", "other")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    first = pool.prepare_receipt_worktree(repo_a, receipt(sha_a, pr_number=1))
    lease_row = ledger._connection.execute(
        "SELECT slot_id, lease_version, owner_pid FROM worktree_pool_slots"
    ).fetchone()
    from github_pr_feedback.ledger import WorktreeSlotLease

    pool.release(WorktreeSlotLease(*lease_row))
    second = pool.prepare_receipt_worktree(
        repo_b, receipt(sha_b, pr_number=2, repository="acme/other")
    )

    assert second.path != first.path
    assert first.path.is_dir()
    assert (first.path / "a.txt").is_file()
    assert (second.path / "other.txt").is_file()
    ledger.close()


def test_pool_preserves_legacy_global_slot_directory(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path / "repo")
    sha = commit(repo, "a.txt", "a")
    pool_root = tmp_path / "pool"
    legacy_slot = pool_root / "slot-0"
    legacy_slot.mkdir(parents=True)
    legacy_marker = legacy_slot / "owned-wip.txt"
    legacy_marker.write_text("preserve me", encoding="utf-8")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, pool_root, slot_count=1, owner_pid=lambda: 4242
    )

    prepared = pool.prepare_receipt_worktree(repo, receipt(sha))

    assert prepared.path != legacy_slot
    assert legacy_marker.read_text(encoding="utf-8") == "preserve me"
    ledger.close()


def test_pool_refuses_when_every_slot_is_leased_and_not_stale(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    with pytest.raises(WorktreePoolExhausted):
        pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    ledger.close()


def test_pool_reclaims_a_stale_lease_without_an_explicit_release(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    pool = PooledLocalGitRepository(
        ledger,
        tmp_path / "pool",
        slot_count=1,
        owner_pid=lambda: 4242,
        clock=clock,
        lease_timeout=timedelta(hours=1),
    )

    pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    # Still within the lease: a second acquire attempt must refuse.
    clock.now += timedelta(minutes=30)
    with pytest.raises(WorktreePoolExhausted):
        pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))

    # Past the lease timeout: reclaimable even though nothing ever called release().
    clock.now += timedelta(hours=1)
    prepared = pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    assert prepared.expected_sha == sha_b
    ledger.close()


def test_pool_release_makes_a_slot_immediately_reusable(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    prepared_a = pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    lease = ledger._connection.execute(
        "SELECT slot_id, lease_version, owner_pid FROM worktree_pool_slots"
    ).fetchone()
    from github_pr_feedback.ledger import WorktreeSlotLease

    pool.release(WorktreeSlotLease(lease[0], lease[1], lease[2]))

    prepared_b = pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    assert prepared_b.path == prepared_a.path
    assert prepared_b.expected_sha == sha_b
    ledger.close()


def test_pool_never_removes_the_linked_venv_between_reuses(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    make_governed_venv(repo)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    prepared_a = pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    venv_link = prepared_a.path / ".venv"
    assert venv_link.is_symlink()
    assert (venv_link / "bin" / "python").is_file()
    # Simulate build/test byproducts a real CI run would leave behind.
    (prepared_a.path / "__pycache__").mkdir()
    (prepared_a.path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    (prepared_a.path / "stray_untracked.txt").write_text("x", encoding="utf-8")

    lease = ledger._connection.execute(
        "SELECT slot_id, lease_version, owner_pid FROM worktree_pool_slots"
    ).fetchone()
    from github_pr_feedback.ledger import WorktreeSlotLease

    pool.release(WorktreeSlotLease(lease[0], lease[1], lease[2]))
    prepared_b = pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))

    assert venv_link.is_symlink()
    assert (venv_link / "bin" / "python").is_file()
    assert not (prepared_b.path / "__pycache__").exists()
    assert not (prepared_b.path / "stray_untracked.txt").exists()
    ledger.close()


def test_link_governed_venv_accepts_profile_managed_environment(tmp_path: Path) -> None:
    profile_root = tmp_path / ".hermes"
    repo = initialized_repository(profile_root)
    managed_venv = profile_root / "venvs" / "hermes-3136"
    (managed_venv / "bin").mkdir(parents=True)
    (managed_venv / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / ".venv").symlink_to(managed_venv, target_is_directory=True)
    workspace = tmp_path / "receipt-worktree"
    workspace.mkdir()

    LocalGitRepository._link_governed_venv(repo, workspace)

    assert (workspace / ".venv").is_symlink()
    assert (workspace / ".venv").resolve(strict=True) == managed_venv.resolve(strict=True)


class FakeKanban:
    def __init__(self, statuses: dict[tuple[str, str], str | None]) -> None:
        self._statuses = statuses
        self.calls: list[tuple[str, str]] = []

    def task_status(self, board: str, task_id: str) -> str | None:
        self.calls.append((board, task_id))
        return self._statuses.get((board, task_id))


def test_reconcile_releases_a_slot_whose_task_is_done(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    prepared_a = pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    pool.bind_task(receipt(sha_a, pr_number=1), "task-1", "repairs")

    kanban = FakeKanban({("repairs", "task-1"): "done"})
    released = pool.reconcile_leases(kanban)
    assert released == 1

    prepared_b = pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    assert prepared_b.path == prepared_a.path  # reclaimed without waiting for the lease timeout
    ledger.close()


@pytest.mark.parametrize("status", ["blocked", "triage"])
def test_reconcile_keeps_a_slot_for_a_retryable_task(
    tmp_path: Path, status: str
) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    prepared_a = pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    pool.bind_task(receipt(sha_a, pr_number=1), "task-1", "repairs")

    released = pool.reconcile_leases(
        FakeKanban({("repairs", "task-1"): status})
    )
    assert released == 0

    with pytest.raises(WorktreePoolExhausted):
        pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    assert prepared_a.path.is_dir()
    ledger.close()


def test_reconcile_keeps_bound_retryable_task_past_stale_timeout(
    tmp_path: Path,
) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    now = [datetime(2026, 8, 31, tzinfo=UTC)]
    pool = PooledLocalGitRepository(
        ledger,
        tmp_path / "pool",
        slot_count=1,
        owner_pid=lambda: 4242,
        clock=lambda: now[0],
        lease_timeout=timedelta(hours=10),
    )

    pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    pool.bind_task(receipt(sha_a, pr_number=1), "task-1", "repairs")
    now[0] += timedelta(hours=11)

    assert pool.reconcile_leases(FakeKanban({("repairs", "task-1"): "blocked"})) == 0
    with pytest.raises(WorktreePoolExhausted):
        pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    ledger.close()


def test_reconcile_leaves_a_slot_whose_task_is_still_running(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    pool.bind_task(receipt(sha_a, pr_number=1), "task-1", "repairs")

    kanban = FakeKanban({("repairs", "task-1"): "running"})
    released = pool.reconcile_leases(kanban)
    assert released == 0

    with pytest.raises(WorktreePoolExhausted):
        pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    ledger.close()


def test_reconcile_releases_a_slot_whose_task_has_vanished(tmp_path: Path) -> None:
    """A board that no longer knows the task is not positive evidence anything

    is still using the slot -- same "missing is not active" interpretation
    the codebase already uses for base-refresh task bindings.
    """

    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    prepared_a = pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    pool.bind_task(receipt(sha_a, pr_number=1), "task-1", "repairs")

    kanban = FakeKanban({})  # task_status returns None: unknown to the board
    released = pool.reconcile_leases(kanban)
    assert released == 1

    prepared_b = pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    assert prepared_b.path == prepared_a.path
    ledger.close()


def test_reconcile_ignores_a_slot_with_no_bound_task_yet(tmp_path: Path) -> None:
    """A slot mid-dispatch (acquired but bind_task not yet called) has nothing

    for reconcile to check and must be left alone -- it stays protected by
    its lease timeout, never silently reclaimed.
    """

    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    sha_b = commit(repo, "b.txt", "b")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=1))
    # No bind_task call.

    released = pool.reconcile_leases(FakeKanban({}))
    assert released == 0
    with pytest.raises(WorktreePoolExhausted):
        pool.prepare_receipt_worktree(repo, receipt(sha_b, pr_number=2))
    ledger.close()


def test_pool_prepare_failure_releases_the_slot_immediately(tmp_path: Path) -> None:
    repo = initialized_repository(tmp_path)
    sha_a = commit(repo, "a.txt", "a")
    ledger = FeedbackLedger(tmp_path / "ledger.sqlite3")
    pool = PooledLocalGitRepository(
        ledger, tmp_path / "pool", slot_count=1, owner_pid=lambda: 4242
    )

    bogus = receipt("f" * 40, pr_number=1)  # well-formed SHA that doesn't exist anywhere
    with pytest.raises(Exception):
        pool.prepare_receipt_worktree(repo, bogus)

    # The failed acquire must not strand the only slot -- immediately usable.
    prepared = pool.prepare_receipt_worktree(repo, receipt(sha_a, pr_number=2))
    assert prepared.expected_sha == sha_a
    ledger.close()


def test_receipt_preparation_uses_exact_head_overflow_when_pool_is_exhausted(
    tmp_path: Path,
) -> None:
    repo = initialized_repository(tmp_path)
    sha = commit(repo, "a.txt", "a")

    class ExhaustedPool:
        def prepare_receipt_worktree(self, _path: Path, _receipt: FeedbackReceipt):
            raise WorktreePoolExhausted("pool full")

    prepared = _prepare_receipt_worktree_with_overflow(
        ExhaustedPool(), repo, receipt(sha), tmp_path / "overflow-worktrees"
    )

    assert prepared.expected_sha == sha
    assert prepared.path.is_relative_to(tmp_path / "overflow-worktrees")
