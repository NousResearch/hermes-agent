"""Behaviour contracts for the treehouse worktree pool (kanban workspace provider).

Unit tests drive :mod:`hermes_cli.kanban_treehouse` through a fake CLI result so
the parsing/cap/fencing contract is pinned without the binary; the integration
test runs the real ``treehouse`` CLI against a throwaway repo, which is the only
way to prove the lease survives ``prune`` and comes back on ``return``.
"""

from __future__ import annotations

import json

import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_treehouse as kbth
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True,
        encoding="utf-8",
    ).stdout


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _write_config(repo: Path, *, max_trees: int = 2, base_branch: str = "main") -> Path:
    path = repo / kbth.CONFIG_FILENAME
    path.write_text(
        f'max_trees = {max_trees}\nroot = "."\nbase_branch = "{base_branch}"\n'
        "unique_leaf = true\n",
        encoding="utf-8",
    )
    return path


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["treehouse"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _tasks_conn(rows: list[tuple]) -> sqlite3.Connection:
    """A DB carrying only what the lease sweep reads.

    Rows are ``(id, status, claim_expires, worker_pid, workspace_path,
    last_run_ended_at)``; ``task_runs`` holds the closing run's timestamp, which
    is what the terminal grace is measured from.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, claim_expires INTEGER,"
        " worker_pid INTEGER, workspace_path TEXT)"
    )
    conn.execute("CREATE TABLE task_runs (task_id TEXT, ended_at INTEGER)")
    for task_id, status, claim_expires, worker_pid, workspace_path, ended_at in rows:
        conn.execute(
            "INSERT INTO tasks (id, status, claim_expires, worker_pid, workspace_path)"
            " VALUES (?, ?, ?, ?, ?)",
            (task_id, status, claim_expires, worker_pid, workspace_path),
        )
        if ended_at is not None:
            conn.execute(
                "INSERT INTO task_runs (task_id, ended_at) VALUES (?, ?)",
                (task_id, ended_at),
            )
    conn.commit()
    return conn


class TestReadPoolConfig:
    def test_absent_config_means_not_pool_enabled(self, tmp_path):
        repo = _make_repo(tmp_path)
        assert kbth.read_pool_config(repo) is None

    def test_reads_the_cap_and_base_branch(self, tmp_path):
        repo = _make_repo(tmp_path)
        _write_config(repo, max_trees=4, base_branch="develop")
        config = kbth.read_pool_config(repo)
        assert config is not None
        assert config.max_trees == 4
        assert config.base_branch == "develop"
        assert config.pool_root == repo / kbth.POOL_DIRNAME

    @pytest.mark.parametrize("text", ["max_trees = 0\n", "max_trees = -1\n", 'max_trees = "8"\n'])
    def test_rejects_a_cap_that_cannot_be_enforced(self, tmp_path, text):
        repo = _make_repo(tmp_path)
        (repo / kbth.CONFIG_FILENAME).write_text(text, encoding="utf-8")
        with pytest.raises(kbth.TreehouseError, match="max_trees"):
            kbth.read_pool_config(repo)

    def test_malformed_toml_fails_loudly(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / kbth.CONFIG_FILENAME).write_text("max_trees = [\n", encoding="utf-8")
        with pytest.raises(kbth.TreehouseError, match="cannot read"):
            kbth.read_pool_config(repo)


def test_pool_repo_root_is_the_repo_above_the_pool_dir(tmp_path):
    repo = _make_repo(tmp_path)
    slot = repo / ".treehouse" / "repo-abc" / "1" / "repo-1"
    assert kbth.pool_repo_root(slot) == repo
    assert kbth.pool_repo_root(repo / ".worktrees" / "t_x") is None
    assert kbth.pool_repo_root(None) is None


def test_acquire_reports_cap_exhaustion_with_the_treehouse_reason(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=2)
    config = kbth.read_pool_config(repo)
    assert config is not None
    monkeypatch.setattr(
        kbth, "_treehouse",
        lambda *a, **k: _completed(
            stderr="all 2 worktrees are in use or dirty (max_trees = 2).", returncode=1
        ),
    )
    with pytest.raises(kbth.PoolExhausted, match="max_trees = 2"):
        kbth.acquire_lease(config, "t_abc123", "wt/t_abc123")


def test_acquire_refuses_a_slot_outside_the_in_repo_pool(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _write_config(repo)
    config = kbth.read_pool_config(repo)
    assert config is not None
    outside = tmp_path / "elsewhere" / "repo-1"
    monkeypatch.setattr(
        kbth, "_treehouse",
        lambda *a, **k: _completed(stdout=json.dumps({"path": str(outside), "lease_id": "x"})),
    )
    with pytest.raises(kbth.TreehouseError, match="outside the in-repo pool"):
        kbth.acquire_lease(config, "t_abc123", "wt/t_abc123")


def test_release_passes_the_holder_fence_and_force(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _write_config(repo)
    slot = repo / ".treehouse" / "repo-abc" / "1" / "repo-1"
    slot.mkdir(parents=True)
    calls: list[tuple] = []

    def fake_treehouse(config, *args, **kwargs):
        calls.append(args)
        return _completed()

    monkeypatch.setattr(kbth, "_treehouse", fake_treehouse)
    assert kbth.release_lease(slot, "t_abc123") is True
    args = calls[0]
    assert args[:2] == ("return", "--force")
    assert "--if-lease-holder" in args
    assert args[args.index("--if-lease-holder") + 1] == "kanban:t_abc123"
    assert args[-1] == str(slot)


def test_release_never_returns_another_cards_lease(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _write_config(repo)
    slot = repo / ".treehouse" / "repo-abc" / "1" / "repo-1"
    slot.mkdir(parents=True)
    monkeypatch.setattr(
        kbth, "_treehouse",
        lambda *a, **k: _completed(
            stderr="lease precondition failed: lease holder does not match worktree", returncode=1
        ),
    )
    assert kbth.release_lease(slot, "t_abc123") is False


def test_sweep_returns_leases_of_cards_that_no_longer_run_a_worker(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    _write_config(repo)
    slot_a = repo / ".treehouse" / "repo-abc" / "1" / "repo-1"
    slot_b = repo / ".treehouse" / "repo-abc" / "2" / "repo-2"
    slot_c = repo / ".treehouse" / "repo-abc" / "3" / "repo-3"
    conn = _tasks_conn([
        ("t_done", "done", None, None, str(slot_a), 0),
        ("t_running", "running", 9_999_999_999, None, str(slot_b), None),
        ("t_blocked", "blocked", None, None, str(slot_c), 0),
    ])
    monkeypatch.setattr(kbth, "read_pool_config", lambda root: kbth.PoolConfig(
        repo_root=repo, config_path=repo / kbth.CONFIG_FILENAME, max_trees=3, base_branch="main",
    ))
    monkeypatch.setattr(kbth, "_slot_status", lambda config: [
        {"name": "1", "path": str(slot_a), "status": "leased", "lease_holder": "kanban:t_done"},
        {"name": "2", "path": str(slot_b), "status": "leased", "lease_holder": "kanban:t_running"},
        {"name": "3", "path": str(slot_c), "status": "leased", "lease_holder": "kanban:t_blocked"},
        # Held by another board's database: this DB must not reclaim it.
        {"name": "4", "path": str(repo / ".treehouse" / "repo-abc" / "4" / "repo-4"),
         "status": "leased", "lease_holder": "kanban:t_unknown"},
    ])
    released: list[str] = []
    monkeypatch.setattr(
        kbth, "release_lease", lambda path, task_id: released.append(task_id) or True,
    )
    assert sorted(kbth.release_orphaned_leases(conn)) == ["t_blocked", "t_done"]
    assert sorted(released) == ["t_blocked", "t_done"]


def test_sweep_treats_a_dead_workers_running_card_as_finished(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    slot = repo / ".treehouse" / "repo-abc" / "1" / "repo-1"
    conn = _tasks_conn([("t_killed", "running", None, 999_999, str(slot), None)])
    monkeypatch.setattr(kbth, "read_pool_config", lambda root: kbth.PoolConfig(
        repo_root=repo, config_path=repo / kbth.CONFIG_FILENAME, max_trees=1, base_branch="main",
    ))
    monkeypatch.setattr(kbth, "_slot_status", lambda config: [
        {"name": "1", "path": str(slot), "status": "leased", "lease_holder": "kanban:t_killed"},
    ])
    monkeypatch.setattr(kbth, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(kbth, "release_lease", lambda path, task_id: True)
    assert kbth.release_orphaned_leases(conn) == ["t_killed"]


def test_sweep_returns_a_duplicate_lease_of_a_live_card(tmp_path, monkeypatch):
    """A card holds one lease. If a crashed dispatcher left a second one, the
    slot the card is NOT recorded in is returned even while the card runs."""
    repo = _make_repo(tmp_path)
    recorded = repo / ".treehouse" / "repo-abc" / "1" / "repo-1"
    duplicate = repo / ".treehouse" / "repo-abc" / "2" / "repo-2"
    conn = _tasks_conn([("t_dup", "running", None, 999_999, str(recorded), None)])
    monkeypatch.setattr(kbth, "read_pool_config", lambda root: kbth.PoolConfig(
        repo_root=repo, config_path=repo / kbth.CONFIG_FILENAME, max_trees=2, base_branch="main",
    ))
    monkeypatch.setattr(kbth, "_slot_status", lambda config: [
        {"name": "1", "path": str(recorded), "status": "leased", "lease_holder": "kanban:t_dup"},
        {"name": "2", "path": str(duplicate), "status": "leased", "lease_holder": "kanban:t_dup"},
    ])
    monkeypatch.setattr(kbth, "_pid_alive", lambda pid: True)
    released: list[str] = []
    monkeypatch.setattr(
        kbth, "release_lease", lambda path, task_id: released.append(path) or True,
    )
    assert kbth.release_orphaned_leases(conn) == ["t_dup"]
    assert released == [str(duplicate)]


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_real_pool_caps_recycles_and_survives_prune_while_leased(tmp_path):
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=1)
    config = kbth.read_pool_config(repo)
    assert config is not None
    binary = kbth.treehouse_binary()
    assert binary is not None

    lease = kbth.acquire_lease(config, "t_abc123", "wt/t_abc123")
    assert repo / kbth.POOL_DIRNAME in lease.path.parents
    assert _git(lease.path, "branch", "--show-current").strip() == "wt/t_abc123"

    slots = kbth._slot_status(config)
    assert [(s["name"], s["status"], s["lease_holder"]) for s in slots] == [
        ("1", "leased", "kanban:t_abc123")
    ]

    # A leased slot is never a prune candidate, even with nothing running in it.
    prune = subprocess.run(
        [binary, "prune", "--yes"], check=False,
        cwd=str(repo), capture_output=True, text=True, encoding="utf-8",
    )
    assert prune.returncode == 0, prune.stderr
    assert lease.path.is_dir(), "prune removed a leased slot"

    # The cap holds: a second card waits instead of spawning outside the pool.
    with pytest.raises(kbth.PoolExhausted):
        kbth.acquire_lease(config, "t_def456", "wt/t_def456")

    assert kbth.release_lease(lease.path, "t_abc123") is True
    assert kbth._slot_status(config)[0]["status"] == "available"

    # Returned -> the slot becomes reclaimable.
    prune = subprocess.run(
        [binary, "prune", "--yes"], check=False,
        cwd=str(repo), capture_output=True, text=True, encoding="utf-8",
    )
    assert prune.returncode == 0, prune.stderr
    assert not lease.path.exists()
    assert kbth._slot_status(config) == []


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_resolver_acquires_a_pool_lease_for_a_pool_enabled_repo(kanban_home, tmp_path):
    """A repo-root ``workspace_path`` resolves to a leased pool slot, not
    ``<repo>/.worktrees/<id>`` -- and the card id lands in the branch name."""
    repo = _make_repo(tmp_path)
    _write_config(repo)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="pool card", workspace_kind="worktree",
            workspace_path=str(repo),
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert repo / kbth.POOL_DIRNAME in workspace.parents
    assert branch == f"wt/{tid}"
    assert (workspace / "a.txt").exists()
    assert _git(workspace, "branch", "--show-current").strip() == branch
    assert not (repo / ".worktrees" / tid).exists()


def test_resolver_raises_for_a_repo_without_config(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="legacy card", workspace_kind="worktree",
            workspace_path=str(repo),
        )
        task = kb.get_task(conn, tid)
        assert task is not None

    with pytest.raises(kbth.TreehouseRequired) as exc_info:
        kbw._resolve_worktree_workspace(task)

    assert f"has no {kbth.CONFIG_FILENAME}" in str(exc_info.value)
    assert "exclusively through treehouse" in str(exc_info.value)


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_cap_reason_is_quoted_only_when_the_pool_is_actually_full(tmp_path):
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=1)
    config = kbth.read_pool_config(repo)
    assert config is not None
    assert kbth.cap_reason(config) is None  # empty pool: treehouse can create a slot

    kbth.acquire_lease(config, "t_filler", "wt/t_filler")
    reason = kbth.cap_reason(config)
    assert reason is not None
    assert "max_trees = 1" in reason
    assert "1 leased" in str(reason)


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_resolver_reuses_the_cards_own_lease(kanban_home, tmp_path):
    """A retry must run in the slot it already holds, not add a second slot."""
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=2)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="retry card", workspace_kind="worktree", workspace_path=str(repo),
        )
        task = kb.get_task(conn, tid)
        assert task is not None
    first, branch = kbw._resolve_worktree_workspace(task)
    with kbc.connect() as conn:
        kbw.set_workspace_path(conn, tid, str(first))
        kbw.set_branch_name(conn, tid, branch)
        task = kb.get_task(conn, tid)
        assert task is not None

    second, _branch = kbw._resolve_worktree_workspace(task)

    config = kbth.read_pool_config(repo)
    assert config is not None
    slots = kbth._slot_status(config)
    assert second == first
    assert len(slots) == 1
    assert slots[0]["lease_holder"] == f"kanban:{tid}"


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_resolver_replaces_a_pruned_slot_with_a_fresh_lease(kanban_home, tmp_path, monkeypatch):
    """A recorded slot returned and pruned must not be recreated by git.

    The card's durable artefact is the branch, so the lease is taken through
    the pool again (treehouse may re-create the same slot path) and the branch
    is checked out there. ``git worktree add`` must not run at all.
    """
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=2)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="pruned card", workspace_kind="worktree", workspace_path=str(repo),
        )
        task = kb.get_task(conn, tid)
        assert task is not None
    pruned, branch = kbw._resolve_worktree_workspace(task)
    assert kbth.release_lease(pruned, tid) is True
    prune = subprocess.run(
        ["treehouse", "prune", "--yes"], check=False,
        cwd=str(repo), capture_output=True, text=True,
    )
    assert prune.returncode == 0, prune.stderr
    assert not pruned.exists()

    with kbc.connect() as conn:
        kbw.set_workspace_path(conn, tid, str(pruned))
        kbw.set_branch_name(conn, tid, branch)
        task = kb.get_task(conn, tid)
        assert task is not None
    git_worktree_calls: list[tuple] = []
    monkeypatch.setattr(kbw, "_ensure_git_worktree", lambda *args: git_worktree_calls.append(args))
    fresh, fresh_branch = kbw._resolve_worktree_workspace(task)

    assert git_worktree_calls == [], "the resolver bypassed the pool with git worktree add"
    assert repo / kbth.POOL_DIRNAME in fresh.parents
    assert fresh.exists()
    assert fresh_branch == branch
    assert _git(fresh, "branch", "--show-current").strip() == branch
    config = kbth.read_pool_config(repo)
    assert config is not None
    entry = next(e for e in kbth._slot_status(config) if e["path"] == str(fresh))
    assert entry["status"] == "leased"
    assert entry["lease_holder"] == f"kanban:{tid}"


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_resolver_never_adopts_another_cards_lease(kanban_home, tmp_path):
    """A card whose recorded path is now another card's slot gets its own slot.

    Re-leasing somebody else's slot is refused by treehouse; the resolver must
    not fail the card over it (nor touch that slot or its branch).
    """
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=2)
    config = kbth.read_pool_config(repo)
    assert config is not None
    other = kbth.acquire_lease(config, "t_card_a", "wt/t_card_a")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="aliased card", workspace_kind="worktree", workspace_path=str(other.path),
        )
        task = kb.get_task(conn, tid)
        assert task is not None
    workspace, branch = kbw._resolve_worktree_workspace(task)

    assert workspace != other.path
    assert repo / kbth.POOL_DIRNAME in workspace.parents
    assert _git(workspace, "branch", "--show-current").strip() == branch == f"wt/{tid}"
    assert _git(other.path, "branch", "--show-current").strip() == "wt/t_card_a"
    holders = {entry["lease_holder"] for entry in kbth._slot_status(config)}
    assert holders == {"kanban:t_card_a", f"kanban:{tid}"}


def test_completion_cleanup_never_removes_a_pool_slot(kanban_home, tmp_path):
    """The pool owns slot lifecycle. Git removal on completion would delete
    treehouse's recorded slot and break the lease it still holds."""
    repo = _make_repo(tmp_path)
    _write_config(repo)
    config = kbth.read_pool_config(repo)
    assert config is not None
    lease = kbth.acquire_lease(config, "t_keep", "wt/t_keep")

    kbw._cleanup_worktree_workspace("t_keep", str(lease.path), "wt/t_keep")

    assert lease.path.is_dir()
    entry = next(e for e in kbth._slot_status(config) if e["path"] == str(lease.path))
    assert entry["status"] == "leased"
    assert entry["lease_holder"] == "kanban:t_keep"


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_dispatch_records_the_pool_lease_as_the_card_workspace(
    kanban_home, all_assignees_spawnable, tmp_path
):
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=2)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="pool card", assignee="alice",
            workspace_kind="worktree", workspace_path=str(repo),
        )
        result = kbd.dispatch_once(conn, spawn_fn=lambda *_a, **_k: None)
        task = kb.get_task(conn, tid)

    assert [spawned[0] for spawned in result.spawned] == [tid]
    workspace = Path(result.spawned[0][2])
    assert repo / kbth.POOL_DIRNAME in workspace.parents
    assert task is not None and task.workspace_path == str(workspace)
    config = kbth.read_pool_config(repo)
    assert config is not None
    assert kbth._slot_status(config)[0]["lease_holder"] == f"kanban:{tid}"


@pytest.mark.skipif(
    kbth.treehouse_binary() is None, reason="treehouse CLI is not installed"
)
def test_a_full_pool_defers_the_card_before_claiming_it(
    kanban_home, all_assignees_spawnable, tmp_path
):
    """No spawn outside the pool, no run row per tick, reason on the result."""
    repo = _make_repo(tmp_path)
    _write_config(repo, max_trees=1)
    config = kbth.read_pool_config(repo)
    assert config is not None
    kbth.acquire_lease(config, "t_filler", "wt/t_filler")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="waits for a slot", assignee="alice",
            workspace_kind="worktree", workspace_path=str(repo),
        )
        result = kbd.dispatch_once(conn, spawn_fn=lambda *_a, **_k: None)
        task = kb.get_task(conn, tid)
        runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,)
        ).fetchone()[0]

    assert [entry[0] for entry in result.skipped_pool_exhausted] == [tid]
    assert "max_trees = 1" in result.skipped_pool_exhausted[0][1]
    assert result.spawned == []
    assert task is not None and task.status == "ready"
    assert runs == 0, "a deferred card must not write a failed run every tick"
    assert not (repo / ".worktrees" / tid).exists()
