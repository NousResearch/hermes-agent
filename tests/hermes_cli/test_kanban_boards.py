"""Tests for the multi-board kanban layer (``hermes kanban boards …``).

Covers the pieces added when boards became a first-class concept:

* Slug validation and normalisation.
* Path resolution for ``default`` (legacy ``<root>/kanban.db``) vs
  named boards (``<root>/kanban/boards/<slug>/kanban.db``).
* Current-board persistence via ``<root>/kanban/current`` and
  ``HERMES_KANBAN_BOARD`` env var.
* ``connect(board=)`` isolation — writes on one board don't leak.
* ``create_board`` / ``list_boards`` / ``remove_board`` round trip.
* CLI surface: ``hermes kanban boards list/create/switch/rm``.
* ``_default_spawn`` injects ``HERMES_KANBAN_BOARD`` into worker env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state.

    The autouse hermetic conftest already nukes credentials + TZ; this
    fixture layers a per-test HERMES_HOME plus a path-init cache reset
    so each test sees a truly empty board set.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    # Also reset hermes_constants cache so get_default_hermes_root() re-reads.
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    # Kanban module-level init cache must not leak between tests.
    kb._INITIALIZED_PATHS.clear()
    return home


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

class TestSlugValidation:
    @pytest.mark.parametrize("good", [
        "default", "atm10-server", "hermes-agent", "proj_1", "a",
        "very-long-but-still-ok-slug-with-hyphens-and-numbers-1234",
    ])
    def test_accepts_valid(self, good):
        assert kb._normalize_board_slug(good) == good


    def test_empty_returns_none(self):
        assert kb._normalize_board_slug(None) is None
        assert kb._normalize_board_slug("") is None
        assert kb._normalize_board_slug("   ") is None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPathResolution:
    def test_default_board_legacy_path(self, fresh_home):
        """The default board's DB lives at ``<root>/kanban.db`` for back-compat."""
        assert kb.kanban_db_path() == fresh_home / "kanban.db"
        assert kb.kanban_db_path(board="default") == fresh_home / "kanban.db"

    def test_named_board_under_boards_dir(self, fresh_home):
        p = kb.kanban_db_path(board="atm10-server")
        assert p == fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"


    def test_env_var_db_override_still_wins(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file regardless of ``board=`` arg for every
        execution the dispatcher fences (the 5ec6baa multi-boards isolation: workers
        physically cannot see other boards): its dispatched workers (``HERMES_KANBAN_TASK``)
        and delegated children / descendants (``HERMES_DELEGATED_CHILD_CONTEXT``). An
        explicit board that outranked the pin would also escape
        ``kanban_path_is_fenced``, which checks the pinned path / fenced root. Outside
        those fences an explicit board is the caller's own intent and wins (see
        ``test_explicit_board_trumps_env_var_db_override`` below)."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board=None) == forced
        # Dispatched worker identity: the pin still fences explicit board intent.
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fence_probe")
        assert kb.kanban_db_path(board="ignored") == forced
        # Delegated children / spawned descendants are fenced the same way.
        from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        assert kb.kanban_db_path(board="ignored") == forced

    def test_env_var_db_override_wins_when_board_not_passed(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file when no explicit ``board=`` is given
        (back-compat for dispatcher-spawned workers with no board override)."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board=None) == forced

    def test_explicit_board_trumps_env_var_db_override(self, fresh_home, tmp_path, monkeypatch):
        """Documented priority (module docstring, predates this test): explicit
        ``board=`` arg > ``HERMES_KANBAN_BOARD`` > ``HERMES_KANBAN_DB`` > current >
        default — for UNFENCED callers. An explicit ``board=`` must resolve to that
        board's own path even when ``HERMES_KANBAN_DB`` pins a different file: this is
        what makes cross-board ``kanban_create(board=...)`` / ``kanban_show(board=...)``
        work from a user-facing session instead of silently landing on the pinned board
        (t_3f1c63a5). Fenced callers (dispatched workers, delegated children) keep the
        pinned path — see ``test_env_var_db_override_still_wins`` above."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        p = kb.kanban_db_path(board="atm10-server")
        assert p == fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"
        assert p != forced


# ---------------------------------------------------------------------------
# Current-board resolution
# ---------------------------------------------------------------------------

class TestCurrentBoard:



    def test_stale_file_pointer_falls_back_to_default(self, fresh_home):
        current = fresh_home / "kanban" / "current"
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("missing-board\n", encoding="utf-8")

        assert kb.get_current_board() == "default"
        assert not kb.board_exists("missing-board")
        assert [b["slug"] for b in kb.list_boards()] == ["default"]



    def test_kanban_db_path_reads_current(self, fresh_home):
        """kanban_db_path() with no args respects the on-disk pointer."""
        kb.create_board("my-proj")
        kb.set_current_board("my-proj")
        expected = fresh_home / "kanban" / "boards" / "my-proj" / "kanban.db"
        assert kb.kanban_db_path() == expected


# ---------------------------------------------------------------------------
# Board CRUD
# ---------------------------------------------------------------------------

class TestBoardCRUD:






    @pytest.mark.parametrize("archive", [True, False])
    def test_remove_clears_init_cache_for_recreated_db(self, fresh_home, archive):
        # Regression for #23833: poll loops that call connect(board=slug) right
        # after remove_board() recreate an empty kanban.db at the same path
        # (connect() does mkdir(exist_ok=True)). If _INITIALIZED_PATHS still
        # contains the resolved path, the CREATE TABLE pass is skipped and
        # downstream readers hit `no such table: task_events`.
        kb.create_board("recycle")
        # First connect populates _INITIALIZED_PATHS for this DB.  Use
        # connect_closing: `with connect() as conn` does NOT close the fd, and
        # on Windows an open connection locks kanban.db so remove_board's
        # rename below fails with WinError 5/32.
        with kbc.connect_closing(board="recycle") as conn:
            kb.create_task(conn, title="t1", assignee="dev")
        db_path = kb.board_dir("recycle") / "kanban.db"
        assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

        kb.remove_board("recycle", archive=archive)
        # remove_board must drop the cache entry so a re-create through
        # connect() gets a fresh schema-init pass.
        assert str(db_path.resolve()) not in kb._INITIALIZED_PATHS

        # Re-creation is an explicit operator action.  A stale dispatcher or
        # surviving worker must not implicitly recreate a removed board.
        kb.create_board("recycle")
        with kbc.connect_closing(board="recycle") as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "task_events" in tables
        assert "tasks" in tables

    @pytest.mark.parametrize("archive", [True, False])
    def test_remove_board_terminates_workers_before_archive_or_delete(
        self, fresh_home, monkeypatch, archive,
    ):
        slug = "archive-live" if archive else "delete-live"
        kb.create_board(slug)
        original = kb.board_dir(slug)
        with kbc.connect_closing(board=slug) as conn:
            tid = kb.create_task(conn, title="running", assignee="dev")
            host = kb._claimer_id().split(":", 1)[0]
            kb.claim_task(conn, tid, claimer=f"{host}:worker")
            monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: f"test-boot|{pid}")
            kbd._set_worker_pid(conn, tid, 54321)

        live = {54321}
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid in live)
        signalled = []

        def signal_worker(pid, sig):
            signalled.append((pid, sig))
            assert original.is_dir(), "worker termination must precede the filesystem hand-off"
            live.discard(pid)

        result = kb.remove_board(slug, archive=archive, signal_fn=signal_worker)

        assert signalled and signalled[0][0] == 54321
        assert not live
        assert not original.exists()
        assert result["action"] == ("archived" if archive else "deleted")
        if archive:
            assert Path(result["new_path"]).is_dir()
        else:
            assert result["new_path"] == ""

        # A stale worker/dispatcher still carrying the removed board's pinned
        # environment cannot recreate its directory after the hand-off.
        monkeypatch.setenv("HERMES_KANBAN_BOARD", slug)
        monkeypatch.setenv("HERMES_KANBAN_DB", str(original / "kanban.db"))
        with pytest.raises(ValueError):
            with kbc.connect_closing():
                pass
        assert not original.exists()
        monkeypatch.delenv("HERMES_KANBAN_BOARD")
        monkeypatch.delenv("HERMES_KANBAN_DB")

        guarded = f"{slug}-unverified"
        kb.create_board(guarded)
        guarded_dir = kb.board_dir(guarded)
        with kbc.connect_closing(board=guarded) as conn:
            guarded_tid = kb.create_task(conn, title="unknown worker", assignee="dev")
            kb.claim_task(conn, guarded_tid, claimer=f"{host}:worker")
            kbd._set_worker_pid(conn, guarded_tid, 65432)
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET worker_started_at = ? WHERE id = ?",
                    (kbd.UNVERIFIED_WORKER_FINGERPRINT, guarded_tid),
                )
                conn.execute(
                    "UPDATE task_runs SET worker_started_at = ? WHERE task_id = ?",
                    (kbd.UNVERIFIED_WORKER_FINGERPRINT, guarded_tid),
                )
        live.add(65432)
        before = list(signalled)

        with pytest.raises(ValueError):
            kb.remove_board(guarded, archive=archive, signal_fn=signal_worker)
        assert signalled == before
        assert guarded_dir.is_dir()
        with kbc.connect_closing(board=guarded) as conn:
            assert kb.get_task(conn, guarded_tid) is not None



# ---------------------------------------------------------------------------
# Connection isolation
# ---------------------------------------------------------------------------

class TestConnectionIsolation:
    def test_tasks_do_not_leak_across_boards(self, fresh_home):
        kb.create_board("alpha")
        kb.create_board("beta")

        with kbc.connect(board="alpha") as conn:
            kb.create_task(conn, title="alpha-task-1", assignee="dev")
            kb.create_task(conn, title="alpha-task-2", assignee="dev")

        with kbc.connect(board="beta") as conn:
            kb.create_task(conn, title="beta-only", assignee="dev")

        with kbc.connect(board="alpha") as conn:
            a = kb.list_tasks(conn)
        with kbc.connect(board="beta") as conn:
            b = kb.list_tasks(conn)
        with kbc.connect(board="default") as conn:
            d = kb.list_tasks(conn)

        assert {t.title for t in a} == {"alpha-task-1", "alpha-task-2"}
        assert {t.title for t in b} == {"beta-only"}
        assert d == []

    def test_connect_without_args_uses_current(self, fresh_home):
        kb.create_board("curr")
        kb.set_current_board("curr")
        with kbc.connect() as conn:
            kb.create_task(conn, title="implicit", assignee="x")
        with kbc.connect(board="curr") as conn:
            tasks = kb.list_tasks(conn)
        assert [t.title for t in tasks] == ["implicit"]

    def test_connect_env_var_overrides_current(self, fresh_home, monkeypatch):
        kb.create_board("persist")
        kb.create_board("envwin")
        kb.set_current_board("persist")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "envwin")
        with kbc.connect() as conn:
            kb.create_task(conn, title="via-env", assignee="x")
        with kbc.connect(board="envwin") as conn:
            assert [t.title for t in kb.list_tasks(conn)] == ["via-env"]
        with kbc.connect(board="persist") as conn:
            assert kb.list_tasks(conn) == []


# ---------------------------------------------------------------------------
# Worker spawn env injection
# ---------------------------------------------------------------------------

class TestWorkerSpawnEnv:
    """Ensure the dispatcher pins ``HERMES_KANBAN_BOARD`` / DB / workspaces on spawn.

    We monkey-patch ``subprocess.Popen`` to capture the child env without
    actually spawning anything.
    """

    def test_default_spawn_sets_env_vars(self, fresh_home, monkeypatch):
        captured = {}

        class FakeProc:
            pid = 12345

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        kb.create_board("spawntest")

        task = kb.Task(
            id="t_abc",
            title="worker test",
            body=None,
            assignee="teknium",
            status="ready",
            priority=0,
            created_by="user",
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
        )

        kbd._default_spawn(task, str(fresh_home / "ws"), board="spawntest")

        env = captured["env"]
        assert env["HERMES_KANBAN_BOARD"] == "spawntest"
        assert env["HERMES_KANBAN_TASK"] == "t_abc"
        # DB path should match the per-board DB, not the legacy default.
        expected_db = fresh_home / "kanban" / "boards" / "spawntest" / "kanban.db"
        assert env["HERMES_KANBAN_DB"] == str(expected_db)
        expected_ws = fresh_home / "kanban" / "boards" / "spawntest" / "workspaces"
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(expected_ws)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def _cli(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``hermes kanban …`` with PYTHONPATH pinned to the worktree."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_WORKTREE)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban"] + args,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
        timeout=30,
    )


class TestCLI:


    def test_per_board_task_isolation_via_cli(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "projA"], env_extra=env).returncode == 0
        assert _cli(["boards", "create", "projB"], env_extra=env).returncode == 0

        # Create one task on each via --board.
        r = _cli(["--board", "projA", "create", "Task A", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr
        r = _cli(["--board", "projB", "create", "Task B", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr

        # list on each board only shows its own.
        listA = _cli(["--board", "projA", "list", "--json"], env_extra=env)
        listB = _cli(["--board", "projB", "list", "--json"], env_extra=env)
        listD = _cli(["list", "--json"], env_extra=env)

        titlesA = [t["title"] for t in json.loads(listA.stdout)]
        titlesB = [t["title"] for t in json.loads(listB.stdout)]
        titlesD = [t["title"] for t in json.loads(listD.stdout)]

        assert titlesA == ["Task A"]
        assert titlesB == ["Task B"]
        assert titlesD == []



