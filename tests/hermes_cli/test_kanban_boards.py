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
        """``HERMES_KANBAN_DB`` pins the file regardless of board= arg."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board="ignored") == forced


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
        # First connect populates _INITIALIZED_PATHS for this DB.
        with kb.connect(board="recycle") as conn:
            kb.create_task(conn, title="t1", assignee="dev")
        db_path = kb.board_dir("recycle") / "kanban.db"
        assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

        kb.remove_board("recycle", archive=archive)
        # remove_board must drop the cache entry so a re-create through
        # connect() gets a fresh schema-init pass.
        assert str(db_path.resolve()) not in kb._INITIALIZED_PATHS

        # Simulate the event-stream poll: re-open the same slug. connect()
        # recreates the directory + empty .db; the schema must be re-applied.
        with kb.connect(board="recycle") as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "task_events" in tables
        assert "tasks" in tables

    def test_rename_updates_metadata(self, fresh_home):
        kb.create_board("slug-immutable")
        kb.write_board_metadata("slug-immutable", name="New Display Name")
        assert kb.read_board_metadata("slug-immutable")["name"] == "New Display Name"
        # Slug must not change.
        assert kb.board_exists("slug-immutable")


# ---------------------------------------------------------------------------
# Connection isolation
# ---------------------------------------------------------------------------

class TestConnectionIsolation:
    def test_tasks_do_not_leak_across_boards(self, fresh_home):
        kb.create_board("alpha")
        kb.create_board("beta")

        with kb.connect(board="alpha") as conn:
            kb.create_task(conn, title="alpha-task-1", assignee="dev")
            kb.create_task(conn, title="alpha-task-2", assignee="dev")

        with kb.connect(board="beta") as conn:
            kb.create_task(conn, title="beta-only", assignee="dev")

        with kb.connect(board="alpha") as conn:
            a = kb.list_tasks(conn)
        with kb.connect(board="beta") as conn:
            b = kb.list_tasks(conn)
        with kb.connect(board="default") as conn:
            d = kb.list_tasks(conn)

        assert {t.title for t in a} == {"alpha-task-1", "alpha-task-2"}
        assert {t.title for t in b} == {"beta-only"}
        assert d == []

    def test_connect_without_args_uses_current(self, fresh_home):
        kb.create_board("curr")
        kb.set_current_board("curr")
        with kb.connect() as conn:
            kb.create_task(conn, title="implicit", assignee="x")
        with kb.connect(board="curr") as conn:
            tasks = kb.list_tasks(conn)
        assert [t.title for t in tasks] == ["implicit"]

    def test_connect_env_var_overrides_current(self, fresh_home, monkeypatch):
        kb.create_board("persist")
        kb.create_board("envwin")
        kb.set_current_board("persist")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "envwin")
        with kb.connect() as conn:
            kb.create_task(conn, title="via-env", assignee="x")
        with kb.connect(board="envwin") as conn:
            assert [t.title for t in kb.list_tasks(conn)] == ["via-env"]
        with kb.connect(board="persist") as conn:
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
        dispatcher_home = fresh_home / "profiles" / "dispatcher"
        dispatcher_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(dispatcher_home))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
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

        kb._default_spawn(task, str(fresh_home / "ws"), board="spawntest")

        env = captured["env"]
        assert env["HERMES_DISPATCH_SOURCE_PROFILE"] == "dispatcher"
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
    def test_boards_list_default_only(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        res = _cli(["boards", "list", "--json"], env_extra=env)
        assert res.returncode == 0, res.stderr
        data = json.loads(res.stdout)
        slugs = [b["slug"] for b in data]
        assert slugs == ["default"]
        assert data[0]["is_current"] is True


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



