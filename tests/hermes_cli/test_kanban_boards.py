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


    def test_env_var_db_override_still_wins_without_explicit_board(
        self, fresh_home, tmp_path, monkeypatch,
    ):
        """``HERMES_KANBAN_DB`` keeps pinning implicit worker operations."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced

    def test_bare_env_var_db_override_keeps_legacy_explicit_pin(
        self, fresh_home, tmp_path, monkeypatch,
    ):
        """A path override without a board slug retains legacy precedence."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path(board="ignored") == forced

    def test_explicit_board_beats_different_worker_db_pin(
        self, fresh_home, tmp_path, monkeypatch,
    ):
        """A worker may explicitly target another board without inheriting its own DB."""
        forced = tmp_path / "worker-board.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")

        # Re-selecting the worker's board preserves the exact dispatcher pin.
        assert kb.kanban_db_path(board="worker-board") == forced
        # A different explicit slug must resolve independently.
        assert kb.kanban_db_path(board="other-board") == (
            fresh_home / "kanban" / "boards" / "other-board" / "kanban.db"
        )
        # The CLI --board path uses the scoped override and then calls connect()
        # without forwarding a board kwarg, so cover that route too.
        with kb.scoped_current_board("other-board"):
            assert kb.kanban_db_path() == (
                fresh_home / "kanban" / "boards" / "other-board" / "kanban.db"
            )

    def test_explicit_board_beats_malformed_worker_board_pin(
        self, fresh_home, tmp_path, monkeypatch,
    ):
        forced = tmp_path / "worker-board.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "../invalid")

        assert kb.kanban_db_path(board="other-board") == (
            fresh_home / "kanban" / "boards" / "other-board" / "kanban.db"
        )

    def test_env_var_workspaces_override(self, fresh_home, tmp_path, monkeypatch):
        forced = tmp_path / "ws"
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(forced))
        assert kb.workspaces_root() == forced
        assert kb.workspaces_root(board="ignored") == forced

    def test_explicit_board_beats_different_worker_workspaces_pin(
        self, fresh_home, tmp_path, monkeypatch,
    ):
        forced = tmp_path / "worker-workspaces"
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(forced))
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")

        assert kb.workspaces_root(board="worker-board") == forced
        assert kb.workspaces_root(board="other-board") == (
            fresh_home / "kanban" / "boards" / "other-board" / "workspaces"
        )


    def test_list_boards_keeps_distinct_paths_under_worker_pin(
        self, fresh_home, monkeypatch,
    ):
        kb.create_board("worker-board")
        kb.create_board("other-board")
        worker_db = fresh_home / "kanban" / "boards" / "worker-board" / "kanban.db"
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "worker-board")
        monkeypatch.setenv("HERMES_KANBAN_DB", str(worker_db))

        paths = {entry["slug"]: entry["db_path"] for entry in kb.list_boards()}

        assert paths["worker-board"] == str(worker_db)
        assert paths["other-board"] == str(
            fresh_home / "kanban" / "boards" / "other-board" / "kanban.db"
        )
        assert len(set(paths.values())) == len(paths)


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

    def test_named_board_cli_routes_past_inherited_worker_pin(self, tmp_path):
        base_env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "alpha"], env_extra=base_env).returncode == 0
        assert _cli(["boards", "create", "beta"], env_extra=base_env).returncode == 0
        assert _cli(
            ["--board", "alpha", "create", "Alpha only", "--assignee", "dev"],
            env_extra=base_env,
        ).returncode == 0
        for title in ("Beta one", "Beta two"):
            assert _cli(
                ["--board", "beta", "create", title, "--assignee", "dev"],
                env_extra=base_env,
            ).returncode == 0

        alpha_root = tmp_path / "kanban" / "boards" / "alpha"
        pinned_env = {
            **base_env,
            "HERMES_KANBAN_BOARD": "alpha",
            "HERMES_KANBAN_DB": str(alpha_root / "kanban.db"),
            "HERMES_KANBAN_WORKSPACES_ROOT": str(alpha_root / "workspaces"),
        }

        boards = json.loads(
            _cli(["boards", "list", "--json"], env_extra=pinned_env).stdout
        )
        by_slug = {entry["slug"]: entry for entry in boards}
        assert by_slug["alpha"]["db_path"] == str(alpha_root / "kanban.db")
        assert by_slug["beta"]["db_path"] == str(
            tmp_path / "kanban" / "boards" / "beta" / "kanban.db"
        )
        assert by_slug["alpha"]["counts"]["ready"] == 1
        assert by_slug["beta"]["counts"]["ready"] == 2

        beta_list = _cli(
            ["--board", "beta", "list", "--json"], env_extra=pinned_env
        )
        assert beta_list.returncode == 0, beta_list.stderr
        assert [task["title"] for task in json.loads(beta_list.stdout)] == [
            "Beta one",
            "Beta two",
        ]

    def test_board_flag_rejects_unknown(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        r = _cli(["--board", "ghost", "list"], env_extra=env)
        # main.py's dispatcher doesn't propagate return codes today, so we
        # assert the user-visible signal: a stderr error message. Whether
        # the exit code stays 0 is a separate (pre-existing) issue.
        assert "does not exist" in r.stderr

