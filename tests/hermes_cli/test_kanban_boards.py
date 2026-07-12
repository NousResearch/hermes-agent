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

import argparse
import errno
import json
import os
import shutil
import sqlite3
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

    @pytest.mark.parametrize("bad", [
        "-leading-hyphen", "_leading_underscore",
        "with/slash", "with space",
        "has.dot", "has?question",
        "..", "../etc", "foo\x00bar",
    ])
    def test_rejects_invalid(self, bad):
        with pytest.raises(ValueError):
            kb._normalize_board_slug(bad)

    def test_empty_returns_none(self):
        assert kb._normalize_board_slug(None) is None
        assert kb._normalize_board_slug("") is None
        assert kb._normalize_board_slug("   ") is None

    def test_auto_lowercases(self):
        # Uppercase is auto-downcased (friendlier than rejecting). ``Default``
        # → ``default``, ``ATM10`` → ``atm10``. The on-disk slug is always
        # lowercase regardless of what the user typed.
        assert kb._normalize_board_slug("Default") == "default"
        assert kb._normalize_board_slug("ATM10-Server") == "atm10-server"


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

    def test_workspaces_per_board(self, fresh_home):
        assert kb.workspaces_root() == fresh_home / "kanban" / "workspaces"
        # Uppercase input gets auto-downcased to the on-disk slug.
        assert kb.workspaces_root(board="projA") == (
            fresh_home / "kanban" / "boards" / "proja" / "workspaces"
        )

    def test_logs_per_board(self, fresh_home):
        assert kb.worker_logs_dir() == fresh_home / "kanban" / "logs"
        assert kb.worker_logs_dir(board="other") == (
            fresh_home / "kanban" / "boards" / "other" / "logs"
        )

    def test_env_var_db_override_still_wins(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file regardless of board= arg."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board="ignored") == forced

    def test_env_var_workspaces_override(self, fresh_home, tmp_path, monkeypatch):
        forced = tmp_path / "ws"
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(forced))
        assert kb.workspaces_root(board="any") == forced


# ---------------------------------------------------------------------------
# Current-board resolution
# ---------------------------------------------------------------------------

class TestCurrentBoard:
    def test_default_when_unset(self, fresh_home):
        assert kb.get_current_board() == "default"

    def test_env_var_takes_precedence(self, fresh_home, monkeypatch):
        # Create the board so the env-var value is honoured (get_current_board
        # trusts env-var validity, but the resolution chain doesn't require
        # the board to exist; we just test that env trumps).
        kb.create_board("envboard")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "envboard")
        assert kb.get_current_board() == "envboard"

    def test_file_pointer_honoured(self, fresh_home):
        kb.create_board("filepick")
        kb.set_current_board("filepick")
        assert kb.get_current_board() == "filepick"

    def test_stale_file_pointer_falls_back_to_default(self, fresh_home):
        current = fresh_home / "kanban" / "current"
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("missing-board\n", encoding="utf-8")

        assert kb.get_current_board() == "default"
        assert not kb.board_exists("missing-board")
        assert [b["slug"] for b in kb.list_boards()] == ["default"]

    def test_empty_board_dir_does_not_count_as_existing(self, fresh_home):
        ghost = fresh_home / "kanban" / "boards" / "ghost"
        ghost.mkdir(parents=True)

        assert not kb.board_exists("ghost")
        assert [b["slug"] for b in kb.list_boards()] == ["default"]

    def test_env_beats_file(self, fresh_home, monkeypatch):
        kb.create_board("a")
        kb.create_board("b")
        kb.set_current_board("a")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "b")
        assert kb.get_current_board() == "b"

    def test_stale_env_falls_through_to_file_pointer(self, fresh_home, monkeypatch):
        kb.create_board("persisted")
        kb.set_current_board("persisted")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "missing-board")
        assert kb.get_current_board() == "persisted"

    def test_invalid_env_falls_through(self, fresh_home, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "!!bad!!")
        # Should not crash — falls through to default.
        assert kb.get_current_board() == "default"

    def test_clear_current_board(self, fresh_home):
        kb.create_board("x")
        kb.set_current_board("x")
        kb.clear_current_board()
        assert kb.get_current_board() == "default"

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
    def test_create_and_list(self, fresh_home):
        assert [b["slug"] for b in kb.list_boards()] == ["default"]
        kb.create_board("foo", name="Foo Board", description="test")
        slugs = [b["slug"] for b in kb.list_boards()]
        assert slugs == ["default", "foo"]

    def test_create_is_idempotent(self, fresh_home):
        kb.create_board("bar")
        kb.create_board("bar")  # no error
        slugs = [b["slug"] for b in kb.list_boards()]
        assert slugs == ["default", "bar"]

    def test_create_writes_metadata(self, fresh_home):
        meta = kb.create_board(
            "baz",
            name="Baz",
            description="desc",
            icon="📦",
            color="#abcdef",
        )
        assert meta["slug"] == "baz"
        assert meta["name"] == "Baz"
        assert meta["icon"] == "📦"
        # Round-trip via read_board_metadata.
        again = kb.read_board_metadata("baz")
        assert again["name"] == "Baz"
        assert again["description"] == "desc"
        assert again["icon"] == "📦"

    def test_remove_archive(self, fresh_home):
        kb.create_board("toremove")
        res = kb.remove_board("toremove")
        assert res["action"] == "archived"
        assert Path(res["new_path"]).exists()
        assert "toremove" not in [b["slug"] for b in kb.list_boards()]

    def test_remove_hard_delete(self, fresh_home):
        kb.create_board("nuke")
        d = kb.board_dir("nuke")
        assert d.exists()
        res = kb.remove_board("nuke", archive=False)
        assert res["action"] == "deleted"
        assert not d.exists()

    def test_remove_hard_delete_retries_transient_detached_cleanup_failure(
        self, fresh_home, monkeypatch,
    ):
        kb.create_board("nuke")
        board_path = kb.board_dir("nuke")
        real_rmtree = shutil.rmtree
        attempts = []

        def _fail_once(path, *args, **kwargs):
            attempts.append(Path(path))
            if len(attempts) == 1:
                raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", _fail_once)

        result = kb.remove_board("nuke", archive=False)

        assert result["action"] == "deleted"
        assert len(attempts) == 2
        assert attempts[0] == attempts[1]
        assert attempts[0].name.startswith(".deleting-nuke-")
        assert not board_path.exists()
        assert not attempts[0].exists()

    def test_remove_hard_delete_restores_board_after_persistent_cleanup_failure(
        self, fresh_home, monkeypatch,
    ):
        kb.create_board("nuke")
        board_path = kb.board_dir("nuke")
        with kb.connect(board="nuke") as conn:
            task_id = kb.create_task(
                conn,
                title="preserve me",
                assignee="dev",
            )
        kb.set_current_board("nuke")
        cache_key = str((board_path / "kanban.db").resolve())
        assert cache_key in kb._INITIALIZED_PATHS
        attempts = []

        def _always_fail(path, *args, **kwargs):
            attempts.append(Path(path))
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))

        monkeypatch.setattr(shutil, "rmtree", _always_fail)

        with pytest.raises(OSError, match="Directory not empty"):
            kb.remove_board("nuke", archive=False)

        assert len(attempts) == 3
        assert board_path.exists()
        assert not list(board_path.parent.glob(".deleting-nuke-*"))
        assert "nuke" in [board["slug"] for board in kb.list_boards()]
        assert kb.get_current_board() == "nuke"
        # Cleanup began before rollback, so the DB may have been partially
        # removed. The next connect must revalidate it instead of trusting the
        # stale pre-delete initialization cache entry.
        assert cache_key not in kb._INITIALIZED_PATHS
        with kb.connect(board="nuke", create_if_missing=False) as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "preserve me"
        assert cache_key in kb._INITIALIZED_PATHS

    def test_remove_hard_delete_surfaces_recovery_path_on_restore_collision(
        self, fresh_home, monkeypatch,
    ):
        kb.create_board("nuke")
        board_path = kb.board_dir("nuke")

        def _fail_after_public_path_is_recreated(path, *args, **kwargs):
            board_path.mkdir()
            (board_path / "concurrent-board").write_text("new", encoding="utf-8")
            raise PermissionError(errno.EACCES, "Permission denied", str(path))

        monkeypatch.setattr(
            shutil,
            "rmtree",
            _fail_after_public_path_is_recreated,
        )

        with pytest.raises(
            kb.BoardDeleteRecoveryError,
            match="inspect the detached board recovery path",
        ) as exc_info:
            kb.remove_board("nuke", archive=False)

        recovery_paths = list(board_path.parent.glob(".deleting-nuke-*"))
        assert len(recovery_paths) == 1
        assert exc_info.value.recovery_path == recovery_paths[0]
        assert str(recovery_paths[0]) in str(exc_info.value)
        assert (board_path / "concurrent-board").read_text(encoding="utf-8") == "new"

    def test_remove_hard_delete_rename_failure_leaves_board_state_untouched(
        self, fresh_home, monkeypatch,
    ):
        kb.create_board("nuke")
        kb.set_current_board("nuke")
        board_path = kb.board_dir("nuke")
        db_path = board_path / "kanban.db"
        with kb.connect(board="nuke") as conn:
            task_id = kb.create_task(conn, title="still here", assignee="dev")
        cache_key = str(db_path.resolve())
        assert cache_key in kb._INITIALIZED_PATHS
        real_rename = Path.rename

        def _deny_board_detach(path, target):
            if path == board_path:
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return real_rename(path, target)

        monkeypatch.setattr(Path, "rename", _deny_board_detach)

        with pytest.raises(PermissionError, match="Permission denied"):
            kb.remove_board("nuke", archive=False)

        assert board_path.exists()
        assert kb.get_current_board() == "nuke"
        assert cache_key in kb._INITIALIZED_PATHS
        with kb.connect(board="nuke", create_if_missing=False) as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "still here"

    def test_remove_default_forbidden(self, fresh_home):
        with pytest.raises(ValueError, match="default"):
            kb.remove_board("default")

    def test_remove_nonexistent_raises(self, fresh_home):
        with pytest.raises(ValueError, match="does not exist"):
            kb.remove_board("nosuch")

    def test_remove_clears_current_pointer(self, fresh_home):
        kb.create_board("pinned")
        kb.set_current_board("pinned")
        kb.remove_board("pinned")
        assert kb.get_current_board() == "default"

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
    def test_connect_existing_readonly_bypasses_schema_init_and_cache(
        self, fresh_home,
    ):
        db_path = fresh_home / "readonly path # with percent%" / "kanban.db"
        db_path.parent.mkdir(parents=True)
        raw = sqlite3.connect(db_path)
        try:
            raw.execute("CREATE TABLE marker (value TEXT)")
            raw.commit()
        finally:
            raw.close()

        cache_key = str(db_path.resolve())
        lock_path = db_path.with_name(db_path.name + ".init.lock")
        assert cache_key not in kb._INITIALIZED_PATHS
        assert not lock_path.exists()

        conn = kb.connect_existing_readonly(db_path=db_path)
        try:
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("CREATE TABLE must_not_exist (value TEXT)")
        finally:
            conn.close()

        assert tables == {"marker"}
        assert cache_key not in kb._INITIALIZED_PATHS
        assert not lock_path.exists()

    def test_connect_existing_readonly_does_not_create_missing_parent(
        self, fresh_home,
    ):
        db_path = fresh_home / "missing readonly path" / "kanban.db"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            kb.connect_existing_readonly(db_path=db_path)

        assert not db_path.parent.exists()

    @pytest.mark.parametrize(
        "kanban_root_name",
        ["plain-home", "kanban home # with percent%"],
        ids=["plain-path", "uri-escaped-path"],
    )
    def test_stale_named_consumer_does_not_recreate_removed_board(
        self, fresh_home, monkeypatch, kanban_root_name,
    ):
        kanban_root = fresh_home / kanban_root_name
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_root))
        kb.create_board("ephemeral")
        board_path = kb.board_dir("ephemeral")
        db_path = kb.kanban_db_path(board="ephemeral")
        assert db_path.is_file()

        kb.remove_board("ephemeral", archive=False)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            kb.connect(board="ephemeral", create_if_missing=False)

        assert not board_path.exists()

    def test_connect_no_create_handles_uri_escaped_existing_path(self, fresh_home):
        db_path = fresh_home / "kanban path # with percent%" / "kanban.db"
        with kb.connect(db_path=db_path) as conn:
            task_id = kb.create_task(conn, title="escaped path", assignee="dev")

        with kb.connect(db_path=db_path, create_if_missing=False) as conn:
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.title == "escaped path"

    def test_connect_no_create_does_not_create_uri_escaped_missing_path(
        self, fresh_home,
    ):
        db_path = fresh_home / "missing path # with percent%" / "kanban.db"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            kb.connect(db_path=db_path, create_if_missing=False)

        assert not db_path.parent.exists()

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

    def test_connect_stale_env_uses_fallback_board_without_recreating_it(
        self, fresh_home, monkeypatch,
    ):
        kb.create_board("ephemeral")
        kb.remove_board("ephemeral")
        kb.create_board("persist")
        kb.set_current_board("persist")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "ephemeral")

        with kb.connect() as conn:
            kb.create_task(conn, title="via-fallback", assignee="x")

        with kb.connect(board="persist") as conn:
            assert [t.title for t in kb.list_tasks(conn)] == ["via-fallback"]
        assert not kb.board_exists("ephemeral")


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

    def test_default_board_spawn_keeps_legacy_paths(self, fresh_home, monkeypatch):
        captured = {}

        class FakeProc:
            pid = 1

        def fake_popen(cmd, *args, **kwargs):
            captured["env"] = kwargs.get("env", {})
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        task = kb.Task(
            id="t_def",
            title="",
            body=None,
            assignee="teknium",
            status="ready",
            priority=0,
            created_by=None,
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
        )
        kb._default_spawn(task, str(fresh_home / "ws"), board=None)
        env = captured["env"]
        assert env["HERMES_KANBAN_BOARD"] == "default"
        assert env["HERMES_KANBAN_DB"] == str(fresh_home / "kanban.db")


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
    def test_boards_rm_surfaces_recovery_path(self, tmp_path, monkeypatch, capsys):
        from hermes_cli import kanban as kanban_cli

        recovery_path = tmp_path / ".deleting-nuke-recoverable"
        error = kb.BoardDeleteRecoveryError(
            "nuke",
            recovery_path,
            PermissionError(errno.EACCES, "Permission denied"),
            FileExistsError(errno.EEXIST, "File exists"),
        )

        def _fail_remove_board(*args, **kwargs):
            raise error

        monkeypatch.setattr(kanban_cli.kb, "remove_board", _fail_remove_board)

        result = kanban_cli._cmd_boards_rm(
            argparse.Namespace(slug="nuke", delete=True, boards_action="rm")
        )

        assert result == 1
        assert str(recovery_path) in capsys.readouterr().err

    def test_boards_list_default_only(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        res = _cli(["boards", "list", "--json"], env_extra=env)
        assert res.returncode == 0, res.stderr
        data = json.loads(res.stdout)
        slugs = [b["slug"] for b in data]
        assert slugs == ["default"]
        assert data[0]["is_current"] is True

    def test_boards_create_and_switch(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        r1 = _cli(
            ["boards", "create", "myproj", "--name", "My Project", "--switch"],
            env_extra=env,
        )
        assert r1.returncode == 0, r1.stderr
        assert "created" in r1.stdout
        assert "Switched" in r1.stdout

        r2 = _cli(["boards", "list", "--json"], env_extra=env)
        data = json.loads(r2.stdout)
        cur = [b for b in data if b["is_current"]][0]
        assert cur["slug"] == "myproj"

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

    def test_board_flag_rejects_unknown(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        r = _cli(["--board", "ghost", "list"], env_extra=env)
        # main.py's dispatcher doesn't propagate return codes today, so we
        # assert the user-visible signal: a stderr error message. Whether
        # the exit code stays 0 is a separate (pre-existing) issue.
        assert "does not exist" in r.stderr

    def test_board_flag_rejects_empty_board_dir(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        ghost = tmp_path / "kanban" / "boards" / "ghost"
        ghost.mkdir(parents=True)
        r = _cli(["--board", "ghost", "list"], env_extra=env)
        assert "does not exist" in r.stderr

    def test_boards_rm_archives(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        _cli(["boards", "create", "rmme"], env_extra=env)
        r = _cli(["boards", "rm", "rmme"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "archived" in r.stdout
        # Default board list no longer shows it.
        res = _cli(["boards", "list", "--json"], env_extra=env)
        slugs = [b["slug"] for b in json.loads(res.stdout)]
        assert "rmme" not in slugs
