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
import json
import os
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
        # Regression for #23833: a poll loop that re-creates a just-removed
        # board must get a fresh schema-init pass — if _INITIALIZED_PATHS
        # still contained the resolved path, the CREATE TABLE pass would be
        # skipped and downstream readers hit `no such table: task_events`.
        # (Since #43243 connect() itself refuses to recreate a removed board,
        # so the re-creation goes through create_board, as it should.)
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

        # Simulate the board being re-created at the same slug: the schema
        # must be re-applied on the fresh DB file.
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
        check=False,
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





# ---------------------------------------------------------------------------
# Archived / deleted board resurrection (#43243)
# ---------------------------------------------------------------------------

class TestBoardResurrection:
    """Read/watch paths must not resurrect archived or deleted boards.

    Stale dashboard tabs, gateway notifiers and event-stream pollers can hand
    ``connect(board=<slug>)`` a slug whose board was archived (moved to
    ``_archived/`` with an ``archived`` tombstone) or hard-deleted
    (``rmtree``).  ``connect`` must refuse to recreate the directory/DB, and
    ``list_boards`` must not surface a DB-only stub as an active board.
    """

    def test_connect_does_not_recreate_archived_board(self, fresh_home):
        kb.create_board("gone")
        kb.remove_board("gone", archive=True)
        tombstone = kb.board_metadata_path("gone")
        assert tombstone.exists() and kb.read_board_metadata("gone")["archived"] is True
        with pytest.raises(ValueError, match="archived"):
            kbc.connect(board="gone")
        assert not (kb.board_dir("gone") / "kanban.db").exists()

    def test_connect_does_not_recreate_deleted_board(self, fresh_home):
        kb.create_board("nuked")
        kb.remove_board("nuked", archive=False)
        assert not kb.board_dir("nuked").exists()
        with pytest.raises(ValueError, match="does not exist"):
            kbc.connect(board="nuked")
        assert not kb.board_dir("nuked").exists()

    def test_connect_still_opens_live_board(self, fresh_home):
        kb.create_board("live")
        with kbc.connect_closing(board="live") as conn:
            conn.execute("SELECT 1").fetchone()
        assert kb.board_exists("live")

    def test_list_boards_ignores_db_only_stub(self, fresh_home):
        # Simulate the historical stub shape: kanban.db with no board.json.
        stub = kb.board_dir("ghost")
        stub.mkdir(parents=True)
        (stub / "kanban.db").touch()
        assert "ghost" not in [b["slug"] for b in kb.list_boards()]
        assert "ghost" not in [b["slug"] for b in kb.list_boards(include_archived=True)]

    def test_archive_tombstone_keeps_name(self, fresh_home):
        kb.create_board("keepname", name="My Board")
        kb.remove_board("keepname", archive=True)
        meta = kb.read_board_metadata("keepname")
        assert meta["archived"] is True
        assert meta["name"] == "My Board"
        assert "keepname" in [b["slug"] for b in kb.list_boards(include_archived=True)]
        assert "keepname" not in [b["slug"] for b in kb.list_boards(include_archived=False)]


def _task_snapshot(slug: str) -> list[tuple[str, str]]:
    with kbc.connect_closing(board=slug) as conn:
        tasks = kb.list_tasks(conn, include_archived=True)
    return sorted((t.title, t.status) for t in tasks)


def _archived_titles(path: Path) -> list[str]:
    conn = sqlite3.connect(path / "kanban.db")
    try:
        return sorted(row[0] for row in conn.execute("SELECT title FROM tasks"))
    finally:
        conn.close()


class TestBoardRestore:
    """``boards restore`` undoes ``remove_board(archive=True)`` without clobbering."""

    def test_restore_puts_tasks_and_metadata_back(self, fresh_home):
        kb.create_board("proj", name="Proj Name", description="keep me")
        with kbc.connect_closing(board="proj") as conn:
            kb.create_task(conn, title="running-one", assignee="dev")
            kb.create_task(conn, title="blocked-one", assignee="dev", initial_status="blocked")
            kb.create_task(conn, title="triage-one", assignee="dev", triage=True)
        before = _task_snapshot("proj")
        kb.remove_board("proj", archive=True)
        assert kb.read_board_metadata("proj")["archived"] is True
        assert not (kb.board_dir("proj") / "kanban.db").exists()

        res = kb.restore_board("proj")
        assert res["action"] == "restored"
        assert res["slug"] == "proj"
        assert Path(res["new_path"]) == kb.board_dir("proj")
        assert not Path(res["from_path"]).exists()
        assert _task_snapshot("proj") == before
        meta = kb.read_board_metadata("proj")
        assert meta["archived"] is False
        assert meta["name"] == "Proj Name"
        assert meta["description"] == "keep me"
        assert not kb.has_archived_board("proj")
        assert not (kb.boards_root() / ".proj.restore-tombstone").exists()

    def test_restore_refuses_live_board_and_keeps_both_copies(self, fresh_home):
        kb.create_board("proj", name="Original")
        with kbc.connect_closing(board="proj") as conn:
            kb.create_task(conn, title="archived-task", assignee="dev")
            kb.create_task(conn, title="blocked-task", assignee="dev", initial_status="blocked")
        kb.remove_board("proj", archive=True)
        archived = kb.list_archived_board_dirs("proj")
        assert len(archived) == 1
        kb.create_board("proj", name="Fresh live")
        with kbc.connect_closing(board="proj") as conn:
            kb.create_task(conn, title="do-not-delete", assignee="dev")

        with pytest.raises(ValueError, match="already exists"):
            kb.restore_board("proj")

        snap = _task_snapshot("proj")
        assert [title for title, _status in snap] == ["do-not-delete"]
        assert archived[0].exists()
        assert _archived_titles(archived[0]) == ["archived-task", "blocked-task"]
        assert (kb.board_dir("proj") / "kanban.db").exists()

    def test_restore_refuses_non_tombstone_extras(self, fresh_home):
        kb.create_board("junk")
        with kbc.connect_closing(board="junk") as conn:
            kb.create_task(conn, title="safe-in-archive", assignee="dev")
        kb.remove_board("junk", archive=True)
        live = kb.board_dir("junk")
        (live / "notes.txt").write_text("keep me", encoding="utf-8")
        with pytest.raises(ValueError, match="already exists"):
            kb.restore_board("junk")
        assert (live / "notes.txt").read_text(encoding="utf-8") == "keep me"
        assert (live / "board.json").exists()
        assert _archived_titles(kb.list_archived_board_dirs("junk")[0]) == ["safe-in-archive"]

    def test_restore_replaces_empty_shell(self, fresh_home):
        kb.create_board("shell", name="Shell")
        with kbc.connect_closing(board="shell") as conn:
            kb.create_task(conn, title="inside", assignee="dev")
        kb.remove_board("shell", archive=True)
        live = kb.board_dir("shell")
        (live / "board.json").unlink()
        assert live.is_dir() and not any(live.iterdir())
        kb.restore_board("shell")
        assert _archived_titles(live) == ["inside"]
        assert kb.read_board_metadata("shell")["name"] == "Shell"

    def test_from_rejects_external_and_other_slug(self, fresh_home, tmp_path):
        kb.create_board("proj")
        with kbc.connect_closing(board="proj") as conn:
            kb.create_task(conn, title="stay-archived", assignee="dev")
        kb.remove_board("proj", archive=True)
        archived = kb.list_archived_board_dirs("proj")[0]

        outside = tmp_path / "not-an-archive"
        outside.mkdir()
        (outside / "kanban.db").touch()
        with pytest.raises(ValueError, match="under"):
            kb.restore_board("proj", archive_path=outside)
        assert archived.exists()
        assert (kb.board_dir("proj") / "board.json").exists()
        assert not (kb.board_dir("proj") / "kanban.db").exists()

        kb.create_board("other")
        kb.remove_board("other", archive=True)
        other = kb.list_archived_board_dirs("other")[0]
        with pytest.raises(ValueError, match="does not match"):
            kb.restore_board("proj", archive_path=other)
        assert other.exists() and archived.exists()

        res = kb.restore_board("proj", archive_path=archived)
        assert Path(res["from_path"]) == archived
        assert _archived_titles(kb.board_dir("proj")) == ["stay-archived"]
        assert other.exists()

    def test_slug_prefix_does_not_steal_another_board(self, fresh_home):
        kb.create_board("foo-bar", name="Bar")
        with kbc.connect_closing(board="foo-bar") as conn:
            kb.create_task(conn, title="bar-task", assignee="dev")
        kb.remove_board("foo-bar", archive=True)
        with pytest.raises(ValueError, match="no archived board"):
            kb.restore_board("foo")
        assert kb.has_archived_board("foo-bar")
        assert not kb.has_archived_board("foo")
        assert _archived_titles(kb.list_archived_board_dirs("foo-bar")[0]) == ["bar-task"]

    def test_default_cannot_be_restored(self, fresh_home):
        with pytest.raises(ValueError, match="default"):
            kb.restore_board("default")

    def test_newest_archive_wins_and_older_is_kept(self, fresh_home):
        kb.create_board("multi", name="First")
        with kbc.connect_closing(board="multi") as conn:
            kb.create_task(conn, title="old-task", assignee="dev")
        kb.remove_board("multi", archive=True)
        older = kb.list_archived_board_dirs("multi")[0]
        kb.create_board("multi", name="Second")
        with kbc.connect_closing(board="multi") as conn:
            kb.create_task(conn, title="new-task", assignee="dev")
        kb.remove_board("multi", archive=True)

        res = kb.restore_board("multi")
        assert kb.read_board_metadata("multi")["name"] == "Second"
        assert _archived_titles(kb.board_dir("multi")) == ["new-task"]
        assert older.exists()
        assert _archived_titles(older) == ["old-task"]
        assert Path(res["from_path"]) != older

    def test_post_check_collision_is_valueerror_and_keeps_archive(self, fresh_home, monkeypatch):
        kb.create_board("race")
        with kbc.connect_closing(board="race") as conn:
            kb.create_task(conn, title="keep", assignee="dev")
        kb.remove_board("race", archive=True)
        archived = kb.list_archived_board_dirs("race")[0]
        real_rename = Path.rename

        def racing_rename(self, target):
            target = Path(target)
            if self.parent.name == "_archived" and target.name == "race":
                target.mkdir(parents=True, exist_ok=True)
                (target / "kanban.db").write_bytes(b"live-db")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", racing_rename)
        with pytest.raises(ValueError, match="already exists"):
            kb.restore_board("race")
        assert archived.exists()
        assert _archived_titles(archived) == ["keep"]
        assert (kb.board_dir("race") / "kanban.db").read_bytes() == b"live-db"

    def test_cli_restore_round_trip(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "proj", "--name", "Proj"], env_extra=env).returncode == 0
        created = _cli(
            ["--board", "proj", "create", "Keep me", "--assignee", "dev"],
            env_extra=env,
        )
        assert created.returncode == 0, created.stderr
        removed = _cli(["boards", "rm", "proj"], env_extra=env)
        assert removed.returncode == 0, removed.stderr
        assert "boards restore proj" in removed.stdout
        restored = _cli(["boards", "restore", "proj"], env_extra=env)
        assert restored.returncode == 0, restored.stderr
        listed = _cli(["--board", "proj", "list", "--json"], env_extra=env)
        assert listed.returncode == 0, listed.stderr
        assert [task["title"] for task in json.loads(listed.stdout)] == ["Keep me"]

        rejected = _cli(
            ["boards", "restore", "proj", "--from", str(tmp_path / "outside")],
            env_extra=env,
        )
        assert rejected.returncode == 1
        assert "under" in rejected.stderr

    def test_parser_accepts_restore_and_unarchive(self):
        from hermes_cli.kanban_parser import build_parser

        parent = argparse.ArgumentParser()
        parser = build_parser(parent.add_subparsers(dest="top"))
        restored = parser.parse_args(["boards", "restore", "proj", "--from", "rel"])
        assert restored.boards_action == "restore"
        assert restored.from_path == "rel"
        alias = parser.parse_args(["boards", "unarchive", "proj"])
        assert alias.boards_action == "unarchive"
        assert alias.from_path is None
