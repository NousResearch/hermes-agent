"""worktree_id (ORCA-06 kanban<->orca auto-link) — DB layer, CLI, migration.

Covers the additive ``tasks.worktree_id`` column end to end below the MCP
surface: identifier validation, create/edit round-trip, the guarantee that
cards without a link behave exactly as before, and the migration applying
cleanly to a pre-existing board DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    "has space", "!!bad", "-leading-dash", ".leading-dot", "/leading-slash",
    "tab\tsep", "x" * 256,
])
def test_validate_rejects_non_identifiers(bad):
    with pytest.raises(ValueError, match="worktree_id"):
        kb.validate_worktree_id(bad)


@pytest.mark.parametrize("bad_type", [123, ["wt_1"], {"id": 1}])
def test_validate_requires_a_string(bad_type):
    with pytest.raises(ValueError, match="string"):
        kb.validate_worktree_id(bad_type)


@pytest.mark.parametrize("good", [
    "wt_9f2c1a", "team/wt-42", "abc", "a" * 255, "WT.2026.10", "9numeric-start",
    "repo-42::/Users/me/orca/fix-login",  # Orca two-part address <repoId>::<path>
])
def test_validate_accepts_identifiers(good):
    assert kb.validate_worktree_id(good) == good


def test_validate_blank_and_none_mean_no_link():
    assert kb.validate_worktree_id(None) is None
    assert kb.validate_worktree_id("") is None
    assert kb.validate_worktree_id("   ") is None
    assert kb.validate_worktree_id("  wt_1  ") == "wt_1"  # surrounding whitespace stripped


# ---------------------------------------------------------------------------
# Create / read round-trip
# ---------------------------------------------------------------------------


def test_create_without_link_defaults_to_none(conn):
    tid = kb.create_task(conn, title="plain", assignee="worker")
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.worktree_id is None


def test_create_roundtrips_link(conn):
    tid = kb.create_task(
        conn, title="orca-linked", assignee="worker", worktree_id="wt_9f2c1a")
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.worktree_id == "wt_9f2c1a"
    # The created event carries the link for auditability.
    ev = next(e for e in kb.list_events(conn, tid) if e.kind == "created")
    assert ev.payload["worktree_id"] == "wt_9f2c1a"


def test_create_blank_stores_null(conn):
    tid = kb.create_task(conn, title="blank", assignee="worker", worktree_id="   ")
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.worktree_id is None


def test_create_rejects_invalid_link(conn):
    with pytest.raises(ValueError, match="worktree_id"):
        kb.create_task(conn, title="bad", assignee="worker", worktree_id="!!")
    # Nothing was written.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE title = 'bad'").fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# Edit: set / replace / clear / untouched
# ---------------------------------------------------------------------------


def test_edit_set_replace_clear_untouched(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")

    # None (default) leaves the link untouched.
    assert kb.edit_task(conn, tid, title="t2") is True
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.worktree_id is None

    # Set, then replace.
    assert kb.edit_task(conn, tid, worktree_id="wt_1") is True
    assert kb.get_task(conn, tid).worktree_id == "wt_1"
    assert kb.edit_task(conn, tid, worktree_id="wt_2") is True
    assert kb.get_task(conn, tid).worktree_id == "wt_2"

    # Blank clears.
    assert kb.edit_task(conn, tid, worktree_id="  ") is True
    assert kb.get_task(conn, tid).worktree_id is None

    # Other edits don't touch the link.
    kb.edit_task(conn, tid, worktree_id="wt_3")
    assert kb.edit_task(conn, tid, priority=5) is True
    assert kb.get_task(conn, tid).worktree_id == "wt_3"

    # Invalid values are rejected and not persisted.
    with pytest.raises(ValueError, match="worktree_id"):
        kb.edit_task(conn, tid, worktree_id="no spaces")
    assert kb.get_task(conn, tid).worktree_id == "wt_3"

    # An "edited" event records the field change.
    events = [e for e in kb.list_events(conn, tid) if e.kind == "edited"]
    assert any("worktree_id" in (e.payload.get("fields") or []) for e in events)


# ---------------------------------------------------------------------------
# Migration — pre-existing deployments
# ---------------------------------------------------------------------------


def test_migration_appends_column_to_preexisting_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="legacy")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Emulate a pre-existing deployment: current schema MINUS worktree_id,
    # with real rows in it.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(tasks)")]
    keep = [c for c in cols if c != "worktree_id"]
    conn.execute(f"CREATE TABLE tasks_legacy AS SELECT {', '.join(keep)} FROM tasks")
    conn.execute("DROP TABLE tasks")
    conn.execute("ALTER TABLE tasks_legacy RENAME TO tasks")
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at, assignee) "
        "VALUES ('t-1', 'legacy card', 'ready', 1700000000, 'someone')")
    conn.commit()
    conn.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    # A fresh process connecting (CLI / dispatcher / dashboard) migrates in place.
    with kbc.connect_closing(db_path=db_path) as conn:
        cols_now = [r["name"] for r in conn.execute("PRAGMA table_info(tasks)")]
        assert "worktree_id" in cols_now
        row = conn.execute(
            "SELECT title, status, assignee, worktree_id FROM tasks WHERE id = 't-1'"
        ).fetchone()
        assert row["title"] == "legacy card" and row["status"] == "ready"
        assert row["assignee"] == "someone"
        assert row["worktree_id"] is None
        # Post-migration writes work.
        tid = kb.create_task(conn, title="post-migration", assignee="worker", worktree_id="wt_x")
        assert kb.get_task(conn, tid).worktree_id == "wt_x"


def test_migration_is_idempotent(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="idem")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb.init_db(db_path=db_path)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    # Second full init pass (as a restart or concurrent first-connect would do).
    kb.init_db(db_path=db_path)
    with kbc.connect_closing(db_path=db_path) as conn:
        assert "worktree_id" in {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}


# ---------------------------------------------------------------------------
# CLI round-trip
# ---------------------------------------------------------------------------


def test_cli_create_edit_show_roundtrip(kanban_home):
    out = kc.run_slash("create 'orca card' --assignee worker --worktree-id wt_9f2c1a")
    assert "Created" in out
    tid = out.split()[1]

    shown = json.loads(kc.run_slash(f"show {tid} --json"))
    assert shown["task"]["worktree_id"] == "wt_9f2c1a"

    text = kc.run_slash(f"show {tid}")
    assert "wt_9f2c1a" in text

    # Replace via edit.
    kc.run_slash(f"edit {tid} --worktree-id wt_new")
    assert json.loads(
        kc.run_slash(f"show {tid} --json"))["task"]["worktree_id"] == "wt_new"

    # Blank clears.
    kc.run_slash(f'edit {tid} --worktree-id ""')
    assert json.loads(
        kc.run_slash(f"show {tid} --json"))["task"]["worktree_id"] is None


def test_cli_create_without_link_is_unchanged(kanban_home):
    out = kc.run_slash("create 'plain card' --assignee worker")
    tid = out.split()[1]
    shown = json.loads(kc.run_slash(f"show {tid} --json"))
    assert shown["task"]["worktree_id"] is None


def test_cli_edit_without_link_flag_leaves_it(kanban_home):
    out = kc.run_slash("create 'linked card' --assignee worker --worktree-id wt_keep")
    tid = out.split()[1]
    kc.run_slash(f"edit {tid} --title 'renamed'")
    assert json.loads(
        kc.run_slash(f"show {tid} --json"))["task"]["worktree_id"] == "wt_keep"


def test_cli_rejects_invalid_link(kanban_home):
    out = kc.run_slash("create 'bad' --assignee worker --worktree-id 'has space'")
    assert "worktree_id" in out
    titles = [t["title"] for t in json.loads(kc.run_slash("list --json"))]
    assert "bad" not in titles

    # Same guard on the edit path.
    out2 = kc.run_slash("create 'bad2' --assignee worker")
    tid2 = out2.split()[1]
    err = kc.run_slash(f"edit {tid2} --worktree-id '!!'")
    assert "worktree_id" in err
    assert json.loads(kc.run_slash(f"show {tid2} --json"))["task"]["worktree_id"] is None
