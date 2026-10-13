"""Integrationstests fuer hermes_cli.kanban_auto_archive (21-Tage-Regel).

Laeuft gegen eine frische, isolierte Test-DB (tmp_path + HERMES_HOME),
so wie die restlichen Kanban-DB-Tests in tests/hermes_cli/.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_auto_archive as auto
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolierte HERMES_HOME mit frischer, leerer Kanban-DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect_closing() as c:
        yield c


CONFIG_KEYS = ("KANBAN_AUTO_ARCHIVE_AFTER_DAYS", "KANBAN_STALE_THRESHOLD_DAYS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)


def _task(conn: sqlite3.Connection, title: str, status: str, age_days: float) -> str:
    """Erzeugt einen Task mit rueckgerechnetem created_at (spec-Falldecke)."""
    tid = kb.create_task(
        conn, title=title, assignee="operator", created_by="test",
        initial_status="running",
    )
    conn.execute(
        "UPDATE tasks SET status = ?, created_at = ? WHERE id = ?",
        (status, int(time.time() - age_days * 86400), tid),
    )
    conn.commit()
    return tid


def test_task_older_than_21d_is_archived(conn):
    old = _task(conn, "alter task", "todo", age_days=25)
    new = _task(conn, "frischer task", "todo", age_days=2)

    archived = auto.auto_archive(conn)

    assert [t["id"] for t in archived] == [old]
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (old,)).fetchone()[0] == "archived"
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (new,)).fetchone()[0] == "todo"


def test_task_younger_than_3d_floor_is_never_archived(conn):
    """Spec-Fall: '<3 Tage alt' (hier: 10 Tage, unter der 21d-Schwelle) bleibt unberuehrt."""
    mid = _task(conn, "10-Tage-Task", "ready", age_days=10)
    archived = auto.auto_archive(conn)
    assert archived == []
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (mid,)).fetchone()[0] == "ready"


def test_stale_floor_above_archive_days_rejected():
    """Floor > Schwelle wuerde die Regel stilllegen -> Konfigfehler, kein stiller Fixup."""
    with pytest.raises(ValueError):
        auto.load_settings(env={"KANBAN_AUTO_ARCHIVE_AFTER_DAYS": "1", "KANBAN_STALE_THRESHOLD_DAYS": "3"})


def test_audit_log_written_with_trigger(conn):
    old = _task(conn, "old ready task", "ready", age_days=30)
    auto.auto_archive(conn)
    row = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id=? AND kind=?",
        (old, auto.EVENT_KIND),
    ).fetchone()
    assert row is not None
    payload = json.loads(row[1])
    assert payload["old_status"] == "ready"
    assert payload["new_status"] == "archived"
    assert payload["trigger"] == "auto-21d"
    assert payload["task_id"] == old


def test_idempotent_second_run_archives_nothing(conn):
    old = _task(conn, "old task", "todo", age_days=40)
    first = auto.auto_archive(conn)
    assert [t["id"] for t in first] == [old]
    second = auto.auto_archive(conn)
    assert second == []
    # nur ein Audit-Event, kein Doppel-Archivieren
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind=?",
        (old, auto.EVENT_KIND),
    ).fetchone()[0] == 1


def test_running_and_done_are_not_touched(conn):
    running = _task(conn, "running task", "running", age_days=60)
    done = _task(conn, "done task", "done", age_days=60)
    archived_ids = {t["id"] for t in auto.auto_archive(conn)}
    assert running not in archived_ids and done not in archived_ids
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (running,)).fetchone()[0] == "running"
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (done,)).fetchone()[0] == "done"


def test_settings_from_env_override():
    s = auto.load_settings(env={"KANBAN_AUTO_ARCHIVE_AFTER_DAYS": "7"})
    assert s.archive_after_days == 7
    assert s.stale_threshold_days == auto.DEFAULT_STALE_THRESHOLD_DAYS


def test_main_exit_code_0_on_success(conn, monkeypatch, capsys):
    _task(conn, "some old", "todo", age_days=30)
    rc = auto.main(["--dry-run"])
    assert rc == 0


def test_main_exit_code_1_on_error(monkeypatch, tmp_path):
    # connect() legt fehlende DBs automatisch an -> fuer rc=1 eine echte
    # Unmoeglichkeit: Pfad ist ein Verzeichnis, sqlite connect schlagen fehl.
    bad = tmp_path / "not-a-db"
    bad.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(bad))
    rc = auto.main([])
    assert rc == 1


def test_dry_run_changes_nothing(conn):
    old = _task(conn, "old dry", "todo", age_days=30)
    candidates = auto.auto_archive(conn, auto.load_settings({}), dry_run=True)
    assert [t["id"] for t in candidates] == [old]
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (old,)).fetchone()[0] == "todo"
    # kein Event im Dry-Run
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind=?", (old, auto.EVENT_KIND)
    ).fetchone()[0] == 0