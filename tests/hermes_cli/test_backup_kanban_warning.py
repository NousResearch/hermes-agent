"""Import warnings for older Kanban boards (#137157)."""

import sqlite3
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest


class TestImportKanbanDatabaseWarning:
    """Regression for #137157: older boards must not be restored silently."""

    def test_import_reports_smaller_kanban_board(self, tmp_path, monkeypatch, capsys):
        from hermes_cli.backup import run_import

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        live_db = home / "kanban.db"
        _write_kanban_db(live_db, tasks=3, events=4)
        staged = tmp_path / "backup-kanban.db"
        _write_kanban_db(staged, tasks=2, events=2)
        zip_path = tmp_path / "backup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("config.yaml", "model: restored\n")
            zf.write(staged, "kanban.db")

        run_import(Namespace(zipfile=str(zip_path), force=True))

        out = capsys.readouterr().out
        assert "Kanban board data replaced by older backup contents" in out
        assert "kanban.db: 3 task(s) / 4 event(s) -> 2 / 2" in out

    @pytest.mark.parametrize("imported_tasks, imported_events", [(3, 4), (4, 5)])
    def test_import_equal_or_newer_kanban_board_has_no_older_warning(
        self, tmp_path, monkeypatch, capsys, imported_tasks, imported_events,
    ):
        from hermes_cli.backup import run_import

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        live_db = home / "kanban.db"
        _write_kanban_db(live_db, tasks=3, events=4)
        staged = tmp_path / "backup-kanban.db"
        _write_kanban_db(staged, tasks=imported_tasks, events=imported_events)
        zip_path = tmp_path / "backup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("config.yaml", "model: restored\n")
            zf.write(staged, "kanban.db")

        run_import(Namespace(zipfile=str(zip_path), force=True))

        assert "Kanban board data replaced by older backup contents" not in capsys.readouterr().out
        with sqlite3.connect(live_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == imported_tasks
            assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == imported_events


def _write_kanban_db(path: Path, tasks: int, events: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO tasks VALUES (?)", [(str(i),) for i in range(tasks)])
        conn.executemany("INSERT INTO task_events VALUES (?)", [(i,) for i in range(events)])
