"""Import warnings for older Kanban boards (#137157)."""

import sqlite3
import zipfile
from argparse import Namespace
from contextlib import closing
from pathlib import Path

import pytest


class TestImportKanbanDatabaseWarning:
    """Regression for #137157: older boards must not be restored silently."""

    @pytest.mark.parametrize("board_rel", ["kanban.db", "kanban/boards/project/kanban.db"])
    @pytest.mark.parametrize("imported_tasks, imported_events", [(2, 4), (2, 5), (3, 3), (2, 2)])
    def test_import_reports_smaller_kanban_board(
        self, tmp_path, monkeypatch, capsys, board_rel, imported_tasks, imported_events,
    ):
        from hermes_cli.backup import run_import

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        live_db = home / board_rel
        _write_kanban_db(live_db, tasks=3, events=4)
        staged = tmp_path / "backup-kanban.db"
        _write_kanban_db(staged, tasks=imported_tasks, events=imported_events)
        zip_path = tmp_path / "backup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("config.yaml", "model: restored\n")
            zf.write(staged, board_rel)

        inode_before = live_db.stat().st_ino
        with closing(sqlite3.connect(live_db)) as holder:
            assert holder.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 3
            assert run_import(Namespace(zipfile=str(zip_path), force=True)) is None
            assert holder.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == imported_tasks
        assert live_db.stat().st_ino == inode_before

        out = capsys.readouterr().out
        assert "Kanban board data replaced by older backup contents" in out
        assert f"{board_rel}: 3 task(s) / 4 event(s) -> {imported_tasks} / {imported_events}" in out
        with sqlite3.connect(live_db) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == imported_tasks
            assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == imported_events

    @pytest.mark.parametrize("board_rel", ["kanban.db", "kanban/boards/project/kanban.db"])
    @pytest.mark.parametrize("imported_tasks, imported_events", [(3, 4), (4, 5), (3, 5), (4, 4)])
    def test_import_equal_or_newer_kanban_board_has_no_older_warning(
        self, tmp_path, monkeypatch, capsys, board_rel, imported_tasks, imported_events,
    ):
        from hermes_cli.backup import run_import

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        live_db = home / board_rel
        _write_kanban_db(live_db, tasks=3, events=4)
        staged = tmp_path / "backup-kanban.db"
        _write_kanban_db(staged, tasks=imported_tasks, events=imported_events)
        zip_path = tmp_path / "backup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("config.yaml", "model: restored\n")
            zf.write(staged, board_rel)

        assert run_import(Namespace(zipfile=str(zip_path), force=True)) is None

        assert "Kanban board data replaced by older backup contents" not in capsys.readouterr().out
        with sqlite3.connect(live_db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == imported_tasks
            assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == imported_events


def _write_kanban_db(path: Path, tasks: int, events: int) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    # Real cards emit created events; comments add history without adding cards.
    # Distinct boards can therefore have fewer cards but equal or more events.
    kb.init_db(path)
    with kbc.connect_closing(path) as conn:
        task_ids = [kb.create_task(conn, title=f"Task {i}", workspace_kind="scratch") for i in range(tasks)]
        for i in range(events - tasks):
            kb.add_comment(conn, task_ids[0], "operator", f"Progress {i}")
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == tasks
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("kind", ["missing", "invalid", "unrelated"])
def test_kanban_count_unknown_database_is_best_effort(tmp_path, kind):
    from hermes_cli.backup_restore import _count_kanban_rows

    path = tmp_path / "other.db"
    if kind == "invalid":
        path.write_bytes(b"not a SQLite database")
    elif kind == "unrelated":
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE other (value TEXT)")
            conn.execute("INSERT INTO other VALUES ('preserved')")
    before = path.read_bytes() if path.exists() else None

    assert _count_kanban_rows(path) is None
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.platforms("posix")
def test_kanban_count_unreadable_database_is_best_effort(tmp_path):
    import os

    from hermes_cli.backup_restore import _count_kanban_rows

    path = tmp_path / "locked.db"
    path.write_bytes(b"unreadable")
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("host can read files with mode 0")
        assert _count_kanban_rows(path) is None
    finally:
        path.chmod(0o600)


def test_import_unrelated_sqlite_database_has_no_data_loss_warning(tmp_path, monkeypatch, capsys):
    from hermes_cli.backup import run_import

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    live_db = home / "plugin.db"
    staged = tmp_path / "backup-plugin.db"
    for path, value in [(live_db, "live"), (staged, "backup")]:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE plugin_data (value TEXT)")
            conn.execute("INSERT INTO plugin_data VALUES (?)", (value,))
    zip_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("config.yaml", "model: restored\n")
        zf.write(staged, "plugin.db")

    assert run_import(Namespace(zipfile=str(zip_path), force=True)) is None

    out = capsys.readouterr().out
    assert "Import complete" in out
    assert "data replaced by older backup contents" not in out
    with sqlite3.connect(live_db) as conn:
        assert conn.execute("SELECT value FROM plugin_data").fetchone()[0] == "backup"
