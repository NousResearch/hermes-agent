import sqlite3

from hermes_cli import kanban_db


def _journal_mode(path):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


def test_connect_respects_delete_journal_policy(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_JOURNAL", "delete")

    conn = kanban_db.connect(db_path=db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        conn.close()

    assert _journal_mode(db_path) == "delete"


def test_connect_defaults_to_delete_journal_policy(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.delenv("HERMES_KANBAN_JOURNAL", raising=False)

    conn = kanban_db.connect(db_path=db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        conn.close()

    assert _journal_mode(db_path) == "delete"
