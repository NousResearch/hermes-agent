"""Regression for #131075: a retention backup must cover the rows prune can delete."""

import json
import sys
from pathlib import Path

import pytest

from hermes_cli import main
from hermes_state import SessionDB


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with SessionDB() as db:
        for session_id in ("normal", "pinned", "archived"):
            db.create_session(session_id, "cli")
            db.append_message(session_id, "user", f"Keep {session_id}")
            db.end_session(session_id, "done")
        db.set_session_pinned("pinned", True)
        db.set_session_archived("archived", True)
        db._conn.execute("UPDATE sessions SET started_at=1")
        db._conn.commit()
    return home


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *args])
    return main.main()


def test_filtered_backup_covers_protected_sessions_before_retention_prune(store, tmp_path, monkeypatch):
    backup = tmp_path / "backup.jsonl"
    _run(monkeypatch, "export", "--before", "2000-01-01", "--format", "jsonl", "--yes", str(backup))
    exported = {row["id"]: row for row in map(json.loads, backup.read_text(encoding="utf-8").splitlines())}
    with SessionDB() as db:
        assert Path(db.db_path).resolve().is_relative_to(tmp_path.resolve())
    assert set(exported) == {"normal", "pinned", "archived"}
    assert all(row["messages"] for row in exported.values())
    assert exported["pinned"]["pinned"] == 1
    assert exported["archived"]["archived"] == 1

    # This fixture is the sole writer; bypass the unrelated host-wide holder inventory.
    _run(monkeypatch, "prune", "--before", "2000-01-01", "--include-archived", "--include-pinned", "--yes", "--force")
    with SessionDB() as db:
        assert all(db.get_session(session_id) is None for session_id in exported)


def test_explicit_missing_export_fails_without_creating_an_output(store, tmp_path, monkeypatch, capsys):
    for fmt in ("jsonl", "html", "md", "qmd", "trace"):
        output = tmp_path / f"missing-{fmt}"
        with pytest.raises(SystemExit) as error:
            _run(monkeypatch, "export", "--session-id", "missing-session", "--format", fmt, str(output))
        assert error.value.code != 0
        assert "missing-session" in capsys.readouterr().out
        assert not output.exists()
