import sqlite3
from pathlib import Path

import hermes_cli.doctor as doctor


def test_check_persistent_memory_health_reports_db_and_exports(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("alpha", encoding="utf-8")
    (memories / "USER.md").write_text("beta", encoding="utf-8")

    db_path = hermes_home / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memory_entries (id TEXT PRIMARY KEY, target TEXT, kind TEXT, content TEXT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO memory_entries(id, target, kind, content, status) VALUES ('1', 'memory', 'lesson', 'alpha', 'active')"
    )
    conn.execute(
        "INSERT INTO memory_entries(id, target, kind, content, status) VALUES ('2', 'user', 'preference', 'beta', 'active')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(doctor, "_DHH", "~/.hermes")

    doctor._check_persistent_memory_health([])
    out = capsys.readouterr().out

    assert "memory.db exists (2 active entries)" in out
    assert "MEMORY.md exists (5 chars)" in out
    assert "USER.md exists (4 chars)" in out


def test_check_persistent_memory_health_warns_on_db_issue(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "memory.db").write_text("not a sqlite db", encoding="utf-8")

    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(doctor, "_DHH", "~/.hermes")

    doctor._check_persistent_memory_health([])
    out = capsys.readouterr().out

    assert "memory.db exists but has issues" in out
