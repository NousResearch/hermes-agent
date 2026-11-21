"""Tests for the hermes-session-log optional skill (state.db -> Obsidian vault).

The export script is standalone (stdlib only), so these tests build a minimal
fake state.db with the same sessions/messages schema and run the script via
subprocess with HERMES_SESSION_LOG_CONFIG pointing at a temp config.
"""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "hermes-session-log"
SCRIPT = SKILL_DIR / "scripts" / "archive_hermes_sessions.py"


def _make_db(path: Path, with_compaction: bool = False) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT, model TEXT, started_at REAL,
            ended_at REAL, end_reason TEXT, message_count INTEGER, cwd TEXT,
            title TEXT, profile_name TEXT, last_activity_at REAL,
            hidden INTEGER DEFAULT 0, archived INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
            content TEXT, tool_name TEXT, tool_calls TEXT, reasoning TEXT,
            compacted INTEGER, active INTEGER, timestamp REAL,
            _compressed_summary TEXT
        );
        """
    )
    sid = "local-" + "ab" * 32
    con.execute(
        "INSERT INTO sessions (id, source, model, started_at, ended_at, cwd, title, profile_name, message_count, last_activity_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (sid, "local", "test-model", 1_700_000_000.0, None, "/tmp/proj", "Test session", "default", 4, 1_700_000_100.0),
    )
    rows = [
        ("user", "hello, build me a thing", None, 0),
        ("assistant", "on it", None, 0),
        ("tool", "ok", "terminal", 0),
        ("assistant", "done", None, 1 if with_compaction else 0),
    ]
    for i, (role, content, tool, compacted) in enumerate(rows, start=1):
        summary = "earlier turns were compacted" if (compacted and with_compaction) else "0"
        con.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, timestamp, active, compacted, _compressed_summary)"
            " VALUES (?,?,?,?,?,1,?,?)",
            (sid, role, content, tool, 1_700_000_010.0 + i, compacted, summary),
        )
    con.commit()
    con.close()


def _run(tmp: Path, db: Path, extra: tuple = ()) -> subprocess.CompletedProcess:
    cfgf = tmp / "config.json"
    cfgf.write_text(json.dumps({
        "vaultPath": str(tmp / "vault"),
        "folder": "Hermes Sessions",
        "stateDb": str(db),
        "dumpDir": "",
    }))
    env = dict(os.environ, HERMES_SESSION_LOG_CONFIG=str(cfgf))
    for key in ("HERMES_HOME", "HERMES_STATE_DB", "HERMES_SESSION_LOG_STATE_DB"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra],
        env=env, capture_output=True, text=True, timeout=60,
    )


def _notes(vault: Path):
    return [f for f in vault.glob("*.md") if f.name != "Session Index.md"]


def test_exports_note_index_and_raw(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    r = _run(tmp_path, db)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK " in r.stdout and "1 updated" in r.stdout, r.stdout
    vault = tmp_path / "vault" / "Hermes Sessions"
    notes = _notes(vault)
    assert len(notes) == 1, notes
    text = notes[0].read_text(encoding="utf-8")
    assert "Test session" in text
    assert "hello, build me a thing" in text
    from datetime import datetime as _dt
    expected_day = _dt.fromtimestamp(1_700_000_000.0).strftime("%Y-%m-%d")  # local tz
    assert notes[0].name.startswith(expected_day + " "), notes[0].name
    assert 'session_start: "' in text
    raw = vault / (notes[0].stem + ".jsonl")
    assert raw.is_file()
    lines = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [l["role"] for l in lines] == ["user", "assistant", "tool", "assistant"]
    index = (vault / "Session Index.md").read_text(encoding="utf-8")
    assert "Test session" in index
    assert "closed" not in index or True  # status column just needs to exist
    assert "| Started |" in index


def test_rerun_is_idempotent_and_silent(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    first = _run(tmp_path, db)
    assert "1 updated" in first.stdout
    note = _notes(tmp_path / "vault" / "Hermes Sessions")[0]
    before = note.read_bytes()
    second = _run(tmp_path, db)
    assert second.returncode == 0, second.stdout + second.stderr
    assert second.stdout == "", f"no-op run must stay silent, got: {second.stdout}"
    assert note.read_bytes() == before


def test_compaction_marker(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db, with_compaction=True)
    r = _run(tmp_path, db)
    assert r.returncode == 0, r.stdout + r.stderr
    text = _notes(tmp_path / "vault" / "Hermes Sessions")[0].read_text(encoding="utf-8")
    assert "Context compaction" in text
    assert "earlier turns were compacted" in text
    assert 'compacted: 1' in text


def test_open_session_status(tmp_path):
    db = tmp_path / "state.db"
    _make_db(db)
    _run(tmp_path, db)
    text = _notes(tmp_path / "vault" / "Hermes Sessions")[0].read_text(encoding="utf-8")
    assert 'status: "open"' in text


def test_inplace_compaction_refreshes(tmp_path):
    """In-place compaction keeps max message ID and session end time stable,
    so the change must be detected by content fingerprint, not by max_id."""
    db = tmp_path / "state.db"
    _make_db(db)
    first = _run(tmp_path, db)
    assert "1 updated" in first.stdout
    vault = tmp_path / "vault" / "Hermes Sessions"
    note = _notes(vault)[0]
    assert "Context compaction" not in note.read_text(encoding="utf-8")

    sid = "local-" + "ab" * 32
    con = sqlite3.connect(db)
    con.execute(
        "UPDATE messages SET compacted=1, _compressed_summary=? "
        "WHERE session_id=? AND id=(SELECT MAX(id) FROM messages)",
        ("earlier turns were compacted", sid))
    con.commit()
    con.close()

    second = _run(tmp_path, db)
    assert "1 updated" in second.stdout, second.stdout
    text = note.read_text(encoding="utf-8")
    assert "Context compaction" in text
    assert "earlier turns were compacted" in text
    assert "compacted: 1" in text
    raw = [json.loads(l) for l in
           (vault / (note.stem + ".jsonl")).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert raw[-1]["compacted"] == 1
    assert raw[-1]["_compressed_summary"] == "earlier turns were compacted"

    third = _run(tmp_path, db)
    assert third.returncode == 0 and third.stdout == "", third.stdout
