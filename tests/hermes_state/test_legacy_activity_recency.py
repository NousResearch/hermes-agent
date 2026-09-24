"""Legacy nonnumeric activity cells must not outrank real activity (#120418)."""
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_cli.status import _render_sessions


@pytest.mark.parametrize("bad", ["last_activity_at", "", b"last_activity_at", None])
@pytest.mark.parametrize("has_message", [False, True])
def test_gateway_recency_ignores_nonnumeric_heartbeat(tmp_path, monkeypatch, capsys, bad, has_message):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", path)
    db = SessionDB(db_path=path)
    try:
        for sid in ("legacy", "recent"):
            db.create_session(sid, "telegram", session_key=f"telegram:{sid}")
        if has_message:
            db.append_message("legacy", "user", "preserve this message")
    finally:
        db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE sessions SET started_at=1700000000, last_activity_at=? WHERE id='legacy'", (bad,))
        conn.execute("UPDATE sessions SET started_at=1700000100, last_activity_at=1700000300 WHERE id='recent'")
        conn.execute("UPDATE messages SET timestamp=1700000200 WHERE session_id='legacy'")
    db = SessionDB(db_path=path, read_only=True)
    try:
        rows = db.list_gateway_sessions()
        assert [r["id"] for r in rows] == ["recent", "legacy"]
        assert rows[1]["last_active"] == (1700000200 if has_message else 1700000000)
        assert db.get_session("legacy")["last_activity_at"] == bad
    finally:
        db.close()
    _render_sessions(SimpleNamespace(config={}))
    assert "Last activity:" in capsys.readouterr().out


@pytest.mark.parametrize("heartbeat", [1700000100, 1700000300, "1700000300"])
def test_freshest_numeric_message_or_heartbeat_wins(tmp_path, heartbeat):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session("s", "telegram", session_key="telegram:s")
        db.append_message("s", "user", "valid")
        db.append_message("s", "assistant", "legacy timestamp")
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE sessions SET last_activity_at=? WHERE id='s'", (heartbeat,))
            conn.execute("UPDATE messages SET timestamp=1700000200 WHERE role='user'")
            conn.execute("UPDATE messages SET timestamp='timestamp' WHERE role='assistant'")
        expected = max(float(heartbeat), 1700000200)
        assert db.list_gateway_sessions()[0]["last_active"] == expected
        assert db.list_sessions_rich()[0]["last_active"] == expected
        assert db.search_sessions(limit=1)[0]["last_active"] == expected
    finally:
        db.close()


def test_status_reads_legacy_activity_without_crashing(tmp_path, monkeypatch, capsys):
    path = tmp_path / "state.db"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", path)
    db = SessionDB(db_path=path)
    try:
        db.create_session("s", "telegram", session_key="telegram:s")
    finally:
        db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE sessions SET last_activity_at='last_activity_at'")
    _render_sessions(SimpleNamespace(config={}))
    output = capsys.readouterr().out
    assert "1 session(s)" in output
    assert "Last activity:" in output
