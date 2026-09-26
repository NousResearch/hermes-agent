"""Doctor distinguishes policy skips from a probe reporting no error."""

import sqlite3
from unittest.mock import Mock

import pytest

from hermes_cli import doctor_state
from hermes_cli.doctor_report import Finding
from hermes_state import SessionDB
import hermes_state_holders
import hermes_state_repair


@pytest.fixture
def state_db(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("existing", source="cli")
    db.append_message("existing", "user", "searchable message")
    db.close()
    return path


@pytest.mark.parametrize("held,should_fix", [(False, False), (True, False), (True, True)])
def test_write_health_distinguishes_skipped_from_completed(
    state_db, monkeypatch, held, should_fix, capsys,
):
    monkeypatch.setattr(hermes_state_holders, "live_writer_holds_db", lambda *_a, **_k: held)
    monkeypatch.setattr(doctor_state, "_WRITE_PROBE_SNAPSHOT_MAX_BYTES", 0)
    probe = Mock(wraps=hermes_state_repair._db_opens_cleanly)
    connect = Mock(wraps=sqlite3.connect)
    monkeypatch.setattr(hermes_state_repair, "_db_opens_cleanly", probe)
    monkeypatch.setattr(sqlite3, "connect", connect)

    result = doctor_state._write_health_result(state_db, should_fix=should_fix)

    if held and not should_fix:
        assert result.status == "skipped"
        assert "live writer" in result.reason and "--fix" in result.reason
        probe.assert_not_called()
        connect.assert_not_called()
    else:
        assert result.status == "no_error_reported" and result.reason is None
        probe.assert_called_once()
        probed_path = probe.call_args.args[0]
        assert (probed_path != state_db) == held
    # Rendering belongs to the consumer, even when the probe is skipped.
    assert capsys.readouterr().out == ""
    with sqlite3.connect(state_db) as conn:
        assert conn.execute("SELECT id FROM sessions").fetchall() == [("existing",)]
        assert conn.execute("SELECT content FROM messages").fetchall() == [("searchable message",)]


@pytest.mark.parametrize("outcome", ["no_error_reported", "failed", "skipped", "missing_schema"])
def test_doctor_only_routes_failed_write_probes_to_repair(state_db, monkeypatch, capsys, outcome):
    monkeypatch.setattr(
        hermes_state_holders, "live_writer_holds_db", lambda *_a, **_k: outcome == "skipped",
    )
    monkeypatch.setattr(doctor_state, "_WRITE_PROBE_SNAPSHOT_MAX_BYTES", 0)
    if outcome == "failed":
        with sqlite3.connect(state_db) as conn:
            conn.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    elif outcome == "missing_schema":
        state_db = state_db.with_name("incomplete.db")
        with sqlite3.connect(state_db) as conn:
            conn.execute("CREATE TABLE sessions (id TEXT, source TEXT, started_at REAL)")
    repair = Mock()
    monkeypatch.setattr(doctor_state, "_repair_state_db", repair)

    result = doctor_state._write_health_result(state_db, should_fix=False)
    assert result.status == ("no_error_reported" if outcome == "missing_schema" else outcome)
    finding = Finding()
    doctor_state._state_db_health(finding, False, state_db, "test-home")
    output = capsys.readouterr().out

    if outcome == "failed":
        assert result.reason and result.reason in output
        assert "fails a write-health probe" in output
        repair.assert_called_once_with(finding, False, state_db, "fts")
    else:
        repair.assert_not_called()
        assert "fails a write-health probe" not in output
        if outcome == "skipped":
            assert f"write-health probe skipped: {result.reason}" in output
        else:
            assert "write-health probe skipped" not in output
    assert finding.fixed == 0
    if outcome == "missing_schema":
        assert result.reason is None
        with sqlite3.connect(state_db) as conn:
            # Missing messages prevented a complete write probe; its session
            # was still rolled back, and no error does not certify health.
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
            assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'messages'").fetchall() == []
