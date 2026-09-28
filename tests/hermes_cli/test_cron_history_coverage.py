"""Retained cron history must not be presented as complete run history (#125873)."""
from datetime import datetime, timezone


def test_runs_discloses_retained_window_independently_of_display_limit(monkeypatch, tmp_path, capsys):
    from cron import executions
    from hermes_cli.cron import cron_runs

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 2)
    for day, job in [(1, "quiet"), (2, "busy"), (3, "busy")]:
        stamp = datetime(2026, 1, day, tzinfo=timezone.utc)
        monkeypatch.setattr(executions, "_hermes_now", lambda: stamp)
        row = executions.create_execution(job, source="builtin")
        executions.finish_execution(row["id"], success=True)

    cron_runs("busy", limit=1)
    output = capsys.readouterr().out
    assert "Retained history: 2 attempt(s)" in output
    assert "2026-01-02T00:00:00+00:00" in output
    assert "2026-01-03T00:00:00+00:00" in output
    assert "Showing 1" in output
    assert output.count("source=builtin") == 1
    assert "missing row does not prove" in output

    cron_runs("quiet")
    output = capsys.readouterr().out
    assert "No retained cron execution attempts" in output
    assert "missing row does not prove" in output
    assert "No cron execution attempts recorded." not in output


def test_retained_summary_tracks_selected_home_and_empty_filter(monkeypatch, tmp_path):
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    for name, job in [("a", "first"), ("b", "second")]:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / name))
        executions.create_execution(job, source="builtin")
    for name, job in [("a", "first"), ("b", "second"), ("a", "first")]:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / name))
        summary = executions.execution_history_summary(job_id=job)
        assert summary["retained_count"] == 1
        assert summary["oldest_claimed_at"] == summary["newest_claimed_at"]
        assert executions.execution_history_summary(job_id="missing") == {
            "retained_count": 0, "oldest_claimed_at": None, "newest_claimed_at": None,
        }
    # Timestamp text sorts these in the opposite order from their actual instants.
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET claimed_at=?", ("2026-01-02T00:00:00+08:00",))
    monkeypatch.setattr(executions, "_hermes_now", lambda: datetime(2026, 1, 1, 20, tzinfo=timezone.utc))
    executions.create_execution("later", source="builtin")
    summary = executions.execution_history_summary()
    assert summary["retained_count"] == 2
    assert summary["oldest_claimed_at"] == "2026-01-02T00:00:00+08:00"
    assert summary["newest_claimed_at"] == "2026-01-01T20:00:00+00:00"

    import sqlite3
    import pytest
    from hermes_cli.cron import cron_runs
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "broken.db")
    executions.EXECUTIONS_FILE.write_bytes(b"not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        cron_runs()
