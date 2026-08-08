from types import SimpleNamespace

import pytest

import hermes_state
from hermes_cli.sessions_cmd import cmd_sessions


class _CountConnection:
    def __init__(self, *, query_error: bool = False):
        self.query_error = query_error

    def execute(self, _sql):
        if self.query_error:
            raise RuntimeError("count failed")
        return self

    def fetchone(self):
        return (3,)


class _RecordingSessionDB:
    def __init__(self, *, query_error: bool = False, close_error: bool = False):
        self._conn = _CountConnection(query_error=query_error)
        self.close_calls = 0
        self.close_error = close_error

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("close failed")


def _run_repair(monkeypatch, tmp_path, db):
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"broken")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(hermes_state, "_db_opens_cleanly", lambda _path: "malformed")
    monkeypatch.setattr(
        hermes_state,
        "repair_state_db_schema",
        lambda _path, *, backup: {
            "repaired": True,
            "backup_path": None,
            "strategy": "test",
        },
    )
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)

    cmd_sessions(
        SimpleNamespace(
            sessions_action="repair",
            check_only=False,
            no_backup=False,
        )
    )


def test_repair_closes_count_database(monkeypatch, tmp_path, capsys):
    db = _RecordingSessionDB()

    _run_repair(monkeypatch, tmp_path, db)

    assert db.close_calls == 1
    assert "3 sessions recovered" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("query_error", "close_error"),
    [(True, False), (False, True)],
)
def test_repair_cleanup_failures_remain_best_effort(
    monkeypatch,
    tmp_path,
    capsys,
    query_error,
    close_error,
):
    db = _RecordingSessionDB(
        query_error=query_error,
        close_error=close_error,
    )

    _run_repair(monkeypatch, tmp_path, db)

    assert db.close_calls == 1
    assert "✓ Repaired" in capsys.readouterr().out
