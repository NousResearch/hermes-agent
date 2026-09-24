"""A malformed state.db must not turn dashboard analytics polling into a traceback storm (#96591)."""
import logging
import re
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.web_routers import _common, analytics
from hermes_state import SessionDB


def _malformed_state_db(home: Path) -> Path:
    db_path = home / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="cli", model="m")
    db.close()
    for side in ("-wal", "-shm"):
        Path(str(db_path) + side).unlink(missing_ok=True)
    with open(db_path, "r+b") as f:
        f.seek(100)
        f.write(b"\xff" * (4096 - 100))
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(str(db_path)).execute("SELECT count(*) FROM sqlite_master")
    return db_path


def test_corrupt_store_polls_return_status_and_warn_once_per_interval(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_state
    db_path = _malformed_state_db(tmp_path)
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)
    monkeypatch.setattr(_common, "_corrupt_store_warned_at", {})
    app = FastAPI()
    app.include_router(analytics.router)
    client = TestClient(app)

    with caplog.at_level(logging.DEBUG, logger="hermes_cli.web_server"):
        first = client.get("/api/analytics/usage?days=7")
        second = client.get("/api/analytics/usage?days=7")
        third = client.get("/api/analytics/models?days=7")
    for resp in (first, second, third):
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "state_db_corrupt"
    # Only the dashboard's own warning counts: hermes_state logs an unrelated
    # once-per-process SQLite-version advisory on some interpreters (CI's 3.50.4).
    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and r.name.startswith("hermes_cli.web_server")
    ]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert not any(r.exc_info for r in caplog.records), "no tracebacks for a known corrupt store"
    assert db_path.exists() and db_path.stat().st_size > 0, "dashboard must never quarantine the file"

    # The gate re-arms once the interval has elapsed (aged, not slept).
    _common._corrupt_store_warned_at[str(db_path)] -= _common._CORRUPT_STORE_WARN_INTERVAL_S + 1
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        assert client.get("/api/analytics/usage?days=7").status_code == 503
    assert sum(
        r.levelno >= logging.WARNING and r.name.startswith("hermes_cli.web_server")
        for r in caplog.records
    ) == 1


def test_corrupt_store_as_status_maps_replaced_store_errors_to_503_without_fix_nudge(tmp_path, monkeypatch):
    """A retired WAL generation / replaced state.db (RuntimeError subclasses, not sqlite3 errors)
    must come back as a structured 503 like the corrupt case, and the guidance must never tell the
    user to run `doctor --fix` while a holder is live (#110054). Busy/locked still propagates."""
    from fastapi import HTTPException
    from hermes_state_errors import DeletedWalGenerationError, StateDbReplacedError

    monkeypatch.setattr(_common, "_corrupt_store_warned_at", {})
    db_path = tmp_path / "state.db"
    for exc_cls, code in ((DeletedWalGenerationError, "state_db_deleted_wal"), (StateDbReplacedError, "state_db_replaced")):
        with pytest.raises(HTTPException) as info:
            with _common.corrupt_store_as_status(db_path):
                raise exc_cls("retired WAL held by pid 4242")
        assert info.value.status_code == 503
        assert info.value.detail["error"] == code
        assert info.value.detail["path"] == str(db_path)
        msg = info.value.detail["message"]
        assert "run `hermes doctor`" in msg
        # `--fix` may only appear negated — never as the action to take while a holder is live.
        negated = re.findall(r"(?i)(?:do not|don't|never) run `hermes doctor --fix`", msg)
        assert len(negated) == msg.count("`hermes doctor --fix`"), msg

    with pytest.raises(sqlite3.OperationalError):
        with _common.corrupt_store_as_status(db_path):
            raise sqlite3.OperationalError("database is locked")

    # Sibling route: GET /api/sessions and the session-id resolver used to let the same
    # RuntimeError family fall through to a generic 500.
    from hermes_cli.web_routers import sessions

    class _RetiredDb:
        db_path = str(tmp_path / "state.db")

        def resolve_session_id(self, session_id):
            raise DeletedWalGenerationError("retired WAL held by pid 4242")

        def list_sessions_rich(self, **kwargs):
            raise DeletedWalGenerationError("retired WAL held by pid 4242")

        def close(self):
            pass

    with pytest.raises(HTTPException) as info:
        sessions._resolve_session_id(_RetiredDb(), "abc")
    assert (info.value.status_code, info.value.detail["error"]) == (503, "state_db_deleted_wal")

    monkeypatch.setattr(sessions, "_maybe_auto_archive_for_profile", lambda profile: None)
    monkeypatch.setattr(sessions, "_session_db_path_for_profile", lambda profile: db_path)
    monkeypatch.setattr(sessions, "_open_session_db_for_profile", lambda profile, read_only: _RetiredDb())
    app = FastAPI()
    app.include_router(sessions.list_router)
    resp = TestClient(app, raise_server_exceptions=False).get("/api/sessions")
    assert (resp.status_code, resp.json()["detail"]["error"]) == (503, "state_db_deleted_wal")
