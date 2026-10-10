"""Regression for #130962: transient dashboard slowness must not cascade.

Windows Desktop attaches to a standalone `hermes dashboard` on loopback and
treats a single 5s probe timeout as "backend gone", invalidating the slot and
superseding in-flight API callers. The server side of that contract is:

- GET /api/health stays lightweight (no SessionDB open) so heavy SQLite reads
  never stall the liveness probe itself.
- GET /api/sessions maps transient SQLite busyness to 503 with a retry hint,
  never 500 or an authoritative empty list that clears the desktop sidebar.
- The /api/status active-session garnish times out fast and returns 0 instead
  of holding the request open behind a locked store.
"""

import asyncio
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.web_routers import sessions as sessions_router
from hermes_cli.web_routers import status as status_router


def test_health_is_lightweight_and_needs_no_store(tmp_path, monkeypatch):
    """GET /api/health answers without opening state.db."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    opened = []

    import hermes_cli.web_server_sessions as web_server_sessions

    def _unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("health must not open the session store")

    monkeypatch.setattr(
        web_server_sessions, "_open_session_db_for_profile", _unexpected_open
    )

    app = FastAPI()
    app.include_router(status_router.router)
    client = TestClient(app)

    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert opened == []
    assert not (tmp_path / "state.db").exists()


def test_sessions_busy_store_returns_503_retry_not_empty(tmp_path, monkeypatch):
    """A locked/busy store is 503 'busy, retry', not 500 or an empty list."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _BusyDB:
        db_path = str(tmp_path / "state.db")

        def list_sessions_rich(self, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        def close(self):
            pass

    monkeypatch.setattr(
        sessions_router, "_maybe_auto_archive_for_profile", lambda profile: None
    )
    monkeypatch.setattr(
        sessions_router,
        "_open_session_db_for_profile",
        lambda profile, read_only: _BusyDB(),
    )

    app = FastAPI()
    app.include_router(sessions_router.list_router)
    resp = TestClient(app, raise_server_exceptions=False).get("/api/sessions")

    assert resp.status_code == 503
    assert "Retry" in resp.json()["detail"]
    # Must never read as an authoritative empty list downstream.
    assert resp.json()["detail"] != {"sessions": [], "total": 0}


def test_status_active_sessions_garnish_times_out_to_zero(monkeypatch):
    """The /api/status active-session count never holds the request open."""

    async def _never():
        await asyncio.sleep(10)

        return 5

    monkeypatch.setattr(
        status_router, "run_in_threadpool", lambda *args, **kwargs: _never()
    )

    result = asyncio.run(status_router._status_active_sessions())
    assert result == 0
