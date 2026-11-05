"""NULL-recency session rows must not 500 the dashboard's session listings (#134033).

The #91536 recency SQL yields ``last_active = NULL`` — key present, value None — when
every timestamp cell of a salvaged row (the started_at fallback included) is outside
the ``coerce_epoch`` window. The ``is_active`` call sites used
``row.get("last_active", <started_at fallback>)``, which only substitutes on a
*missing* key, so a None value reached ``now - None`` and raised TypeError: one such
row 500ed ``GET /api/sessions`` on every dashboard poll (~900 identical tracebacks a
day in the report). A row with no trusted timestamp counts as not active.
"""

import sqlite3

import pytest


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def _plant_null_recency_session(session_id: str, source: str = "cli") -> None:
    """A session whose only timestamp is out-of-window garbage (1e300): the recency
    SQL resolves its ``last_active`` to NULL — the #134033 row shape."""
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    db_path = get_hermes_home() / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id, source=source)
    finally:
        db.close()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (1e300, session_id))
        conn.commit()
    finally:
        conn.close()


def test_is_recently_active_treats_null_recency_as_not_active():
    from hermes_cli.web_routers._common import is_recently_active

    now = 1_000_000.0
    # The salvaged shape: no trusted cell at all, garbage started_at must not be used.
    assert is_recently_active({"ended_at": None, "last_active": None, "started_at": 1e300}, now) is False
    assert is_recently_active({"ended_at": None, "last_active": now - 60}, now) is True
    assert is_recently_active({"ended_at": now - 10, "last_active": now - 5}, now) is False
    # Rows whose source never selected the column keep the old not-active answer.
    assert is_recently_active({"ended_at": None}, now) is False


def test_get_sessions_null_recency_row_is_inactive_not_500(client):
    _plant_null_recency_session("salvaged-garbage-recency")

    response = client.get("/api/sessions?limit=50&offset=0")

    assert response.status_code == 200
    row = next(s for s in response.json()["sessions"] if s["id"] == "salvaged-garbage-recency")
    assert row["is_active"] is False


def test_profiles_sessions_null_recency_row_is_inactive_not_500(client):
    _plant_null_recency_session("salvaged-garbage-recency")

    response = client.get("/api/profiles/sessions?limit=50&offset=0")

    assert response.status_code == 200
    row = next(s for s in response.json()["sessions"] if s["id"] == "salvaged-garbage-recency")
    assert row["is_active"] is False


def test_status_active_session_count_survives_null_recency(client):
    from hermes_cli.web_routers.status import _count_status_active_sessions

    _plant_null_recency_session("salvaged-garbage-recency")

    assert _count_status_active_sessions() == 0


def test_cron_runs_null_recency_row_is_inactive(monkeypatch, _isolate_hermes_home):
    """Run rows go through the same heuristic inline in the runs loop."""
    import hermes_cli.web_routers.cron as _rt_cron

    class _FakeDB:
        def list_cron_job_runs(self, canonical, limit=20, offset=0):
            return [{
                "id": "cron_nullrec_20260726_090046", "source": "cron",
                "started_at": 1e300, "last_active": None, "ended_at": None, "archived": False,
            }]

        def close(self):
            pass

    monkeypatch.setattr(_rt_cron, "_open_session_db_for_profile", lambda _p, *, read_only: _FakeDB())
    monkeypatch.setattr(_rt_cron, "_job_owner_profile", lambda _j, _p: "default")
    monkeypatch.setattr(
        _rt_cron, "_call_cron_for_profile",
        lambda _p, cmd, *_a, **_k: {"id": "nullrec"} if cmd == "get_job" else None,
    )

    result = _rt_cron._list_cron_job_runs_sync("nullrec", limit=10)

    assert result["runs"][0]["is_active"] is False
