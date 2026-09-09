"""Legacy NULL-profile session owner backfill (#94724).

Pre-#95407 session rows carry ``profile_name = NULL``. Once a Desktop has
registry topology (≥2 registered connections) the fail-closed owner ladder
can no longer route those rows, making every pre-campaign session
unresumable. POST /api/sessions/owner-backfill stamps each store's own
serving-profile identity onto its legacy rows — single-match by construction
(a profile's state.db belongs to exactly one profile), idempotent, and never
overwriting a non-NULL owner.
"""

import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _seed(db_path, rows):
    """Insert bare session rows the way a pre-ownership install left them."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=db_path)
    try:
        for session_id, profile_name in rows:
            db.create_session(session_id, source="cli", profile_name=profile_name)
            db.append_message(
                session_id, role="user", content=f"hello from {session_id}"
            )
        # create_session backfills nothing here, but be explicit: force the
        # legacy shape at the SQL level so the fixture cannot silently depend
        # on create_session's own COALESCE behavior.
        for session_id, profile_name in rows:
            if profile_name is None:
                db._conn.execute(
                    "UPDATE sessions SET profile_name = NULL WHERE id = ?",
                    (session_id,),
                )
        db._conn.commit()
    finally:
        db.close()


def _profiles(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return dict(conn.execute("SELECT id, profile_name FROM sessions").fetchall())
    finally:
        conn.close()


def test_backfill_stamps_only_null_rows_and_is_idempotent(client):
    from hermes_constants import get_hermes_home

    db_path = get_hermes_home() / "state.db"
    _seed(
        db_path,
        [
            ("legacy-null-1", None),
            ("legacy-null-2", None),
            ("owned-other", "researcher"),
        ],
    )

    resp = client.post("/api/sessions/owner-backfill", json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    # Exactly the two legacy rows were stamped; the count is reported so the
    # caller can log it.
    assert body["stamped"] == 2
    assert body["profile"] == "default"

    stamped = _profiles(db_path)
    assert stamped["legacy-null-1"] == "default"
    assert stamped["legacy-null-2"] == "default"
    # Fail-closed contract: a non-NULL owner is NEVER overwritten, even when
    # it names a different profile than the serving store.
    assert stamped["owned-other"] == "researcher"

    # One-shot-per-row: a second run finds nothing left to stamp and the rows
    # are byte-identical.
    resp2 = client.post("/api/sessions/owner-backfill", json={})
    assert resp2.status_code == 200
    assert resp2.json()["stamped"] == 0
    assert _profiles(db_path) == stamped


def test_backfilled_rows_circulate_owned_on_the_list_endpoint(client):
    """After the backfill, the durable stamp (not just the per-response
    serving-profile decoration) owns the rows: the raw DB column is non-NULL,
    which is what survives into any other consumer of state.db."""
    from hermes_constants import get_hermes_home

    db_path = get_hermes_home() / "state.db"
    _seed(db_path, [("legacy-null-3", None)])

    assert _profiles(db_path)["legacy-null-3"] is None

    resp = client.post("/api/sessions/owner-backfill", json={})
    assert resp.status_code == 200
    assert resp.json()["stamped"] == 1

    listed = client.get("/api/sessions?limit=50&offset=0").json()["sessions"]
    row = next(s for s in listed if s["id"] == "legacy-null-3")
    assert row["profile"] == "default"
    assert _profiles(db_path)["legacy-null-3"] == "default"


def test_backfill_treats_empty_string_profile_as_legacy(client):
    """TRIM('') rows are the same stranded class as NULL — stamp them too."""
    from hermes_constants import get_hermes_home

    db_path = get_hermes_home() / "state.db"
    _seed(db_path, [("legacy-empty", None)])

    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE sessions SET profile_name = '  ' WHERE id = 'legacy-empty'")
    conn.commit()
    conn.close()

    resp = client.post("/api/sessions/owner-backfill", json={})
    assert resp.status_code == 200
    assert resp.json()["stamped"] == 1
    assert _profiles(get_hermes_home() / "state.db")["legacy-empty"] == "default"
