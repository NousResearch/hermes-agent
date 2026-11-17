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


# --- Split topology: a standalone gateway owns state.db (#148531) --------------------
#
# The Desktop calls this endpoint on every boot. A standalone gateway holds the profile's
# owner reservation for its whole lifetime, so the offline write path is refused for as long
# as it runs — `POST /api/sessions/owner-backfill` answered 409 on every boot even when the
# store had nothing left to stamp, and the client fails closed: no session could be resumed
# and no mutation went through. The owner runs this same idempotent stamp at bring-up
# (gateway/session_authority.py::backfill_legacy_session_owners), so an owned store must
# answer without reserving it.

@pytest.fixture
def split_home(tmp_path, monkeypatch):
    """A profile home of its own, with no runtime owning it."""
    monkeypatch.setattr("hermes_state._default_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def owner_held(split_home):
    """Hold the real profile reservation the way a running gateway does."""
    from gateway.runtime_ownership import ProfileOwnership

    owner = ProfileOwnership()
    owner.reserve([split_home])
    try:
        yield split_home
    finally:
        owner.close()


def _split_client(app):
    import httpx

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")


def _app():
    from fastapi import FastAPI
    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    return app


def _seed_legacy_null(db_path):
    from hermes_state import SessionDB

    with SessionDB(db_path=db_path) as db:
        db.create_session("legacy-null", source="cli", profile_name="default")
        db._conn.execute("UPDATE sessions SET profile_name = NULL WHERE id = 'legacy-null'")
        db._conn.commit()


@pytest.mark.asyncio
async def test_owner_backfill_answers_without_reserving_when_nothing_is_pending(owner_held):
    """The boot path: nothing to stamp, so it must not ask for the writer's lock at all."""
    from hermes_state import SessionDB

    with SessionDB(db_path=owner_held / "state.db") as db:
        db.create_session("owned", source="cli", profile_name="default")

    async with _split_client(_app()) as client:
        resp = await client.post("/api/sessions/owner-backfill", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "stamped": 0, "profile": "default"}


class _FakeOwnerSocket:
    """Stands in for the live gateway's control socket: the owner stamps its own store."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def rpc(self, verb, **params):
        self.calls.append((verb, params))
        from hermes_state import SessionDB

        with SessionDB(db_path=self.db_path) as db:
            stamp = db._own_profile_name()
            return {"stamped": db.backfill_null_session_profiles(stamp), "profile": stamp}


@pytest.mark.asyncio
async def test_owner_backfill_stamps_pending_rows_through_the_live_owner(owner_held, monkeypatch):
    """Pending rows + a discoverable owner: the owner stamps them, not a bare "deferred" 200.

    Answering ``stamped: 0`` while the rows stay NULL is what strands them: the Desktop marks the
    scope attempted before the POST and only retries on a transport failure, so a false success
    fails those rows closed for good.
    """
    from types import SimpleNamespace

    _seed_legacy_null(owner_held / "state.db")
    socket = _FakeOwnerSocket(owner_held / "state.db")
    monkeypatch.setattr(
        "hermes_cli.gateway_runtime.discover_gateway_endpoint",
        lambda home, *, timeout: SimpleNamespace(
            endpoint=SimpleNamespace(home=str(home)), state="live", reason_code=None))
    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway",
                        lambda endpoint=None, subject=None: socket)

    async with _split_client(_app()) as client:
        resp = await client.post("/api/sessions/owner-backfill", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "stamped": 1, "profile": "default"}
    assert socket.calls == [("session.owner_backfill", {})]
    # The owner's write is the durable one the fail-closed ladder reads.
    assert _profiles(owner_held / "state.db")["legacy-null"] == "default"


@pytest.mark.asyncio
async def test_owner_backfill_still_writes_when_no_owner_holds_the_profile(split_home):
    """An unowned store keeps the offline write path (the pre-split-topology behaviour)."""
    _seed_legacy_null(split_home / "state.db")

    async with _split_client(_app()) as client:
        resp = await client.post("/api/sessions/owner-backfill", json={})
    assert resp.status_code == 200, resp.text
    assert resp.json()["stamped"] == 1
    assert _profiles(split_home / "state.db")["legacy-null"] == "default"
