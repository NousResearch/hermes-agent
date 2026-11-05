"""A structurally corrupt state.db is published once and read the same way everywhere (#72046).

Before this, a damaged store showed up as an empty Desktop sidebar (200 + an ``errors`` row),
a 500 from ``/api/sessions``, ``components.storage: ok`` on ``/api/status`` and green
readiness. Each surface guessed; none said the store was damaged, so it read as deleted
history. ``persistence.sqlite._state.health`` is now the one latch: readers, writers and the readiness
probe publish into it, and every surface below reads it.
"""

import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._state.errors import StateDbCorruptError, classify_persistence_error
from persistence.sqlite._state.health import (
    is_structural_corruption_error,
    note_storage_error,
    reset_storage_state,
    storage_state,
)
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, list_sessions_rich


@pytest.fixture(autouse=True)
def _fresh_latch():
    reset_storage_state()
    yield
    reset_storage_state()


def _seed(db_path, count=40):
    db = StoreHolder(db_path)
    try:
        for i in range(count):
            sid = f"s{i:03d}"
            create_session(db, sid, "cli")
            append_message(db, session_id=sid, role="user", content="hello " * 50)
    finally:
        db.close()


def _corrupt_sessions_btree(db_path):
    """Overwrite the root page of the ``sessions`` table: real B-tree damage, not a mock."""
    conn = sqlite3.connect(db_path)
    try:
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        root = conn.execute("SELECT rootpage FROM sqlite_master WHERE name = 'sessions'").fetchone()[0]
    finally:
        conn.close()
    with open(db_path, "r+b") as fh:
        fh.seek(page_size * (root - 1))
        fh.write(b"\xde\xad\xbe\xef" * (page_size // 4))


# ── The latch itself ───────────────────────────────────────────────────────


class TestLatch:
    def test_only_structural_corruption_latches(self, tmp_path):
        path = tmp_path / "state.db"
        assert note_storage_error(path, sqlite3.OperationalError("database is locked")) is False
        assert note_storage_error(path, sqlite3.DatabaseError("malformed database schema (messages_fts)")) is False
        fts = sqlite3.DatabaseError("database disk image is malformed")
        fts.sqlite_errorcode = 267  # SQLITE_CORRUPT_VTAB: FTS-scoped, has its own fail-open path
        assert note_storage_error(path, fts) is False
        assert storage_state(path) == "ok"

        assert note_storage_error(path, sqlite3.DatabaseError("database disk image is malformed")) is True
        assert storage_state(path) == "corrupt"

    def test_not_a_database_is_structural(self):
        assert is_structural_corruption_error(sqlite3.DatabaseError("file is not a database"))

    def test_a_real_read_on_a_damaged_btree_latches(self, tmp_path):
        path = tmp_path / "state.db"
        _seed(path)
        _corrupt_sessions_btree(path)
        db = StoreHolder(path, read_only=True)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                list_sessions_rich(db, limit=50, offset=0)
        finally:
            db.close()
        assert storage_state(path) == "corrupt"


class TestPeerWritesAfterLatch:
    """A second handle in the same process must not keep writing to a file another handle
    already found damaged, and its refusal must be the SAME terminal error the gateway and
    agent flush already divert on (not a new class they would retry)."""

    def test_fresh_handle_refuses_writes_with_state_db_corrupt_error(self, tmp_path):
        path = tmp_path / "state.db"
        _seed(path, count=1)
        note_storage_error(path, sqlite3.DatabaseError("database disk image is malformed"))

        peer = StoreHolder(path)
        try:
            with pytest.raises(StateDbCorruptError) as excinfo:
                append_message(peer, session_id="s000", role="user", content="after the latch")
            assert classify_persistence_error(excinfo.value) == "corrupt"
            assert peer._db_corrupt is True
        finally:
            peer.close()

        conn = sqlite3.connect(path)
        try:
            contents = [row[0] for row in conn.execute("SELECT content FROM messages")]
        finally:
            conn.close()
        assert "after the latch" not in contents

    def test_quarantine_on_one_handle_publishes_the_latch(self, tmp_path):
        path = tmp_path / "state.db"
        db = StoreHolder(path)
        real = db._conn

        class _Malformed:
            def execute(self, *a, **k):
                raise sqlite3.DatabaseError("database disk image is malformed")

            def __getattr__(self, name):
                return getattr(real, name)

        try:
            create_session(db, "s1", "cli")
            db._conn = _Malformed()
            with pytest.raises(StateDbCorruptError):
                create_session(db, "s2", "cli")
        finally:
            db._conn = real
            db.close()
        assert storage_state(path) == "corrupt"


# ── Readiness ──────────────────────────────────────────────────────────────


def test_readiness_state_db_and_session_store_agree_on_corrupt(_isolate_hermes_home):
    from gateway.readiness import collect_runtime_readiness
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "state.db"
    _seed(path, count=1)
    healthy = collect_runtime_readiness(configured_model="m", runtime_status={"session_store": {"status": "ok"}})
    assert healthy["checks"]["state_db"]["status"] == "ok"

    note_storage_error(path, sqlite3.DatabaseError("database disk image is malformed"))
    ready = collect_runtime_readiness(configured_model="m", runtime_status={"session_store": {"status": "ok"}})
    assert ready["status"] == "degraded"
    assert ready["checks"]["state_db"] == {"status": "degraded", "detail": "corrupt"}
    # The gateway's handle cache says "ok"; an open handle on a damaged file is not a working store.
    assert ready["checks"]["session_store"] == {"status": "unavailable", "detail": "corrupt"}


def test_readiness_probe_latches_a_file_that_is_not_a_database(_isolate_hermes_home):
    from gateway.readiness import _probe_state_db
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    (home / "state.db").write_bytes(b"this is not sqlite" * 512)
    assert _probe_state_db(home) == {"status": "degraded", "detail": "corrupt"}
    assert storage_state(home / "state.db") == "corrupt"


# ── Desktop-facing HTTP surfaces, end to end on a real damaged file ─────────


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_routers import profiles as profiles_routes
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(profiles_routes, "_SIDEBAR_CACHE_TTL_SECONDS", 0.0)
    (get_hermes_home() / "config.yaml").write_text("{}\n", encoding="utf-8")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_corrupt_state_db_is_published_to_every_desktop_surface(client):
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "state.db"
    _seed(path)

    healthy = client.get("/api/profiles/sessions/sidebar?recents_profile=all").json()
    assert len(healthy["recents"]["sessions"]) > 0
    assert healthy["storage"] == {}
    assert client.get("/api/sessions").json()["storage"] == {}
    assert client.get("/api/status").json()["components"]["storage"] == {"status": "ok"}

    _corrupt_sessions_btree(path)

    # The sidebar used to be the only surface that noticed, as an empty 200 plus an errors row.
    sidebar = client.get("/api/profiles/sessions/sidebar?recents_profile=all")
    assert sidebar.status_code == 200
    body = sidebar.json()
    assert body["recents"]["sessions"] == []
    assert body["storage"] == {"default": "corrupt"}

    # The flat list says "unavailable", not "internal server error".
    flat = client.get("/api/sessions")
    assert flat.status_code == 503
    assert flat.json()["detail"]["error"] == "state_db_corrupt"

    # Status (what Desktop polls) and readiness now say the same thing.
    status = client.get("/api/status").json()
    assert status["components"]["storage"] == {"status": "degraded", "reason": "corrupt"}
    assert status["overall"] == "degraded"
    assert client.get("/api/profiles/sessions?profile=all").json()["storage"] == {"default": "corrupt"}
