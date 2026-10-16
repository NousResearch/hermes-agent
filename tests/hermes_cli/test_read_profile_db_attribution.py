"""_read_profile_db must attribute failures to the database that actually failed (#134865).

The cross-profile fan-out opens the profile's state.db read-only, then runs a callback
that also reads sibling stores under the home scope (projects.db, repo scans). The old
single try/except called note_storage_error(state.db, ...) for callback failures too, so a
physically damaged projects.db latched healthy session storage as corrupt for the life of
the process — Desktop then refused every session operation with "Session database is
damaged" while state.db was verifiably healthy.

Only the open is attributed now. Structural state.db corruption raised inside the callback
is still latched, by SessionDB's own read chokepoint (_read_ctx notes it).
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state_health import reset_storage_state, storage_state


@pytest.fixture(autouse=True)
def _fresh_latch():
    reset_storage_state()
    yield
    reset_storage_state()


def _seed_store(home, count=3):
    from hermes_state import SessionDB

    home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    try:
        for i in range(count):
            db.create_session(f"s{i}", source="cli")
    finally:
        db.close()
    return home / "state.db"


def _damage_sessions_btree(db_path):
    """Real B-tree damage on the ``sessions`` root page: the read-only open's schema
    probe only PREPAREs ``LIMIT 0`` statements (no row access), so the open succeeds and
    the first row read of ``sessions`` is what fails."""
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


def test_sibling_projects_db_failure_does_not_latch_state_db(tmp_path):
    from hermes_cli.web_routers.profiles import _read_profile_db

    home = tmp_path / "profiles" / "code"
    state_db = _seed_store(home)
    (home / "projects.db").write_bytes(b"SQLit\x17\x03\x03 not a database")

    def _read(db):
        conn = sqlite3.connect(home / "projects.db")
        try:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        finally:
            conn.close()

    errors = []
    assert _read_profile_db("code", home, errors, _read) is None
    assert [e["profile"] for e in errors] == ["code"]
    assert storage_state(state_db) == "ok"

    # Healthy session writes/chat stay allowed: a fresh handle appends without refusal.
    from hermes_state import SessionDB

    peer = SessionDB(db_path=state_db)
    try:
        peer.append_message(session_id="s0", role="user", content="still writable")
    finally:
        peer.close()


def test_corrupt_state_db_on_open_latches(tmp_path):
    from hermes_cli.web_routers.profiles import _read_profile_db

    home = tmp_path / "profiles" / "code"
    state_db = _seed_store(home)
    state_db.write_bytes(b"SQLit\x17\x03\x03\x00\x13 tls-shaped garbage, not a store")

    errors = []
    result = _read_profile_db("code", home, errors, lambda db: "unreachable")
    assert result is None
    assert errors
    assert storage_state(state_db) == "corrupt"


def test_corrupt_state_db_during_callback_reads_latches(tmp_path):
    from hermes_cli.web_routers.profiles import _read_profile_db

    home = tmp_path / "profiles" / "code"
    state_db = _seed_store(home)
    _damage_sessions_btree(state_db)

    errors = []
    assert _read_profile_db("code", home, errors, lambda db: db.list_sessions_rich(limit=5, offset=0)) is None
    assert storage_state(state_db) == "corrupt"


def test_direct_read_ctx_damage_during_callback_latches(tmp_path):
    """Callback reads that bypass the ``_read_*`` helpers must publish the same latch:
    the _read_ctx chokepoint notes structural corruption off its own connection."""
    from hermes_cli.web_routers.profiles import _read_profile_db

    home = tmp_path / "profiles" / "code"
    state_db = _seed_store(home)
    _damage_sessions_btree(state_db)

    def _read(db):
        with db._read_ctx() as conn:
            conn.execute("SELECT * FROM sessions LIMIT 5").fetchall()

    errors = []
    assert _read_profile_db("code", home, errors, _read) is None
    assert storage_state(state_db) == "corrupt"


def test_ordinary_callback_error_is_reported_without_latching(tmp_path):
    from hermes_cli.web_routers.profiles import _read_profile_db

    home = tmp_path / "profiles" / "code"
    state_db = _seed_store(home)

    def _read(db):
        raise ValueError("repo scan exploded")

    errors = []
    assert _read_profile_db("code", home, errors, _read) is None
    assert [e["error"] for e in errors] == ["repo scan exploded"]
    assert storage_state(state_db) == "ok"
