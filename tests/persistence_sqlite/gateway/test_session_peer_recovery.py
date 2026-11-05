"""Gateway peer recovery and routing in the SQLite backend: the routing repository's ``latest_session_for_peer`` round trip, its reset-boundary and own-profile fences, child profile inheritance within a key namespace, ``session_for_origin`` matching, and the rebuild of a legacy ``gateway_routing`` primary key on open."""

import sqlite3

import pytest

from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite._schema import _heal_gateway_routing_pk
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session
from tests.persistence_sqlite.conftest import store_for


@pytest.fixture
def store(tmp_path):
    """A backend store outside the profile tree, so it has no owning profile."""
    scope = ProfileScope(profile_dir=tmp_path, name="test")
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / "test_state.db"})
    try:
        yield backend.open(scope)
    finally:
        backend.close()


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


def test_gateway_session_peer_round_trip_and_recovery(db, store):
    create_session(db,
        "gw-session",
        "telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
        thread_id=None,
    )
    append_message(db, "gw-session", "user", "hello")

    row = get_session(db, "gw-session")
    assert row["session_key"] == "agent:main:telegram:dm:chat-1"
    assert row["chat_id"] == "chat-1"
    assert row["chat_type"] == "dm"

    recovered = store.routing.latest_session_for_peer(
        source="telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
    )
    assert recovered.session.session_id == "gw-session"


@pytest.mark.parametrize(
    "persisted_session_key",
    ["agent:main:telegram:dm:chat-1", None],
    ids=["exact-key", "peer-fallback"],
)
def test_gateway_session_recovery_does_not_cross_newer_reset_boundary(
    db, store, persisted_session_key
):
    """A newer session_reset row fences recovery for the peer (#68539).

    Recovery must never reach *behind* an intentional /new boundary and
    resurrect an older still-open row — if the newest boundary row for the
    peer is reset-ended, recovery returns nothing.
    """
    peer = {
        "user_id": "user-1",
        "session_key": persisted_session_key,
        "chat_id": "chat-1",
        "chat_type": "dm",
    }
    create_session(db, "gw-before-reset", "telegram", **peer)
    append_message(db, "gw-before-reset", "user", "old context")
    create_session(db, "gw-reset", "telegram", **peer)
    append_message(db, "gw-reset", "user", "/new")
    end_session(db, "gw-reset", "session_reset")

    assert store.routing.latest_session_for_peer(
        source="telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
    ) is None


def test_peer_fallback_never_adopts_a_sibling_profiles_row(tmp_path, monkeypatch):
    """#74285: the peer-tuple fallback is fenced by the store's own profile.

    A Telegram DM peer tuple (chat_id == user_id, no thread) is identical for
    every bot, so a legacy sibling-profile row sitting in this store — written
    before the per-profile partition — must lose to the older own row, and
    with no own row recovery must return nothing rather than the sibling's.
    """
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    scope = ProfileScope(profile_dir=root, name="default")
    backend = SQLiteBackend({scope.hermes_home_key: root / "state.db"})  # owner: default
    try:
        routing = backend.open(scope).routing
        store = backend.open(scope).holder()
        peer = {"user_id": "42", "chat_id": "42", "chat_type": "dm"}
        create_session(store, "sibling", "telegram", session_key="agent:bot2:telegram:dm:42",
                             profile_name="bot2", **peer)
        append_message(store, "sibling", "user", "bot2's conversation")

        def recover():
            return routing.latest_session_for_peer(
                source="telegram", session_key="agent:main:telegram:dm:42", **peer
            )

        assert recover() is None  # only the sibling exists: fail closed

        create_session(store, "own", "telegram", session_key="agent:main:telegram:dm:42:old", **peer)
        append_message(store, "own", "user", "default's conversation")
        store._execute_write(
            lambda c: c.execute("UPDATE sessions SET last_activity_at = 1 WHERE id = 'own'")
        )
        assert recover().session.session_id == "own"  # older own row beats newer sibling row
    finally:
        backend.close()


def test_peer_fallback_reset_boundary_is_profile_fenced(tmp_path, monkeypatch):
    """#119121: the recovery reset fence carries the same profile predicate as the candidate.

    A Telegram DM peer tuple is identical for every bot, so a sibling profile's
    newer session_reset row used to suppress THIS profile's recoverable session.
    The profile's own reset must still fence.
    """
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    scope = ProfileScope(profile_dir=root, name="default")
    backend = SQLiteBackend({scope.hermes_home_key: root / "state.db"})  # owner: default
    try:
        routing = backend.open(scope).routing
        store = backend.open(scope).holder()
        peer = {"user_id": "42", "chat_id": "42", "chat_type": "dm"}

        def recover():
            return routing.latest_session_for_peer(
                source="telegram", session_key="agent:main:telegram:dm:42", **peer
            )

        def reset(session_id, session_key, profile_name):
            create_session(store, session_id, "telegram", session_key=session_key,
                                 profile_name=profile_name, **peer)
            append_message(store, session_id, "user", "/new")
            end_session(store, session_id, "session_reset")

        create_session(store, "own", "telegram", session_key="agent:main:telegram:dm:42:old",
                             profile_name="default", **peer)
        append_message(store, "own", "user", "default's conversation")
        store._execute_write(
            lambda c: c.execute("UPDATE sessions SET last_activity_at = 1 WHERE id = 'own'")
        )

        reset("sibling-reset", "agent:bot2:telegram:dm:42", "bot2")
        assert recover().session.session_id == "own"  # a sibling profile's reset is not this profile's boundary

        reset("own-reset", "agent:main:telegram:dm:42:r2", "default")
        assert recover() is None  # the profile's own reset still fences
    finally:
        backend.close()


def test_child_inherits_parent_profile_only_within_its_key_namespace(db):
    """#88381: parent→child ``profile_name`` COALESCE is fenced by ``agent:<ns>:``.

    A default child (``agent:main:``) forked from a sibling profile's row must
    not be durably mislabelled as that profile's; same-namespace and keyless
    (CLI/subagent) children keep inheriting.
    """
    create_session(db, "parent", "telegram", session_key="agent:bot2:telegram:dm:42",
                      profile_name="bot2")
    create_session(db, "cross", "telegram", parent_session_id="parent",
                      session_key="agent:main:telegram:dm:42")
    create_session(db, "same", "telegram", parent_session_id="parent",
                      session_key="agent:bot2:telegram:dm:42:r2")
    create_session(db, "keyless", "cli", parent_session_id="parent")
    assert get_session(db, "cross")["profile_name"] is None
    assert get_session(db, "same")["profile_name"] == "bot2"
    assert get_session(db, "keyless")["profile_name"] == "bot2"


def test_find_session_by_origin_matching_rules(db, store):
    create_session(db,
        "gw-o1", "telegram", user_id="u1",
        session_key="agent:main:telegram:group:c9:u1", chat_id="c9", chat_type="group",
    )
    create_session(db,
        "gw-o2", "telegram", user_id="u2",
        session_key="agent:main:telegram:group:c9:u2", chat_id="c9", chat_type="group",
    )

    # Exact user match wins.
    assert store.routing.session_for_origin(
        platform="telegram", chat_id="c9", user_id="u2"
    ) == "gw-o2"
    # Unknown user among multiple distinct users -> None (no contamination).
    assert store.routing.session_for_origin(
        platform="telegram", chat_id="c9", user_id="u3"
    ) is None
    # No user given + multiple distinct users -> None.
    assert store.routing.session_for_origin(platform="telegram", chat_id="c9") is None
    # Ended sessions are ignored: only gw-o1 remains as a live candidate.
    # A single remaining candidate is returned even without an exact user
    # match — mirrors the original sessions.json scan semantics.
    end_session(db, "gw-o2", "session_reset")
    assert store.routing.session_for_origin(
        platform="telegram", chat_id="c9", user_id="u2"
    ) == "gw-o1"
    # Single remaining candidate resolves without user_id.
    assert store.routing.session_for_origin(platform="telegram", chat_id="c9") == "gw-o1"
    # Thread filter.
    create_session(db,
        "gw-th", "discord", user_id="u9",
        session_key="agent:main:discord:thread:t7", chat_id="ch7",
        chat_type="thread", thread_id="t7",
    )
    assert store.routing.session_for_origin(
        platform="discord", chat_id="ch7", thread_id="t7"
    ) == "gw-th"
    assert store.routing.session_for_origin(
        platform="discord", chat_id="ch7", thread_id="other"
    ) is None


class TestGatewayRoutingPkHeal:
    """Legacy gateway_routing tables (session_key-only PK) get rebuilt on open.

    Early builds of the #59203 routing-index migration created gateway_routing
    with ``session_key TEXT PRIMARY KEY`` and no ``scope`` column. The column
    reconciler ADDs ``scope`` but cannot change the PK, so on those databases
    every routing save failed ("ON CONFLICT clause does not match any PRIMARY
    KEY or UNIQUE constraint" / "UNIQUE constraint failed:
    gateway_routing.session_key") and spammed warnings on each save.
    """

    LEGACY_SQL = """
        CREATE TABLE gateway_routing (
            session_key TEXT PRIMARY KEY,
            entry_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        , "scope" TEXT DEFAULT '')
    """

    def _make_legacy_db(self, db_path, rows=()):
        conn = sqlite3.connect(db_path)
        conn.execute(self.LEGACY_SQL)
        conn.executemany(
            "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
            "VALUES (?, ?, ?, ?)",
            list(rows),
        )
        conn.commit()
        conn.close()
        return db_path

    def _pk_cols(self, db):
        rows = db._conn.execute('PRAGMA table_info("gateway_routing")').fetchall()
        cols = sorted(
            ((r["pk"], r["name"]) for r in rows if r["pk"]),
        )
        return [name for _, name in cols]

    def test_legacy_pk_rebuilt_to_composite(self, backend, scope_a):
        self._make_legacy_db(
            store_for(scope_a), rows=[("/home/u/.hermes/sessions", "agent:main:telegram:dm:1", "{}", 1.0)]
        )
        store = backend.open(scope_a)
        db = store.holder()
        assert self._pk_cols(db) == ["scope", "session_key"]
        # Existing rows survive the rebuild.
        entries = store.routing.load_routing_entries(routing_scope="/home/u/.hermes/sessions")
        assert entries == {"agent:main:telegram:dm:1": "{}"}

    def test_current_shape_left_untouched(self, db, store):
        """A DB born with the composite PK is not rebuilt (idempotence)."""
        store.routing.upsert_routing_entry("k1", "{}", routing_scope="s")
        assert self._pk_cols(db) == ["scope", "session_key"]
        # Re-running the heal is a no-op.
        cur = db._conn.cursor()
        _heal_gateway_routing_pk(db, cur)
        assert store.routing.load_routing_entries(routing_scope="s") == {"k1": "{}"}
