"""The state-store protocol contract and the clear_stored_system_prompts behavior.

``StateStoreInterface`` (state_store_interface.py) is the backend-neutral face
plugins may rely on. Two things are pinned here:

1. **Conformance** — the reference SQLite ``SessionDB`` satisfies the
   ``@runtime_checkable`` protocol structurally (no inheritance), and every
   contracted member exists on the class. This is the plugin-compatibility
   check plugin authors and dev tooling are told to run; it is NOT a runtime
   gate the store itself enforces.
2. **clear behavior** — ``clear_stored_system_prompts`` invalidates prompt
   snapshots in both storage layouts (out-of-line ``system_prompts`` table +
   hash references, and legacy inline ``sessions.system_prompt``), never
   deletes session rows, and is idempotent (a second run clears nothing).

The legacy inline layout is simulated the only way a real database can hold
it: a sessions table created without the ``system_prompt_hash`` column, so the
out-of-line branch is unreachable and the probe falls through to inline.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_maintenance import SessionMaintenanceMixin
from state_store_interface import StateStoreInterface


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    yield handle
    handle.close()


# ── Protocol conformance ────────────────────────────────────────────────────


def test_real_session_db_satisfies_the_protocol(db):
    # Structural typing: SessionDB inherits nothing from the protocol, and a
    # @runtime_checkable protocol with a data member (TITLE_SOURCE_LLM) supports
    # isinstance only — issubclass would raise TypeError. The isinstance pass IS
    # the plugin-compatibility check.
    assert isinstance(db, StateStoreInterface)


def test_protocol_face_members_exist_on_session_db():
    # Existence check for the contracted face (11 core members + the constant):
    # the original nine, plus the search-availability probes and the explicit
    # unit-of-work added with the error/availability/atomicity contracts.
    for member in (
        "get_session_title",
        "get_session_title_source",
        "set_session_title",
        "set_auto_title",
        "sanitize_title",
        "get_session",
        "get_messages_as_conversation",
        "search_messages",
        "is_search_available",
        "fts_health",
        "clear_stored_system_prompts",
        "apply_atomic",
        "TITLE_SOURCE_LLM",
    ):
        assert hasattr(SessionDB, member), member


def test_get_session_reads_created_row_and_missing_id(db):
    db.create_session("s1", source="cli", system_prompt="configured prompt")
    row = db.get_session("s1")
    assert isinstance(row, dict)
    assert row["id"] == "s1"
    assert row["source"] == "cli"
    assert row["system_prompt"] == "configured prompt"
    assert db.get_session("missing") is None


def test_get_messages_as_conversation_reads_appended_user_message(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="hello from the stored transcript")

    conversation = db.get_messages_as_conversation("s1")
    assert isinstance(conversation, list)
    assert len(conversation) == 1
    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"] == "hello from the stored transcript"
    assert db.get_messages_as_conversation("missing") == []


def test_get_session_title_source_reads_manual_and_automatic_provenance(db):
    for session_id in ("manual", "derived", "llm"):
        db.create_session(session_id, source="cli")
        assert db.get_session_title_source(session_id) is None

    assert db.set_session_title("manual", "Chosen by user")
    assert db.set_auto_title("derived", "Derived title", source=SessionDB.TITLE_SOURCE_DERIVED)
    assert db.set_auto_title("llm", "Generated title", source=SessionDB.TITLE_SOURCE_LLM)

    assert db.get_session_title_source("manual") == SessionDB.TITLE_SOURCE_USER
    assert db.get_session_title_source("derived") == SessionDB.TITLE_SOURCE_DERIVED
    assert db.get_session_title_source("llm") == SessionDB.TITLE_SOURCE_LLM
    assert db.get_session_title_source("missing") is None


def test_search_without_fts_reads_canonical_messages(db):
    db.create_session("s1", source="cli")
    hit_id = db.append_message("s1", role="user", content="distinctive fallback needle")
    db.append_message("s1", role="assistant", content="unrelated text")
    db._fts_enabled = False

    rows = db.search_messages("needle", source_filter=["cli"], role_filter=["user"],
                              sort="oldest", limit=1, offset=0, fields=["id", "role", "snippet"])
    assert len(rows) == 1
    assert rows[0]["id"] == hit_id
    assert rows[0]["role"] == "user"
    assert "needle" in rows[0]["snippet"].lower()
    assert set(rows[0]) == {"id", "role", "snippet"}
    assert db.search_messages("unfindable") == []


# ── search_messages availability probes (fail-open distinguishability) ──────


def test_search_availability_probe_reports_the_live_route(db):
    # The probe contract: available=True must mean the indexed route serves
    # the next call. On a healthy store FTS is enabled and reported live,
    # with the sub-indexes the router can pick listed as routes.
    assert db.is_search_available() is True
    health = db.fts_health()
    assert health["available"] is True
    assert health["stale"] is False
    assert health["reason"] is None
    assert health["routes"]["messages_fts"] is True


def test_search_availability_probe_flags_fts_unavailable(db):
    # Index down: search_messages still answers (fail-open [] or fallback),
    # but the probe reports False — the two states a bare [] cannot tell apart.
    db._fts_enabled = False
    assert db.is_search_available() is False
    health = db.fts_health()
    assert health["available"] is False
    assert health["reason"] == "fts_unavailable"
    assert health["routes"] == {}
    # The search call itself stays legal on the fallback route.
    assert db.search_messages("anything") == []


def test_search_availability_probe_flags_fts_stale(db):
    # Fail-open latched (corruption-detached or observed in shared state):
    # reported distinctly from plain unavailability.
    db._fts_stale = True
    assert db.is_search_available() is False
    health = db.fts_health()
    assert health["available"] is False
    assert health["stale"] is True
    assert health["reason"] == "fts_stale"
    db._fts_stale = False


def test_search_availability_probe_never_raises(db, monkeypatch):
    # The probe is the pre-call gate; it must never become a new failure
    # point. An internal probe error degrades to available=False instead of
    # propagating.
    def broken_status():
        raise sqlite3.DatabaseError("injected probe fault")

    monkeypatch.setattr(db, "fts_rebuild_status", broken_status)
    health = db.fts_health()
    assert health["available"] is False
    assert health["reason"] == "probe_error"
    assert db.is_search_available() is False


@pytest.mark.parametrize("route", ["fts", "disabled", "operational_error"])
def test_standalone_near_search_survives_fts_fallback(db, monkeypatch, route):
    db.create_session("s1", source="cli")
    hit_id = db.append_message("s1", role="user", content="standalone NEAR token")
    db.append_message("s1", role="assistant", content="unrelated text")
    assert db._fts_enabled
    faults = []

    if route == "disabled":
        db._fts_enabled = False
    elif route == "operational_error":
        real_read_all = db._read_all

        def read_with_fts_fault(sql, params):
            if "messages_fts MATCH ?" in sql:
                faults.append(params[0])
                raise sqlite3.OperationalError("injected FTS MATCH fault")
            return real_read_all(sql, params)

        monkeypatch.setattr(db, "_read_all", read_with_fts_fault)

    assert db.search_messages("NEAR", fields=["id"]) == [{"id": hit_id}]
    assert db.search_messages("trulyabsentword", fields=["id"]) == []
    if route == "operational_error":
        assert faults == ["NEAR", "trulyabsentword"]


def test_fts_match_operational_error_uses_canonical_read_and_surfaces_its_failure(db, monkeypatch):
    db.create_session("s1", source="cli")
    hit_id = db.append_message("s1", role="user", content="distinctive fallback needle")
    assert db.search_messages("needle", fields=["id"]) == [{"id": hit_id}]

    real_read_all = db._read_all
    fail_canonical = False

    def read_with_fts_fault(sql, params):
        if "messages_fts MATCH ?" in sql:
            raise sqlite3.OperationalError("injected FTS MATCH fault")
        if fail_canonical and "FROM messages m" in sql:
            raise sqlite3.OperationalError("injected canonical read failure")
        return real_read_all(sql, params)

    monkeypatch.setattr(db, "_read_all", read_with_fts_fault)
    assert db.search_messages("needle", fields=["id"]) == [{"id": hit_id}]
    fail_canonical = True
    with pytest.raises(sqlite3.OperationalError, match="injected canonical read failure"):
        db.search_messages("needle", fields=["id"])


def test_sanitize_title_is_the_contracted_normalizer(db):
    assert SessionDB.sanitize_title("  hello \x00world\t ") == "hello world"
    assert SessionDB.sanitize_title("   ") is None


def test_sanitize_title_truncate_is_the_non_raising_length_path(db):
    # The Optional[str] contract needs a path that cannot raise: truncate cuts
    # the normalized title to the cap (still storable), None stays
    # empty-input-only. The raise is kept for callers who must know their
    # title was over the cap instead of silently storing a different one.
    overlong = "x" * (SessionDB.MAX_TITLE_LENGTH + 50)
    assert SessionDB.sanitize_title(overlong, truncate=True) == "x" * SessionDB.MAX_TITLE_LENGTH
    assert SessionDB.sanitize_title("   ", truncate=True) is None
    assert SessionDB.sanitize_title("short", truncate=True) == "short"
    # A truncated value must itself satisfy the store: storing it succeeds.
    db.create_session("s1", source="cli")
    assert db.set_session_title("s1", SessionDB.sanitize_title(overlong, truncate=True))
    assert db.get_session_title("s1") == "x" * SessionDB.MAX_TITLE_LENGTH


def test_auto_title_rejects_invalid_source(db):
    # Contract: an invalid source is a programming error and stays a declared
    # ValueError (set_auto_title is in the contract's raising class); the
    # legitimate sources are the TITLE_SOURCE_* constants the same contract
    # names, so an out-of-set value cannot arrive by accident.
    db.create_session("s1", source="cli")
    with pytest.raises(ValueError, match="invalid automatic title source"):
        db.set_auto_title("s1", "Rejected", source=SessionDB.TITLE_SOURCE_USER)
    assert db.get_session_title("s1") is None


def test_title_length_is_checked_after_sanitizing(db):
    db.create_session("s1", source="cli")
    overlong = "  " + "x" * (SessionDB.MAX_TITLE_LENGTH + 1) + "\x00  "
    with pytest.raises(ValueError, match="Title too long"):
        db.set_auto_title("s1", overlong, source=SessionDB.TITLE_SOURCE_DERIVED)
    assert db.get_session_title("s1") is None


def test_auto_title_collision_with_existing_owner_raises(db):
    db.create_session("owner", source="cli")
    db.create_session("candidate", source="cli")
    assert db.set_session_title("owner", "Taken")
    with pytest.raises(ValueError, match="already in use"):
        db.set_auto_title("candidate", "  Taken  ", source=SessionDB.TITLE_SOURCE_LLM)
    assert db.get_session_title("owner") == "Taken"
    assert db.get_session_title("candidate") is None


def test_auto_title_false_preserves_higher_authority_and_missing_rows(db):
    db.create_session("s1", source="cli")
    assert db.set_session_title("s1", "Manual")
    assert db.set_auto_title("s1", "Generated", source=SessionDB.TITLE_SOURCE_LLM) is False
    assert db.get_session_title("s1") == "Manual"
    assert db.set_auto_title("missing", "Generated", source=SessionDB.TITLE_SOURCE_DERIVED) is False
    assert db.get_session("missing") is None


def test_hidden_canonical_bot_chat_title_is_protected(db):
    db.create_session("bot", source="desktop")
    db.create_session("candidate", source="cli")
    assert db.set_auto_title("bot", SessionDB.CANONICAL_BOT_CHAT_TITLE,
                             source=SessionDB.TITLE_SOURCE_DERIVED)
    assert db.set_session_hidden("bot", True)
    assert db.set_auto_title("bot", "New name", source=SessionDB.TITLE_SOURCE_LLM) is False
    with pytest.raises(ValueError, match="already in use"):
        db.set_auto_title("candidate", SessionDB.CANONICAL_BOT_CHAT_TITLE,
                          source=SessionDB.TITLE_SOURCE_LLM)
    assert db.get_session_title("bot") == SessionDB.CANONICAL_BOT_CHAT_TITLE
    assert db.get_session_title("candidate") is None


# ── clear_stored_system_prompts — out-of-line layout (default schema) ───────


def test_clear_out_of_line_nulls_references_and_deletes_snapshots(db):
    db.create_session("s1", source="cli", system_prompt="prompt one")
    db.create_session("s2", source="cli", system_prompt="prompt two")
    # The prompt store deduplicates: one snapshot row per distinct text.
    hashes_before = {
        row["hash"]
        for row in _rows(db, "SELECT hash FROM system_prompts")
    }
    assert len(hashes_before) == 2

    result = db.clear_stored_system_prompts()

    assert result == {"cleared": 2, "storage_mode": "out-of-line"}
    assert _rows(db, "SELECT id FROM sessions WHERE system_prompt_hash IS NOT NULL") == []
    assert _rows(db, "SELECT id FROM sessions WHERE system_prompt IS NOT NULL AND system_prompt != ''") == []
    # Snapshot rows are gone once nothing references them.
    assert _rows(db, "SELECT hash FROM system_prompts") == []
    # Session rows survive; the resolved prompt reads as unset.
    assert {row["id"] for row in _rows(db, "SELECT id FROM sessions")} == {"s1", "s2"}
    assert db.get_session("s1")["system_prompt"] is None


def test_clear_out_of_line_rolls_back_update_when_snapshot_sweep_fails(db, monkeypatch):
    db.create_session("s1", source="cli", system_prompt="prompt one")
    db.create_session("s2", source="cli", system_prompt="prompt two")
    sessions_sql = "SELECT id, system_prompt, system_prompt_hash FROM sessions ORDER BY id"
    snapshots_sql = "SELECT hash, prompt FROM system_prompts ORDER BY hash"
    sessions_before = _rows(db, sessions_sql)
    snapshots_before = _rows(db, snapshots_sql)
    assert len(snapshots_before) == 2
    assert all(row["system_prompt_hash"] is not None for row in sessions_before)

    def fail_after_update(conn):
        # The UPDATE happened in this transaction, before the snapshot sweep.
        assert all(
            row["system_prompt_hash"] is None and row["system_prompt"] is None
            for row in conn.execute(sessions_sql)
        )
        assert [dict(row) for row in conn.execute(snapshots_sql)] == snapshots_before
        raise RuntimeError("injected snapshot sweep failure")

    monkeypatch.setattr(db, "_delete_unreferenced_system_prompts", fail_after_update)
    with pytest.raises(RuntimeError, match="injected snapshot sweep failure"):
        db.clear_stored_system_prompts()

    assert _rows(db, sessions_sql) == sessions_before
    assert _rows(db, snapshots_sql) == snapshots_before
    assert db.get_session("s1")["system_prompt"] == "prompt one"
    assert db.get_session("s2")["system_prompt"] == "prompt two"


def test_clear_is_idempotent(db):
    db.create_session("s1", source="cli", system_prompt="prompt one")
    first = db.clear_stored_system_prompts()
    second = db.clear_stored_system_prompts()
    assert first["cleared"] == 1
    assert second == {"cleared": 0, "storage_mode": "out-of-line"}


def test_clear_on_an_empty_store_reports_zero(db):
    assert db.clear_stored_system_prompts() == {"cleared": 0, "storage_mode": "out-of-line"}


def test_clear_only_deletes_snapshots_no_longer_referenced(db):
    # clear_stored_system_prompts wipes every reference; re-creating one session
    # with a fresh prompt must leave the other snapshot absent (not resurrected)
    # and the new one present — the sweep is reference-driven, not truncate-all.
    db.create_session("s1", source="cli", system_prompt="prompt one")
    db.clear_stored_system_prompts()
    db.create_session("s2", source="cli", system_prompt="prompt two")
    assert _rows(db, "SELECT hash FROM system_prompts") != []
    assert db.get_session("s2")["system_prompt"] == "prompt two"


# ── clear_stored_system_prompts — legacy inline layout ──────────────────────


def test_clear_inline_layout(tmp_path):
    # A real pre-hash legacy database has sessions.system_prompt text and no
    # system_prompt_hash column. SessionDB's own open reconciles that column in
    # (see test_clear_upgrades_legacy_inline_text below), so the inline branch
    # of the probe is exercised by driving the mixin against the raw layout —
    # the mixin's only dependency is _execute_write, which is provided as a
    # plain transactional connection, matching how the original plugin code
    # (hermes-evolve) drove the same SQL over a raw sqlite3 handle.
    class _RawLayoutStore(SessionMaintenanceMixin):
        def __init__(self, path):
            self.db_path = path

        def _execute_write(self, fn):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                result = fn(conn)
                conn.commit()
                return result
            finally:
                conn.close()

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            system_prompt TEXT,
            started_at REAL NOT NULL
        );
        INSERT INTO sessions (id, source, system_prompt, started_at)
        VALUES ('legacy1', 'cli', 'old inline prompt', 1.0),
               ('legacy2', 'cli', '', 2.0);
        """
    )
    conn.commit()
    conn.close()

    store = _RawLayoutStore(path)
    assert store.clear_stored_system_prompts() == {"cleared": 1, "storage_mode": "inline"}
    rows = sqlite3.connect(path).execute(
        "SELECT id, system_prompt FROM sessions ORDER BY id"
    ).fetchall()
    # The populated prompt is unset; the already-empty row is untouched
    # (rowcount counts only real changes); both session rows survive. The
    # canonical unset is NULL on EVERY layout — the inline branch normalizes
    # to NULL too, so a cleared prompt reads back identically regardless of
    # which storage layout holds it ('' would make unset layout-dependent,
    # the exact split the interface contract forbids).
    assert rows == [("legacy1", None), ("legacy2", "")]
    # Idempotent on the inline layout too.
    assert store.clear_stored_system_prompts() == {"cleared": 0, "storage_mode": "inline"}


def test_clear_upgrades_legacy_inline_text_through_a_real_open(tmp_path):
    # Opening a legacy inline-text store reconciles the hash column in (the
    # index for it lives in DEFERRED_INDEX_SQL precisely so reconcile runs
    # first) but does NOT move the text (the v25 dedupe is version-gated), so
    # the text stays inline until cleared. clear must blank it in one pass:
    # the out-of-line UPDATE's WHERE covers inline text via its
    # ``system_prompt IS NOT NULL`` arm. The legacy shape is derived from
    # SCHEMA_SQL itself (every sessions column except system_prompt_hash, no
    # prompt-store FK) — exactly the v24-era layout, not a hand-typed guess.
    from hermes_state_common import SCHEMA_SQL

    ref = sqlite3.connect(":memory:")
    ref.executescript(SCHEMA_SQL)
    columns = [row for row in ref.execute("PRAGMA table_info(sessions)") if row[1] != "system_prompt_hash"]
    ref.close()
    col_defs = ",\n    ".join(f'{row[1]} {row[2]}' for row in columns)
    pk = next(row[1] for row in columns if row[5])

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        f"""
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (25);
        CREATE TABLE sessions (
            {col_defs},
            PRIMARY KEY ({pk})
        );
        INSERT INTO sessions (id, source, system_prompt, started_at)
        VALUES ('legacy1', 'cli', 'old inline prompt', 1.0);
        """
    )
    conn.commit()
    conn.close()
    assert "system_prompt_hash" not in {
        row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(sessions)")
    }

    handle = SessionDB(path)
    try:
        result = handle.clear_stored_system_prompts()
        assert result == {"cleared": 1, "storage_mode": "out-of-line"}
        row = handle.get_session("legacy1")
        assert row is not None and row["system_prompt"] is None
    finally:
        handle.close()


# ── apply_atomic — explicit unit of work ────────────────────────────────────


def test_apply_atomic_lands_two_writes_in_one_transaction(db):
    # Two store writes composed into one all-or-nothing unit via the public
    # member, no underscore reach: both land together.
    db.create_session("s1", source="cli")
    def unit(conn):
        db._set_session_title_in_transaction(conn, "s1", "Batched title", source="llm")
        conn.execute("UPDATE sessions SET model = ? WHERE id = ?", ("batched-model", "s1"))
        return "done"
    assert db.apply_atomic(unit) == "done"
    assert db.get_session_title("s1") == "Batched title"
    assert db.get_session("s1")["model"] == "batched-model"


def test_apply_atomic_rolls_back_the_whole_unit_on_failure(db):
    # A mid-unit failure leaves NO partial write visible: the write issued
    # before the failure is rolled back with it.
    db.create_session("s1", source="cli")
    def unit(conn):
        conn.execute("UPDATE sessions SET model = ? WHERE id = ?", ("doomed", "s1"))
        raise RuntimeError("injected mid-unit failure")
    with pytest.raises(RuntimeError, match="injected mid-unit failure"):
        db.apply_atomic(unit)
    assert db.get_session("s1")["model"] is None


# ── helpers ─────────────────────────────────────────────────────────────────


def _rows(db, sql):
    return [dict(row) for row in db._read_all(sql, ())]
