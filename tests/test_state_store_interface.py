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
    # Existence check for the contracted face (8 core members + the constant).
    for member in (
        "get_session_title",
        "get_session_title_source",
        "set_session_title",
        "set_auto_title",
        "sanitize_title",
        "get_session",
        "get_messages_as_conversation",
        "search_messages",
        "clear_stored_system_prompts",
        "TITLE_SOURCE_LLM",
    ):
        assert hasattr(SessionDB, member), member


def test_sanitize_title_is_the_contracted_normalizer(db):
    assert SessionDB.sanitize_title("  hello \x00world\t ") == "hello world"
    assert SessionDB.sanitize_title("   ") is None


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
    # The populated prompt is blanked; the already-empty row is untouched
    # (rowcount counts only real changes); both session rows survive.
    assert rows == [("legacy1", ""), ("legacy2", "")]
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
        assert row is not None and not (row["system_prompt"] or "")
    finally:
        handle.close()


# ── helpers ─────────────────────────────────────────────────────────────────


def _rows(db, sql):
    return [dict(row) for row in db._read_all(sql, ())]
