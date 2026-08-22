"""Adaptation C: config/env gate + stale-quarantine for messages_fts_trigram.

Production evidence: recurrent messages_fts_trigram corruption after FTS
rebuilds. Upstream v2026.8.3 already solves the same problem class for the
CJK-bigram index (config gate + breadcrumb quarantine, see
``_cjk_fts_config_enabled`` / ``_quarantine_cjk_after_update_of_migration``
in hermes_state.py / hermes_state_schema.py) but has no equivalent escape
hatch for the trigram index — ``HERMES_DISABLE_FTS_TRIGRAM`` survives only
as a stale docstring reference with no backing code. These tests establish
the same gate for trigram, modeled on the CJK pattern, WITHOUT a bespoke
``_drop_trigram_schema`` that could leave triggers against a half-dismantled
schema.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_state import SessionDB

_FTS_TRIGRAM_TRIGGERS = (
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


def _write_trigram_config(db_path: Path, enabled: bool) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    (db_path.parent / "config.yaml").write_text(
        f"sessions:\n  trigram_fts: {'true' if enabled else 'false'}\n",
        encoding="utf-8",
    )


def _trigger_names(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }


def _table_or_view_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    return row is not None


def test_trigram_enabled_by_default_on_fresh_db(tmp_path: Path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db._conn is not None
        assert db._trigram_available is True
        assert _table_or_view_exists(db._conn, "messages_fts_trigram")
        assert _table_or_view_exists(db._conn, "messages_fts_trigram_src")
        names = _trigger_names(db._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig in names
    finally:
        db.close()


def test_trigram_disabled_prevents_schema_creation_on_fresh_db(
    tmp_path: Path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _write_trigram_config(db_path, False)
    db = SessionDB(db_path=db_path)
    try:
        assert db._conn is not None
        assert db._trigram_available is False
        assert not _table_or_view_exists(db._conn, "messages_fts_trigram")
        assert not _table_or_view_exists(db._conn, "messages_fts_trigram_src")
        names = _trigger_names(db._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names

        # Canonical messages + main FTS must be completely unaffected.
        assert db._fts_enabled is True
        assert _table_or_view_exists(db._conn, "messages_fts")
        assert "messages_fts_insert" in names
        assert "messages_fts_delete" in names
        assert "messages_fts_update" in names
    finally:
        db.close()


def test_trigram_disabled_quarantines_existing_schema_on_reopen(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    assert db._conn is not None
    sid = "s1"
    db.create_session(sid, source="test")
    db.append_message(sid, role="user", content="hello searchable world")
    assert db._trigram_available is True
    db.close()

    _write_trigram_config(path, False)
    db2 = SessionDB(db_path=path)
    try:
        assert db2._conn is not None
        assert db2._trigram_available is False
        # Match the CJK quarantine pattern: preserve the existing index for a
        # later controlled rebuild, but remove every trigger so a corrupt or
        # stale index cannot break canonical message writes.
        assert _table_or_view_exists(db2._conn, "messages_fts_trigram")
        assert _table_or_view_exists(db2._conn, "messages_fts_trigram_src")
        assert db2._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
        names = _trigger_names(db2._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names

        # Main FTS triggers must survive the quarantine untouched.
        assert "messages_fts_insert" in names
        assert "messages_fts_delete" in names
        assert "messages_fts_update" in names

        # Canonical messages row from before quarantine must still be there.
        rows = db2._conn.execute(
            "SELECT content FROM messages WHERE session_id = ?", (sid,)
        ).fetchall()
        assert any("hello searchable world" in (r[0] or "") for r in rows)

        # Writes must keep working after quarantine (no half-dismantled
        # trigger left referencing a dropped table/view).
        mid = db2.append_message(sid, role="user", content="second message")
        assert mid is not None
    finally:
        db2.close()


def test_reenable_rebuilds_quarantined_trigram_from_canonical_messages(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    assert db._conn is not None
    db.create_session("s1", source="test")
    db.append_message("s1", role="user", content="before quarantine 프로젝트")
    db.close()

    _write_trigram_config(path, False)
    disabled = SessionDB(db_path=path)
    assert disabled._conn is not None
    disabled.append_message("s1", role="user", content="during quarantine 관리도구")
    disabled.close()

    _write_trigram_config(path, True)
    rebuilt = SessionDB(db_path=path)
    try:
        assert rebuilt._conn is not None
        assert rebuilt._trigram_available is True
        assert rebuilt._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is None
        names = _trigger_names(rebuilt._conn)
        assert set(_FTS_TRIGRAM_TRIGGERS) <= names
        results = rebuilt.search_messages("관리도구")
        assert any("관리도구" in (r.get("snippet") or "") for r in results)
    finally:
        rebuilt.close()


def test_trigram_disabled_reopen_does_not_repeat_rebuild(
    tmp_path: Path, monkeypatch
):
    """_fts_trigger_count must be parameterized so a disabled trigram index
    doesn't look like "needs repair" forever and re-trigger a full FTS
    rebuild on every single open."""
    path = tmp_path / "state.db"
    _write_trigram_config(path, False)
    db = SessionDB(db_path=path)
    assert db._conn is not None
    db.create_session("s1", source="test")
    db.append_message("s1", role="user", content="hello world")
    db.close()

    import hermes_state_schema

    calls = []
    orig = hermes_state_schema.SessionSchemaMixin._rebuild_fts_indexes

    def _spy(cursor, *, include_trigram=True):
        calls.append(include_trigram)
        return orig(cursor, include_trigram=include_trigram)

    monkeypatch.setattr(
        hermes_state_schema.SessionSchemaMixin, "_rebuild_fts_indexes", staticmethod(_spy)
    )

    db2 = SessionDB(db_path=path)
    try:
        assert db2._conn is not None
        assert calls == [], (
            "reopening a stable disabled-trigram DB must not trigger a "
            f"full FTS rebuild (got calls={calls!r})"
        )
    finally:
        db2.close()


def test_migrate_broad_update_triggers_does_not_resurrect_disabled_trigram(
    tmp_path: Path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _write_trigram_config(db_path, False)
    db = SessionDB(db_path=db_path)
    try:
        assert db._conn is not None
        assert not _table_or_view_exists(db._conn, "messages_fts_trigram")

        # Force a broad (pre-narrowing) main FTS UPDATE trigger the way an
        # older install would have had it, so the migration has real work.
        db._conn.execute("DROP TRIGGER IF EXISTS messages_fts_update")
        db._conn.execute(
            """
            CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages
            BEGIN
                SELECT 1;
            END
            """
        )
        db._conn.commit()

        cursor = db._conn.cursor()
        dropped = db._migrate_broad_fts_update_triggers(cursor)
        db._conn.commit()
        assert dropped >= 1

        # The migration's blanket DDL re-apply must not resurrect trigram.
        # This is a fresh disabled DB, so there is no old storage to quarantine.
        assert not _table_or_view_exists(db._conn, "messages_fts_trigram")
        assert not _table_or_view_exists(db._conn, "messages_fts_trigram_src")
        names = _trigger_names(db._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names
    finally:
        db.close()


def test_search_falls_back_safely_when_trigram_disabled(
    tmp_path: Path, monkeypatch
):
    db_path = tmp_path / "state.db"
    _write_trigram_config(db_path, False)
    db = SessionDB(db_path=db_path)
    try:
        assert db._conn is not None
        sid = "s1"
        db.create_session(sid, source="test")
        db.append_message(sid, role="user", content="the quick brown fox jumps")
        db.append_message(sid, role="user", content="korean term: 프로젝트 관리")

        # Ordinary latin token — must still be answered via the base FTS5
        # index (no trigram dependency at all).
        results = db.search_messages("quick")
        assert any("quick" in (r.get("snippet") or "") for r in results)

        # Short CJK query that would normally route to trigram/LIKE — must
        # not raise even though trigram is disabled.
        cjk_results = db.search_messages("관리")
        assert any("관리" in (r.get("snippet") or "") for r in cjk_results)
    finally:
        db.close()


def test_trigram_fts_config_gate_documented_and_defaults_enabled(tmp_path):
    """The profile-owned YAML gate defaults on and honors false/true."""
    from hermes_state import _trigram_fts_enabled_from_config

    assert _trigram_fts_enabled_from_config.__doc__, (
        "gate must be documented (config/env parity note)"
    )
    db_path = tmp_path / "state.db"
    assert _trigram_fts_enabled_from_config(db_path) is True
    _write_trigram_config(db_path, False)
    assert _trigram_fts_enabled_from_config(db_path) is False
    _write_trigram_config(db_path, True)
    assert _trigram_fts_enabled_from_config(db_path) is True


def test_stale_trigram_docstring_is_fixed():
    """The HERMES_DISABLE_FTS_TRIGRAM docstring reference (audit finding)
    must be replaced by the real gate name, not left dangling."""
    import inspect

    import hermes_state_search

    src = inspect.getsource(hermes_state_search.SessionSearchMixin.optimize_fts)
    assert "HERMES_DISABLE_FTS_TRIGRAM" not in src
    assert "sessions.trigram_fts" in src
