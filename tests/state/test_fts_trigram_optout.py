"""sessions.trigram_fts: profile-safe opt-out for the trigram FTS index (v30).

Salvage of PR #91423 onto the schema-v30 shared predicate architecture.
``sessions.trigram_fts: false`` (config.yaml beside each state.db) prevents
trigram DDL on fresh DBs and quarantines existing trigram storage by dropping
its write triggers — never by destructive table drops on ordinary open.
Canonical messages, the standard ``messages_fts`` index, and the default
v29/v30 ``cron``/``subagent`` source exclusions are untouched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_TRIGRAM_EXCLUDED_SOURCES

_FTS_TRIGRAM_TRIGGERS = (
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


def _write_trigram_config(db_path: Path, value) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        (db_path.parent / "config.yaml").write_text("", encoding="utf-8")
        return
    (db_path.parent / "config.yaml").write_text(
        f"sessions:\n  trigram_fts: {value}\n", encoding="utf-8"
    )


def _trigger_names(conn: sqlite3.Connection) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }


def _exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()
        is not None
    )


def _trigram_rowids(db: SessionDB) -> set:
    return {
        row[0]
        for row in db._conn.execute(
            "SELECT id FROM messages_fts_trigram_docsize"
        ).fetchall()
    }


def _seed_mixed_sources(db: SessionDB) -> dict:
    """One interactive + one cron + one delegate-child session (v30 classes)."""
    db.create_session("root", source="cli")
    db.create_session("cronjob", source="cron")
    db.create_session(
        "kid",
        source="subagent",
        parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    return {
        "root": db.append_message(
            "root", role="user", content="交付状态正常 root-word"
        ),
        "cronjob": db.append_message(
            "cronjob", role="user", content="定时任务内容 cron-word"
        ),
        "kid": db.append_message(
            "kid", role="assistant", content="子任务状态正常 kid-word"
        ),
    }


def _fresh_trigram_db(tmp_path: Path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    if not db._trigram_available:
        db.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")
    return db


# ── 1. Default / true ────────────────────────────────────────────────────


def test_default_no_config_preserves_v30_behavior(tmp_path):
    db = _fresh_trigram_db(tmp_path)
    try:
        assert db._trigram_enabled is True
        ids = _seed_mixed_sources(db)
        # v30 defaults: cron + subagent excluded from trigram, kept in FTS.
        assert _trigram_rowids(db) == {ids["root"]}
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts_docsize"
        ).fetchone()[0] >= 3
    finally:
        db.close()


def test_explicit_true_matches_default(tmp_path):
    _write_trigram_config(tmp_path / "state.db", True)
    db = _fresh_trigram_db(tmp_path)
    try:
        assert db._trigram_enabled is True
        ids = _seed_mixed_sources(db)
        assert _trigram_rowids(db) == {ids["root"]}
    finally:
        db.close()


# ── 2. False on fresh DB ─────────────────────────────────────────────────


def test_false_on_fresh_db_skips_trigram_ddl(tmp_path):
    db_path = tmp_path / "state.db"
    _write_trigram_config(db_path, False)
    db = SessionDB(db_path=db_path)
    try:
        assert db._conn is not None
        assert db._trigram_enabled is False
        assert db._trigram_available is False
        assert not _exists(db._conn, "messages_fts_trigram")
        assert not _exists(db._conn, "messages_fts_trigram_src")
        names = _trigger_names(db._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names
        # Standard FTS + canonical writes unaffected.
        assert db._fts_enabled is True
        assert _exists(db._conn, "messages_fts")
        sid = "s1"
        db.create_session(sid, source="test")
        mid = db.append_message(sid, role="user", content="hello world")
        assert mid is not None
        assert db.search_messages("hello")
    finally:
        db.close()


def test_false_fresh_db_search_falls_back_safely(tmp_path):
    db_path = tmp_path / "state.db"
    _write_trigram_config(db_path, False)
    db = SessionDB(db_path=db_path)
    try:
        db.create_session("s1", source="test")
        db.append_message("s1", role="user", content="the quick brown fox")
        db.append_message("s1", role="user", content="korean: 프로젝트 관리")
        assert any(
            "quick" in (r.get("snippet") or "")
            for r in db.search_messages("quick")
        )
        # CJK substring query must not raise with trigram disabled.
        cjk = db.search_messages("관리")
        assert any("관리" in (r.get("snippet") or "") for r in cjk)
    finally:
        db.close()


# ── 3. Disable existing DB (quarantine) ──────────────────────────────────


def test_false_quarantines_existing_db_non_destructively(tmp_path):
    path = tmp_path / "state.db"
    db = _fresh_trigram_db(tmp_path)
    ids = _seed_mixed_sources(db)
    db.close()

    _write_trigram_config(path, False)
    db2 = SessionDB(db_path=path)
    try:
        assert db2._trigram_enabled is False
        assert db2._trigram_available is False
        # Storage preserved for a later controlled rebuild; triggers gone.
        assert _exists(db2._conn, "messages_fts_trigram")
        assert _exists(db2._conn, "messages_fts_trigram_src")
        assert db2._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
        names = _trigger_names(db2._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names
        # Base FTS triggers survive untouched.
        assert "messages_fts_insert" in names
        # Canonical messages intact; writes keep working.
        assert db2._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE id IN (?, ?, ?)",
            tuple(ids.values()),
        ).fetchone()[0] == 3
        mid = db2.append_message("root", role="user", content="during quarantine")
        assert mid is not None
    finally:
        db2.close()


def test_read_only_open_never_serves_quarantined_trigram(tmp_path):
    path = tmp_path / "state.db"
    db = _fresh_trigram_db(tmp_path)
    _seed_mixed_sources(db)
    db.close()

    _write_trigram_config(path, False)
    writer = SessionDB(db_path=path)
    writer.close()  # quarantine marker + trigger drop happen on writable open

    ro = SessionDB(db_path=path, read_only=True)
    try:
        assert ro._trigram_enabled is False
        assert ro._trigram_available is False
        # Search must still answer via base FTS / LIKE, never trigram.
        assert ro.search_messages("root-word")
        assert ro.search_messages("交付状态")
    finally:
        ro.close()


def test_repeated_disabled_opens_do_not_rebuild_or_oscillate(tmp_path):
    path = tmp_path / "state.db"
    _write_trigram_config(path, False)
    db = SessionDB(db_path=path)
    db.create_session("s1", source="test")
    db.append_message("s1", role="user", content="hello world")
    db.close()

    import hermes_state_schema

    calls = []
    orig = hermes_state_schema.SessionSchemaMixin._rebuild_fts_indexes

    def _spy(cursor, *, include_trigram=True):
        calls.append(include_trigram)
        return orig(cursor, include_trigram=include_trigram)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            hermes_state_schema.SessionSchemaMixin,
            "_rebuild_fts_indexes",
            staticmethod(_spy),
        )
        db2 = SessionDB(db_path=path)
        try:
            assert calls == [], (
                "reopening a stable disabled-trigram DB must not trigger a "
                f"full FTS rebuild (got calls={calls!r})"
            )
        finally:
            db2.close()


# ── 4. Config resolution failure ─────────────────────────────────────────


def test_config_resolution_failure_preserves_quarantine(tmp_path):
    path = tmp_path / "state.db"
    db = _fresh_trigram_db(tmp_path)
    db.close()

    _write_trigram_config(path, False)
    SessionDB(db_path=path).close()  # quarantined

    # Break the config AFTER the quarantine exists: fail closed.
    (path.parent / "config.yaml").write_text(
        "sessions:\n  trigram_fts: [broken\n", encoding="utf-8"
    )
    db3 = SessionDB(db_path=path)
    try:
        assert db3._trigram_available is False
        names = _trigger_names(db3._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names
    finally:
        db3.close()


def test_config_resolution_failure_on_fresh_db_defaults_enabled(tmp_path):
    path = tmp_path / "state.db"
    (path.parent / "config.yaml").write_text(
        "sessions:\n  trigram_fts: [broken\n", encoding="utf-8"
    )
    db = SessionDB(db_path=path)
    try:
        # No quarantine evidence on a fresh DB -> historical default (on).
        assert db._trigram_enabled is True
        assert db._trigram_available is True
    finally:
        db.close()


# ── 5. Re-enable ─────────────────────────────────────────────────────────


def test_reenable_rebuilds_with_v30_exclusions(tmp_path):
    path = tmp_path / "state.db"
    db = _fresh_trigram_db(tmp_path)
    ids = _seed_mixed_sources(db)
    db.close()

    _write_trigram_config(path, False)
    disabled = SessionDB(db_path=path)
    disabled.append_message(
        "root", role="user", content="during quarantine 관리도구"
    )
    disabled.close()

    _write_trigram_config(path, True)
    rebuilt = SessionDB(db_path=path)
    try:
        assert rebuilt._trigram_enabled is True
        if rebuilt._trigram_available:
            # v30 predicate restored: cron/subagent stay excluded after
            # re-enable; the new post-quarantine message IS indexed.
            rebuilt_ids = _trigram_rowids(rebuilt)
            assert rebuilt_ids
            assert ids["root"] in rebuilt_ids
            assert ids["cronjob"] not in rebuilt_ids
            assert ids["kid"] not in rebuilt_ids
            assert rebuilt._conn.execute(
                "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
            ).fetchone() is None
            assert rebuilt.search_messages("관리도구")
    finally:
        rebuilt.close()


def test_read_only_reenable_waits_for_writable_rebuild(tmp_path):
    path = tmp_path / "state.db"
    db = _fresh_trigram_db(tmp_path)
    _seed_mixed_sources(db)
    db.close()

    _write_trigram_config(path, False)
    SessionDB(db_path=path).close()

    _write_trigram_config(path, True)
    ro = SessionDB(db_path=path, read_only=True)
    try:
        # Marker still present: read-only open must not serve half-built
        # trigram results even though config now says true.
        assert ro._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is not None
        assert ro._trigram_available is False
        assert ro.search_messages("交付状态")
    finally:
        ro.close()
    writer = SessionDB(db_path=path)
    try:
        assert writer._trigram_available is True
    finally:
        writer.close()


# ── 6. Optimize-storage and recovery paths ───────────────────────────────


def test_v29_to_v30_migration_respects_false(tmp_path):
    """The v29→v30 trigram migration must not resurrect a disabled index."""
    path = tmp_path / "state.db"
    _write_trigram_config(path, False)
    db = SessionDB(db_path=path)
    try:
        assert not _exists(db._conn, "messages_fts_trigram")
        db._conn.execute("UPDATE schema_version SET version = 29")
        db._conn.commit()
    finally:
        db.close()

    db2 = SessionDB(db_path=path)
    try:
        assert db2._trigram_enabled is False
        assert not _exists(db2._conn, "messages_fts_trigram")
        assert not _exists(db2._conn, "messages_fts_trigram_src")
    finally:
        db2.close()


def test_fts_stale_recovery_respects_false(tmp_path):
    """A stale-FTS recovery (corrupt index rebuild) must not rebuild trigram
    when the knob is false."""
    from hermes_state_common import FTS_STALE_KEY

    path = tmp_path / "state.db"
    db = _fresh_trigram_db(tmp_path)
    _seed_mixed_sources(db)
    db.close()

    _write_trigram_config(path, False)
    disabled = SessionDB(db_path=path)
    conn = disabled._conn
    conn.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
        "ON CONFLICT(key) DO UPDATE SET value = '1'",
        (FTS_STALE_KEY,),
    )
    conn.commit()
    disabled.close()

    recovered = SessionDB(db_path=path)
    try:
        assert recovered._fts_stale is False or recovered._fts_enabled is True
        # Trigram must stay disabled regardless of recovery outcome.
        assert recovered._trigram_available is False
        names = _trigger_names(recovered._conn)
        for trig in _FTS_TRIGRAM_TRIGGERS:
            assert trig not in names
        assert recovered.search_messages("root-word")
    finally:
        recovered.close()


def test_optimize_storage_retires_disabled_trigram(tmp_path):
    path = tmp_path / "state.db"
    db = _fresh_trigram_db(tmp_path)
    _seed_mixed_sources(db)
    db.close()

    _write_trigram_config(path, False)
    db2 = SessionDB(db_path=path)
    try:
        assert db2._trigram_available is False
        assert db2.fts_optimize_available() is True
        result = db2.optimize_fts_storage(vacuum=False)
        assert result["ok"] is True
        assert not _exists(db2._conn, "messages_fts_trigram")
        assert not _exists(db2._conn, "messages_fts_trigram_src")
        assert db2._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_trigram_stale'"
        ).fetchone() is None
        assert db2.search_messages("root-word")
    finally:
        db2.close()


# ── 7. Profile safety ────────────────────────────────────────────────────


def test_profiles_isolated_no_cross_leak(tmp_path):
    """Two profile dirs side by side: one disabled, one enabled. Each DB
    resolves its own adjacent config; no process-global bridge."""
    prof_a = tmp_path / "alpha"
    prof_b = tmp_path / "beta"
    prof_a.mkdir(parents=True)
    prof_b.mkdir(parents=True)
    (prof_a / "config.yaml").write_text(
        "sessions:\n  trigram_fts: false\n", encoding="utf-8"
    )

    db_a = SessionDB(db_path=prof_a / "state.db")
    try:
        assert db_a._trigram_enabled is False
        assert not _exists(db_a._conn, "messages_fts_trigram")
    finally:
        db_a.close()

    db_b = SessionDB(db_path=prof_b / "state.db")
    try:
        assert db_b._trigram_enabled is True
        if db_b._trigram_available:
            assert _exists(db_b._conn, "messages_fts_trigram")
    finally:
        db_b.close()


def test_gate_uses_public_resolver(tmp_path):
    from hermes_state import _trigram_fts_enabled_from_config

    path = tmp_path / "state.db"
    assert _trigram_fts_enabled_from_config(path) is True
    _write_trigram_config(path, False)
    assert _trigram_fts_enabled_from_config(path) is False
    _write_trigram_config(path, True)
    assert _trigram_fts_enabled_from_config(path) is True


def test_stale_docstring_reference_replaced():
    import inspect

    import hermes_state_search

    src = inspect.getsource(
        hermes_state_search.SessionSearchMixin.optimize_fts
    )
    assert "HERMES_DISABLE_FTS_TRIGRAM" not in src
    assert "sessions.trigram_fts" in src
