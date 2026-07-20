"""Tests for the `state.disable_fts_trigram` config flag (issue #55233).

Covers: fresh DBs skip the trigram index when disabled, no boot rebuild loop,
default behavior is unchanged, an existing trigram index is dropped on opt-out,
and the config.yaml read path itself works.
"""

import sqlite3

import pytest

import hermes_state
from hermes_state import (
    _FTS_BASE_TRIGGERS,
    _FTS_TRIGRAM_TRIGGERS,
    SessionDB,
)


def _table_exists(db_path, name):
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None
    finally:
        con.close()


def _trigger_names(db_path):
    con = sqlite3.connect(db_path)
    try:
        return {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
    finally:
        con.close()


def test_disabled_fresh_db_skips_trigram(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_state, "_fts_trigram_disabled", lambda: True)
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)

    assert db._trigram_disabled is True
    assert db._trigram_available is False
    # Trigram table + its 3 triggers must be absent...
    assert not _table_exists(db_path, "messages_fts_trigram")
    assert _trigram_names_absent(db_path)
    # ...but base FTS + its triggers must be present and functional.
    assert _table_exists(db_path, "messages_fts")
    assert set(_FTS_BASE_TRIGGERS).issubset(_trigger_names(db_path))

    # Word search via messages_fts still works.
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="hello trigram world")
    results = db.search_messages("trigram")
    assert any("trigram" in (r.get("snippet") or "") for r in results)


def _trigram_names_absent(db_path):
    names = _trigger_names(db_path)
    return not set(_FTS_TRIGRAM_TRIGGERS) & names


def test_disabled_no_boot_rebuild_loop(tmp_path, monkeypatch):
    """A disabled trigram must NOT make the 2nd open think triggers are missing
    (which would rebuild the whole FTS on every boot)."""
    monkeypatch.setattr(hermes_state, "_fts_trigram_disabled", lambda: True)
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path)

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        # With trigram disabled we only expect the base triggers, and they are
        # all present -> no repair needed.
        assert SessionDB._fts_trigger_count(cur, _FTS_BASE_TRIGGERS) == len(
            _FTS_BASE_TRIGGERS
        )
    finally:
        con.close()

    # Second open must construct cleanly (no exception, trigram still absent).
    db2 = SessionDB(db_path=db_path)
    assert db2._trigram_available is False
    assert not _table_exists(db_path, "messages_fts_trigram")


def test_default_keeps_trigram(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_state, "_fts_trigram_disabled", lambda: False)
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)

    assert db._trigram_disabled is False
    # Default preserves current behavior: trigram present (unless the SQLite
    # build lacks the tokenizer, in which case _trigram_available is False but
    # that is orthogonal to this flag).
    if db._trigram_available:
        assert _table_exists(db_path, "messages_fts_trigram")
        assert set(_FTS_TRIGRAM_TRIGGERS).issubset(_trigger_names(db_path))


def test_existing_trigram_dropped_on_optout(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    # First build WITH trigram (default).
    monkeypatch.setattr(hermes_state, "_fts_trigram_disabled", lambda: False)
    seeded = SessionDB(db_path=db_path)
    if not seeded._trigram_available:
        pytest.skip("SQLite build lacks the trigram tokenizer")
    assert _table_exists(db_path, "messages_fts_trigram")

    # Reopen WITH the flag on -> trigram table + triggers must be dropped.
    monkeypatch.setattr(hermes_state, "_fts_trigram_disabled", lambda: True)
    reopened = SessionDB(db_path=db_path)
    assert reopened._trigram_available is False
    assert not _table_exists(db_path, "messages_fts_trigram")
    assert _trigram_names_absent(db_path)


def test_config_yaml_read_path(tmp_path, monkeypatch):
    """The real config.yaml read path (not just the monkeypatch) resolves the
    flag from `state.disable_fts_trigram`."""
    monkeypatch.delenv("HERMES_DISABLE_FTS_TRIGRAM", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    (tmp_path / "config.yaml").write_text("state:\n  disable_fts_trigram: true\n")
    assert hermes_state._fts_trigram_disabled() is True

    (tmp_path / "config.yaml").write_text("state:\n  disable_fts_trigram: false\n")
    assert hermes_state._fts_trigram_disabled() is False

    (tmp_path / "config.yaml").write_text("other: {}\n")
    assert hermes_state._fts_trigram_disabled() is False


def test_env_bridge_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("state:\n  disable_fts_trigram: false\n")
    monkeypatch.setenv("HERMES_DISABLE_FTS_TRIGRAM", "1")
    assert hermes_state._fts_trigram_disabled() is True
    monkeypatch.setenv("HERMES_DISABLE_FTS_TRIGRAM", "off")
    assert hermes_state._fts_trigram_disabled() is False
