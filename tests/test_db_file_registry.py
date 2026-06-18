"""Tests for the Hermes-home database file registry."""

import logging

from hermes_constants import (
    REGISTERED_DB_FILES,
    registered_db_root_entries,
)
from hermes_state import SessionDB


def test_registered_db_file_registry_includes_current_and_legacy_names():
    assert "state.db" in REGISTERED_DB_FILES
    assert "response_store.db" in REGISTERED_DB_FILES
    assert "kanban.db" in REGISTERED_DB_FILES

    root_entries = registered_db_root_entries()
    assert "state.db" in root_entries
    assert "state.db-wal" in root_entries
    assert "state.db-shm" in root_entries
    assert "hermes_state.db" in root_entries
    assert "response_store.db" in root_entries
    assert "response_store.db-wal" in root_entries


def test_session_db_removes_empty_legacy_state_db(tmp_path, caplog):
    legacy = tmp_path / "hermes_state.db"
    legacy.touch()

    with caplog.at_level(logging.WARNING):
        db = SessionDB(tmp_path / "state.db")
        db.close()

    assert not legacy.exists()
    assert "Removed empty legacy database" in caplog.text


def test_session_db_preserves_nonempty_legacy_state_db(tmp_path, caplog):
    legacy = tmp_path / "hermes_state.db"
    legacy.write_bytes(b"not empty")

    with caplog.at_level(logging.WARNING):
        db = SessionDB(tmp_path / "state.db")
        db.close()

    assert legacy.exists()
    assert "Keeping non-empty legacy database" in caplog.text


def test_session_db_warns_for_unregistered_root_db(tmp_path, caplog):
    rogue = tmp_path / "plugin_cache.db"
    rogue.touch()

    with caplog.at_level(logging.WARNING):
        db = SessionDB(tmp_path / "state.db")
        db.close()

    assert "Unregistered database file" in caplog.text
    assert "plugin_cache.db" in caplog.text
