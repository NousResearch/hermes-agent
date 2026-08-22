"""Tests for string db_path coercion in SessionDB and helper functions in hermes_state.py."""

from pathlib import Path
import pytest
from hermes_state import (
    SessionDB,
    collect_state_db_stats,
    repair_state_db_schema,
)


def test_session_db_init_with_string_path(tmp_path):
    """SessionDB should accept db_path as a string without AttributeError."""
    db_file = tmp_path / "str_test.db"
    db_str = str(db_file)

    db = SessionDB(db_path=db_str)
    assert isinstance(db.db_path, Path)
    assert db.db_path == db_file

    # Verify basic DB functionality works
    session_id = "test_sess_str_1"
    db.create_session(
        session_id=session_id,
        source="cli",
        model="test-model",
        system_prompt="test prompt",
    )
    loaded = db.get_session(session_id)
    assert loaded is not None
    assert loaded["id"] == session_id
    assert loaded["model"] == "test-model"
    db.close()


def test_session_db_init_with_path_object(tmp_path):
    """SessionDB should continue to accept Path objects properly."""
    db_file = tmp_path / "path_test.db"
    db = SessionDB(db_path=db_file)
    assert isinstance(db.db_path, Path)
    assert db.db_path == db_file
    db.close()


def test_collect_state_db_stats_with_string_path(tmp_path):
    """collect_state_db_stats should accept a string db_path."""
    db_file = tmp_path / "stats_test.db"
    db = SessionDB(db_path=db_file)
    db.create_session(session_id="s1", source="cli", model="m", system_prompt="p")
    db.close()

    stats = collect_state_db_stats(str(db_file))
    assert isinstance(stats, dict)
    assert stats["sessions"] == 1


def test_repair_state_db_schema_with_string_path(tmp_path):
    """repair_state_db_schema should accept a string db_path."""
    db_file = tmp_path / "repair_test.db"
    db = SessionDB(db_path=db_file)
    db.create_session(session_id="s2", source="cli", model="m", system_prompt="p")
    db.close()

    report = repair_state_db_schema(str(db_file), backup=False)
    assert isinstance(report, dict)
    assert report["repaired"] is True
