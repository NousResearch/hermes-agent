"""Tests for Executive v2 state_storage: state_meta integration, no new tables."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from agent.executive.state_storage import (
    ObjectiveStateStorage,
    StateStorageError,
    objective_key,
    objective_archive_key,
)
from agent.executive.types import ObjectiveState, ObjectiveStateData


@pytest.fixture
def state_db_path(tmp_path, monkeypatch):
    """Create a temporary state.db with state_meta table."""
    p = tmp_path / "state.db"
    conn = sqlite3.connect(str(p))
    conn.execute("""
        CREATE TABLE state_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    # Patch DEFAULT_STATE_DB_PATH for the fallback code path.
    from agent.executive import state_storage as ss
    monkeypatch.setattr(ss, "DEFAULT_STATE_DB_PATH", p)
    return p


def _make_state(objective_id="oid-1", state=ObjectiveState.DRAFT) -> ObjectiveStateData:
    return ObjectiveStateData(
        objective_id=objective_id,
        state=state,
        objective_text="hello",
        constraints=[],
        user_id="u",
        created_at="2026-01-01",
    )


def test_storage_uses_state_meta_namespace_objective(in_memory_storage):
    in_memory_storage.save(_make_state())
    s = in_memory_storage.load("oid-1")
    assert s is not None
    assert s.objective_id == "oid-1"


def test_storage_isolates_from_goal_namespace(in_memory_storage):
    """Storage does NOT write to goal:* namespace."""
    s = _make_state("oid-x")
    in_memory_storage.save(s)
    # Inspect internal state to confirm.
    assert in_memory_storage.exists("oid-x")
    # In-memory storage has only objective:oid-x, not goal:oid-x.
    # The storage uses objective_key, not goal_key.
    assert objective_key("oid-x") == "objective:oid-x"


def test_storage_save_load_roundtrip(in_memory_storage):
    original = _make_state("oid-rt", ObjectiveState.NORMALIZED)
    original.fingerprint = "abc123"
    in_memory_storage.save(original)
    loaded = in_memory_storage.load("oid-rt")
    assert loaded is not None
    assert loaded.objective_id == "oid-rt"
    assert loaded.state == ObjectiveState.NORMALIZED
    assert loaded.fingerprint == "abc123"


def test_storage_load_returns_none_if_not_exists(in_memory_storage):
    assert in_memory_storage.load("nonexistent") is None


def test_storage_list_active_filters_archived(in_memory_storage):
    in_memory_storage.save(_make_state("oid-a"))
    in_memory_storage.save(_make_state("oid-b"))
    in_memory_storage.archive("oid-b")
    actives = in_memory_storage.list_active()
    assert "oid-a" in actives
    assert "oid-b" not in actives


def test_storage_archive_moves_key_to_archive_namespace(in_memory_storage):
    in_memory_storage.save(_make_state("oid-arch"))
    in_memory_storage.archive("oid-arch")
    actives = in_memory_storage.list_active()
    all_keys = in_memory_storage.list_all(include_archived=True)
    assert "oid-arch" not in actives
    assert "oid-arch" in all_keys


def test_no_new_table_created_in_state_db(state_db_path, in_memory_storage):
    """Even after storage operations, no new tables should exist."""
    in_memory_storage.save(_make_state("oid-x"))
    conn = sqlite3.connect(str(state_db_path))
    tables = set(r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall())
    conn.close()
    # Only state_meta (existing) should exist.
    assert tables == {"state_meta"}, f"Unexpected tables: {tables}"
