"""Tests for the update_session_meta fix.

Verifies that:
1. SessionDB.update_session_meta() exists and works correctly via the
   public _execute_write path (not db._lock / db._conn directly).
2. session.py _persist() no longer touches db._lock or db._conn.
3. update_session_meta updates the correct columns atomically.
"""

import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from hermes_state import SessionDB
from acp_adapter.session import SessionManager


def _tmp_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _mock_agent():
    return MagicMock(name="MockAIAgent")


# ---------------------------------------------------------------------------
# hermes_state.SessionDB.update_session_meta — unit tests
# ---------------------------------------------------------------------------

class TestUpdateSessionMeta:
    """Direct unit tests for the new public method."""

    def test_method_exists(self, tmp_path):
        db = _tmp_db(tmp_path)
        assert hasattr(db, "update_session_meta"), (
            "SessionDB must have update_session_meta() public method"
        )
        assert callable(db.update_session_meta)

    def test_updates_model_config(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s1", source="acp", model="gpt-4")

        new_meta = json.dumps({"cwd": "/new/path", "provider": "openai"})
        db.update_session_meta("s1", new_meta, model=None)

        row = db.get_session("s1")
        stored = json.loads(row["model_config"])
        assert stored["cwd"] == "/new/path"
        assert stored["provider"] == "openai"

    def test_updates_model_when_provided(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s2", source="acp", model="gpt-3.5")

        db.update_session_meta("s2", json.dumps({"cwd": "."}), model="gpt-4o")

        row = db.get_session("s2")
        assert row["model"] == "gpt-4o"

    def test_preserves_existing_model_when_none(self, tmp_path):
        """Passing model=None must leave the stored model unchanged (COALESCE)."""
        db = _tmp_db(tmp_path)
        db.create_session("s3", source="acp", model="claude-3")

        db.update_session_meta("s3", json.dumps({"cwd": "."}), model=None)

        row = db.get_session("s3")
        assert row["model"] == "claude-3"

    def test_uses_execute_write_not_private_api(self, tmp_path):
        """update_session_meta must route through _execute_write, not _conn directly."""
        db = _tmp_db(tmp_path)
        db.create_session("s4", source="acp")

        call_count = [0]
        original = db._execute_write

        def patched(fn):
            call_count[0] += 1
            return original(fn)

        db._execute_write = patched
        db.update_session_meta("s4", json.dumps({"cwd": "."}), model="m")

        assert call_count[0] >= 1, (
            "update_session_meta must call _execute_write at least once"
        )

    def test_noop_on_nonexistent_session(self, tmp_path):
        """Updating a non-existent session must not raise."""
        db = _tmp_db(tmp_path)
        db.update_session_meta("ghost", json.dumps({"cwd": "."}), model=None)


# ---------------------------------------------------------------------------
# AST check: session.py must not access db._lock or db._conn
# ---------------------------------------------------------------------------

class TestNoPrviateDBAccess:
    """_persist() in session.py must not access db._lock or db._conn."""

    def test_no_db_private_lock_access(self):
        with open("acp_adapter/session.py", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            # Looking for: db._lock  or  db._conn
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "db":
                    if node.attr in ("_lock", "_conn"):
                        violations.append(
                            f"db.{node.attr} at line {node.lineno}"
                        )

        assert violations == [], (
            "session.py accesses private SessionDB internals: "
            + ", ".join(violations)
            + " — use db.update_session_meta() instead"
        )

    def test_persist_calls_update_session_meta(self):
        """AST check: _persist must call db.update_session_meta()."""
        with open("acp_adapter/session.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_persist":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute):
                            if func.attr == "update_session_meta":
                                found = True
                                break
                break

        assert found, (
            "_persist() must call db.update_session_meta() "
            "instead of db._conn.execute() directly"
        )


# ---------------------------------------------------------------------------
# Integration: _persist round-trip via SessionManager
# ---------------------------------------------------------------------------

class TestPersistRoundTrip:
    """End-to-end: save a session and verify DB state is correct."""

    def test_cwd_persisted_via_update_session_meta(self, tmp_path):
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session(cwd="/original")
        assert db.get_session(state.session_id) is not None

        # Simulate cwd change and save
        state.cwd = "/updated"
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        mc = json.loads(row["model_config"])
        assert mc["cwd"] == "/updated"

    def test_model_persisted_via_update_session_meta(self, tmp_path):
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session()
        state.model = "new-model-xyz"
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        assert row["model"] == "new-model-xyz"

    def test_existing_model_not_cleared_on_save(self, tmp_path):
        """If state.model is empty, the DB model column must not be overwritten."""
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session()
        # Manually set a model in DB
        db.update_session_meta(state.session_id, json.dumps({"cwd": "."}), model="stored-model")

        # Now save with empty model
        state.model = ""
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        assert row["model"] == "stored-model", (
            "COALESCE must preserve the existing model when new value is NULL"
        )
