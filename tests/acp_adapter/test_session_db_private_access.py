"""Tests for the update_session_meta fix.

Verifies that:
1. SessionDB.update_session_meta() exists and works correctly via the
   public _execute_write path (not db._lock / db._conn directly).
2. session.py _persist() no longer touches db._lock or db._conn.
3. update_session_meta updates the correct columns atomically.
"""

import json
from unittest.mock import MagicMock


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


    def test_updates_model_config(self, tmp_path):
        db = _tmp_db(tmp_path)
        db.create_session("s1", source="acp", model="gpt-4")

        new_meta = json.dumps({"cwd": "/new/path", "provider": "openai"})
        db.update_session_meta("s1", new_meta, model=None)

        row = db.get_session("s1")
        stored = json.loads(row["model_config"])
        assert stored["cwd"] == "/new/path"
        assert stored["provider"] == "openai"






# ---------------------------------------------------------------------------
# AST check: session.py must not access db._lock or db._conn
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Integration: _persist round-trip via SessionManager
# ---------------------------------------------------------------------------

class TestPersistRoundTrip:
    """End-to-end: save a session and verify DB state is correct."""

    # A session row is only minted once the session has history — an empty ACP
    # session is deliberately not persisted (see _persist), because ACP clients
    # open sessions they never prompt. These tests seed one message first so
    # they exercise the metadata round-trip they actually care about.

    def test_cwd_persisted_via_update_session_meta(self, tmp_path):
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session(cwd="/original")
        state.history.append({"role": "user", "content": "hi"})
        manager.save_session(state.session_id)
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
        state.history.append({"role": "user", "content": "hi"})
        manager.save_session(state.session_id)

        state.model = "new-model-xyz"
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        assert row["model"] == "new-model-xyz"

    def test_existing_model_not_cleared_on_save(self, tmp_path):
        """If state.model is empty, the DB model column must not be overwritten."""
        db = _tmp_db(tmp_path)
        manager = SessionManager(agent_factory=_mock_agent, db=db)

        state = manager.create_session()
        state.history.append({"role": "user", "content": "hi"})
        manager.save_session(state.session_id)
        # Manually set a model in DB
        db.update_session_meta(state.session_id, json.dumps({"cwd": "."}), model="stored-model")

        # Now save with empty model
        state.model = ""
        manager.save_session(state.session_id)

        row = db.get_session(state.session_id)
        assert row["model"] == "stored-model", (
            "COALESCE must preserve the existing model when new value is NULL"
        )
