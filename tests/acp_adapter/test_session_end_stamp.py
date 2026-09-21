"""ACP sessions must receive an end stamp when the stdio client disconnects.

`sessions prune` / `sessions archive` share a selector pinned to `ended_at IS NOT NULL`,
so a source='acp' row that never gets ended_at sits outside every bulk cleanup path
forever while the CLI prints "No sessions match" and exits 0.
"""

import sys
import os
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import hermes_state
from acp_adapter.session import SessionManager


def _manager_with_db(tmp_path):
    db = hermes_state.SessionDB(db_path=tmp_path / "state.db")
    return SessionManager(agent_factory=MagicMock, db=db), db


class TestEndAllSessions:
    def test_open_acp_rows_get_ended(self, tmp_path):
        manager, db = _manager_with_db(tmp_path)
        state = manager.create_session(cwd=".")
        state.history.append({"role": "user", "content": "hello"})
        manager._persist(state)
        assert db.get_session(state.session_id)["ended_at"] is None

        manager.end_all_sessions()

        row = db.get_session(state.session_id)
        assert row["ended_at"] is not None
        assert row["end_reason"] == "acp_shutdown"

    def test_already_ended_rows_are_left_alone(self, tmp_path):
        manager, db = _manager_with_db(tmp_path)
        state = manager.create_session(cwd=".")
        state.history.append({"role": "user", "content": "hello"})
        manager._persist(state)
        db.end_session(state.session_id, "compression")

        manager.end_all_sessions()

        row = db.get_session(state.session_id)
        assert row["end_reason"] == "compression"

    def test_idempotent(self, tmp_path):
        manager, db = _manager_with_db(tmp_path)
        state = manager.create_session(cwd=".")
        state.history.append({"role": "user", "content": "hello"})
        manager._persist(state)

        manager.end_all_sessions()
        first = db.get_session(state.session_id)["ended_at"]
        manager.end_all_sessions()
        assert db.get_session(state.session_id)["ended_at"] == first

    def test_end_stamp_prefers_agent_head_row(self, tmp_path):
        manager, db = _manager_with_db(tmp_path)
        state = manager.create_session(cwd=".")
        state.history.append({"role": "user", "content": "hello"})
        manager._persist(state)
        # Compression rotated the agent onto a new head row that the DB knows.
        rotated = "20260921_140000_rotatedh"
        db.create_session(session_id=rotated, source="acp")
        state.agent.session_id = rotated

        manager.end_all_sessions()

        assert db.get_session(rotated)["ended_at"] is not None
