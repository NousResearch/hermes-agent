"""Regression tests: session continuity when the default/global state.db is unavailable.

Root cause (live incident, 2026-08-17): when the default/global state.db is
corrupt or unreadable at gateway startup, ``SessionStore._db`` degrades to
None (JSONL routing fallback) and ``GatewayRunner._session_db`` is None.
Under multiplexed profile routes the AIAgent lazily opens the PROFILE's
state.db (``AIAgent._get_session_db_for_recall``) and its
``_ensure_db_session`` created the session row WITHOUT routing identity
(session_key/chat_id/chat_type/thread_id/user_id/origin_json/display_name
all NULL), so the row was unrecoverable by
``find_latest_gateway_session_for_peer``. The gateway's own
``record_gateway_session_peer`` self-heal never ran because ``_db`` was None.
On the next message, ``SessionStore.load_transcript`` returned [] even though
the transcript existed in the profile DB -> the agent lost prior context.

These tests pin fix 1 (routing identity on lazy row creation):
``AIAgent._ensure_db_session`` persists the full routing identity the
agent already carries, so the lazily-created row is recoverable. (The
transcript-recovery half of the original report is covered separately by
the scope-aware session DB resolution on main.)
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

SESSION_ID = "degraded-db-continuity-session"
CHAT_ID = "-1001234567890"
THREAD_ID = "42"
SESSION_KEY = "agent:orion:telegram:group:-1001234567890:8148316720"


def _make_gateway_agent(session_db):
    """Build an AIAgent carrying gateway routing identity (multiplex profile route)."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=SESSION_ID,
            skip_context_files=True,
            skip_memory=True,
            platform="telegram",
            user_id="8148316720",
            user_name="testuser",
            chat_id=CHAT_ID,
            chat_name="Test Group",
            chat_type="group",
            thread_id=THREAD_ID,
            gateway_session_key=SESSION_KEY,
        )
    # The multiplexed profile scope would resolve the active profile name to
    # the route's profile (e.g. "orion"); pin it so the row + origin_json
    # carry the profile deterministically.
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="orion"):
        agent._ensure_db_session()
    return agent


class TestEnsureDbSessionPersistsRoutingIdentity:
    def test_lazy_creation_writes_session_key_and_chat_identity(self):
        """First-created row must carry gateway routing metadata (not identity-less)."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                _make_gateway_agent(db)
                row = db.get_session(SESSION_ID)
                assert row is not None
                assert row["session_key"] == SESSION_KEY
                assert row["chat_id"] == CHAT_ID
                assert row["chat_type"] == "group"
                assert row["thread_id"] == THREAD_ID
                assert row["user_id"] == "8148316720"
                assert row["display_name"] == "Test Group"
                assert row["source"] == "telegram"
                assert row["profile_name"] == "orion"
                origin = json.loads(row["origin_json"])
                assert origin["platform"] == "telegram"
                assert origin["chat_id"] == CHAT_ID
                assert origin["chat_name"] == "Test Group"
                assert origin["chat_type"] == "group"
                assert origin["user_id"] == "8148316720"
                assert origin["user_name"] == "testuser"
                assert origin["thread_id"] == THREAD_ID
                assert origin["profile"] == "orion"
            finally:
                db.close()

    def test_row_recoverable_by_peer_lookup(self):
        """find_latest_gateway_session_for_peer must find the lazily-created row."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                _make_gateway_agent(db)
                # The peer lookup keys on session_key + source; the regression
                # left session_key NULL so this returned None on every restart.
                found = db.find_latest_gateway_session_for_peer(
                    session_key=SESSION_KEY,
                    source="telegram",
                )
                assert found is not None
                assert found["id"] == SESSION_ID
            finally:
                db.close()

    def test_cli_session_stays_identity_less(self):
        """A plain CLI agent (no gateway fields) keeps the old shape — no origin_json."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                    from run_agent import AIAgent

                    agent = AIAgent(
                        api_key="test-key",
                        base_url="https://openrouter.ai/api/v1",
                        model="test/model",
                        quiet_mode=True,
                        session_db=db,
                        session_id=SESSION_ID + "-cli",
                        skip_context_files=True,
                        skip_memory=True,
                    )
                agent._ensure_db_session()
                row = db.get_session(SESSION_ID + "-cli")
                assert row is not None
                assert row["session_key"] is None
                assert row["chat_id"] is None
                assert row["origin_json"] is None
            finally:
                db.close()
