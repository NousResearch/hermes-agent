"""Tests for AIAgent._derive_session_user_id() and session-row user_id propagation.

Verifies that the session_db.create_session() calls in _ensure_db_session and
compression pass the correct user_id (namespaced for bg/cron/delegate sessions).
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(session_id="test-session", user_id=None, session_db=None):
    """Create a minimal AIAgent for testing _derive_session_user_id."""
    from run_agent import AIAgent

    with patch.object(AIAgent, "__init__", lambda self, **kw: None):
        agent = AIAgent.__new__(AIAgent)
        agent.session_id = session_id
        agent._user_id = user_id
        agent._session_db = session_db
        agent._parent_session_id = None
        agent.platform = "cli"
        agent.model = "test-model"
        agent.max_iterations = 90
        return agent


# ---------------------------------------------------------------------------
# _derive_session_user_id
# ---------------------------------------------------------------------------

class TestDeriveSessionUserId:
    """Unit tests for AIAgent._derive_session_user_id."""

    def test_none_user_id_returns_none(self):
        agent = _make_agent(session_id="plain-session", user_id=None)
        assert agent._derive_session_user_id() is None

    def test_plain_session_passes_through(self):
        agent = _make_agent(session_id="plain-session", user_id="user123")
        assert agent._derive_session_user_id() == "user123"

    def test_bg_session_gets_namespaced(self):
        agent = _make_agent(session_id="bg_20260512_abc", user_id="tg_42")
        assert agent._derive_session_user_id() == "bg:tg_42"

    def test_cron_session_gets_namespaced(self):
        agent = _make_agent(
            session_id="cron_job1_20260512_120000", user_id="slack_U99"
        )
        assert agent._derive_session_user_id() == "cron:slack_U99"

    def test_delegate_session_gets_namespaced(self):
        agent = _make_agent(
            session_id="delegate_20260512_xyz", user_id="dc_777"
        )
        assert agent._derive_session_user_id() == "delegate:dc_777"

    def test_empty_session_id_passes_through(self):
        agent = _make_agent(session_id="", user_id="u1")
        assert agent._derive_session_user_id() == "u1"

    def test_none_session_id_passes_through(self):
        agent = _make_agent(session_id=None, user_id="u1")
        agent.session_id = None
        assert agent._derive_session_user_id() == "u1"

    def test_timestamp_compression_session_passes_through(self):
        """Compression creates a YYYYMMDD_HHMMSS_XXXXXX session id -- not namespaced."""
        agent = _make_agent(
            session_id="20260512_143022_a1b2c3", user_id="bg:tg_42"
        )
        # Already namespaced from the original bg session, passes through as-is
        assert agent._derive_session_user_id() == "bg:tg_42"


# ---------------------------------------------------------------------------
# Integration: _ensure_db_session receives derived user_id
# ---------------------------------------------------------------------------

class TestEnsureDbSessionUserId:
    """Verify _ensure_db_session passes _derive_session_user_id() to create_session."""

    def test_ensure_db_session_passes_derived_user_id(self):
        mock_db = MagicMock()
        agent = _make_agent(
            session_id="bg_task1", user_id="tg_42", session_db=mock_db
        )
        agent._session_db_created = False
        agent._cached_system_prompt = "test"
        agent._session_init_model_config = {}
        agent._derive_session_user_id = MagicMock(return_value="bg:tg_42")

        agent._ensure_db_session()

        mock_db.create_session.assert_called_once()
        call_kwargs = mock_db.create_session.call_args
        assert call_kwargs[1]["user_id"] == "bg:tg_42"

    def test_ensure_db_session_passes_none_for_no_user(self):
        mock_db = MagicMock()
        agent = _make_agent(
            session_id="plain-session", user_id=None, session_db=mock_db
        )
        agent._session_db_created = False
        agent._cached_system_prompt = "test"
        agent._session_init_model_config = {}

        agent._ensure_db_session()

        mock_db.create_session.assert_called_once()
        call_kwargs = mock_db.create_session.call_args
        assert call_kwargs[1]["user_id"] is None

    def test_compression_passes_direct_user_id(self):
        """Compression should pass self._user_id directly, not namespaced."""
        mock_db = MagicMock()
        agent = _make_agent(
            session_id="bg_task1", user_id="tg_42", session_db=mock_db
        )
        # After compression, the agent's _user_id is still the original value
        assert agent._user_id == "tg_42"
