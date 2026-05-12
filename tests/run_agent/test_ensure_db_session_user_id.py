"""Tests for _ensure_db_session propagating self._user_id (fixes #24321).

Orphan session rows with user_id=NULL were created because _ensure_db_session
hardcoded user_id=None instead of forwarding self._user_id.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent


def _make_agent(user_id=None, platform="cli"):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "test-session-001"
    agent._user_id = user_id
    agent.platform = platform
    agent._session_db = MagicMock()
    agent._session_db_created = False
    agent.model = "gpt-4o"
    agent._session_init_model_config = {}
    agent._cached_system_prompt = ""
    agent._parent_session_id = None
    return agent


def test_user_id_propagated_to_create_session():
    agent = _make_agent(user_id="user-123", platform="telegram")
    agent._ensure_db_session()
    kwargs = agent._session_db.create_session.call_args.kwargs
    assert kwargs["user_id"] == "user-123"


def test_none_user_id_still_forwarded():
    """CLI sessions with user_id=None are valid; None must flow through."""
    agent = _make_agent(user_id=None, platform="cli")
    agent._ensure_db_session()
    kwargs = agent._session_db.create_session.call_args.kwargs
    assert kwargs["user_id"] is None


def test_idempotent_second_call_skipped():
    """_ensure_db_session must not call create_session twice."""
    agent = _make_agent(user_id="user-abc")
    agent._ensure_db_session()
    agent._ensure_db_session()
    assert agent._session_db.create_session.call_count == 1
