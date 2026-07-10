"""Tests for explicit cleanup of per-agent Codex app-server sessions."""

from types import SimpleNamespace
from unittest.mock import Mock

from agent.codex_runtime import close_codex_session


def test_close_codex_session_closes_and_clears_session():
    session = Mock()
    agent = SimpleNamespace(_codex_session=session)

    close_codex_session(agent)

    session.close.assert_called_once_with()
    assert agent._codex_session is None


def test_close_codex_session_is_noop_without_session():
    agent_without_attribute = SimpleNamespace()
    agent_with_none = SimpleNamespace(_codex_session=None)

    close_codex_session(agent_without_attribute)
    close_codex_session(agent_with_none)

    assert not hasattr(agent_without_attribute, "_codex_session")
    assert agent_with_none._codex_session is None


def test_close_codex_session_swallows_close_error_and_clears_session():
    session = Mock()
    session.close.side_effect = RuntimeError("close failed")
    agent = SimpleNamespace(_codex_session=session)

    close_codex_session(agent)

    session.close.assert_called_once_with()
    assert agent._codex_session is None
