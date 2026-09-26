"""Honcho's synchronous generic memory ownership contract."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.memory_provider import MemoryWriteIntent
from plugins.memory.honcho import HonchoMemoryProvider


@pytest.fixture
def provider():
    p = HonchoMemoryProvider()
    p._config = SimpleNamespace(save_messages=True)
    p._manager = MagicMock()
    p._manager.create_conclusion.return_value = True
    p._session_key = "session-key"
    p._session_initialized = True
    p._turn_author = {}
    return p


def test_add_user_commits_to_user_peer(provider):
    intent = MemoryWriteIntent(action="add", target="user", content="  Likes jazz  ")
    assert provider.wants_memory_write(intent)
    result = provider.handle_memory_write(intent)
    assert result.handled and result.success
    provider._manager.create_conclusion.assert_called_once_with("session-key", "Likes jazz", peer="user")


@pytest.mark.parametrize("action", ["replace", "remove", "batch"])
def test_unsupported_user_operations_fail_without_remote_write(provider, action):
    intent = MemoryWriteIntent(action=action, target="user", old_text="jazz",
                               operations=[{"action": "add", "content": "Likes jazz"}] if action == "batch" else [])
    result = provider.handle_memory_write(intent)
    assert result.handled and not result.success and result.error
    provider._manager.create_conclusion.assert_not_called()


def test_remote_failure_never_reports_success(provider):
    provider._manager.create_conclusion.return_value = False
    result = provider.handle_memory_write(MemoryWriteIntent(action="add", target="user", content="Likes jazz"))
    assert result.handled and not result.success


def test_replay_context_captures_session_identity_without_private_objects(provider):
    provider.initialize("session-1", platform="telegram", user_id="user-1", cwd="/work",
                        agent_context="primary", opaque=object())
    assert provider.memory_write_replay_context() == {
        "session_id": "session-1",
        "session_key": "session-key",
        "kwargs": {"platform": "telegram", "user_id": "user-1", "cwd": "/work",
                   "agent_context": "primary"},
    }
