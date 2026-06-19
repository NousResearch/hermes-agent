"""AIAgent memory-provider teardown lifecycle tests."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from run_agent import AIAgent


def _make_agent(monkeypatch):
    monkeypatch.setattr("run_agent.cleanup_vm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda *_args, **_kwargs: None)

    agent = object.__new__(AIAgent)
    agent.session_id = "test-session"
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent.client = None
    agent.context_compressor = None
    agent._memory_shutdown_done = False
    manager = SimpleNamespace(
        on_session_end=MagicMock(),
        shutdown_all=MagicMock(),
    )
    agent._memory_manager = manager
    return agent, manager


def test_shutdown_memory_provider_is_idempotent(monkeypatch):
    agent, manager = _make_agent(monkeypatch)
    transcript = [{"role": "user", "content": "hello"}]

    agent.shutdown_memory_provider(transcript)
    agent.shutdown_memory_provider(transcript)

    manager.on_session_end.assert_called_once_with(transcript)
    manager.shutdown_all.assert_called_once()


def test_close_calls_shutdown_memory_provider_once(monkeypatch):
    agent, manager = _make_agent(monkeypatch)

    agent.close()
    agent.close()

    manager.on_session_end.assert_called_once_with([])
    manager.shutdown_all.assert_called_once()


def test_release_clients_does_not_shutdown_memory_provider(monkeypatch):
    agent, manager = _make_agent(monkeypatch)

    agent.release_clients()

    manager.on_session_end.assert_not_called()
    manager.shutdown_all.assert_not_called()
    assert agent._memory_shutdown_done is False
