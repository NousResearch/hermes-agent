"""Session-scoped AgentRuntime cleanup at AIAgent ownership boundaries."""

from __future__ import annotations

from threading import RLock

from run_agent import AIAgent


class _RuntimeBinding:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _bare_agent(binding):
    agent = AIAgent.__new__(AIAgent)
    agent._runtime_session_binding = binding
    agent._active_children_lock = RLock()
    agent._active_children = set()
    agent.client = None
    agent._request_openai_client = None
    agent._request_anthropic_client = None
    agent._memory_manager = None
    agent.context_compressor = None
    agent._session_messages = []
    agent.session_id = "synthetic-session"
    agent._session_db = None
    agent._owns_session_db = False
    return agent


def test_release_clients_closes_cached_runtime_binding_exactly_once():
    binding = _RuntimeBinding()
    agent = _bare_agent(binding)

    agent.release_clients()
    agent.release_clients()

    assert binding.close_calls == 1
    assert agent._runtime_session_binding is None


def test_hard_close_closes_cached_runtime_binding_exactly_once():
    binding = _RuntimeBinding()
    agent = _bare_agent(binding)

    agent.close()
    agent.close()

    assert binding.close_calls == 1
    assert agent._runtime_session_binding is None
