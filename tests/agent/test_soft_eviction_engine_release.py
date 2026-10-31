"""Soft eviction (release_clients) releases an agent-owned context engine exactly once.

Gateway cache eviction (gateway/run_agent_cache.py::_release_evicted_agent_soft) releases cached
agents through release_clients(), not close(). That soft path never touched the agent's
context-engine clone, so its SQLite connection + WAL/SHM fds lived until GC or process exit
(#147087, the gateway-side half of #87328's leak class). These tests pin the soft path to
agent.context_engine.shutdown_context_engine() — idempotent through that module's once-flag —
and scoped to agent-owned engines only.
"""

import threading
from unittest.mock import Mock

from run_agent import AIAgent


def _bare_agent(**attrs):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "sid"
    agent._memory_manager = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._context_engine_shutdown_lock = threading.Lock()
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def test_release_clients_shuts_down_owned_engine_once():
    engine = Mock()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.release_clients()
    agent.release_clients()
    engine.shutdown.assert_called_once_with()


def test_release_clients_never_touches_shared_registered_engine():
    engine = Mock()
    # Ownership is opt-OUT: a path holding the process-shared registered engine must say so.
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=False)
    agent.release_clients()
    engine.shutdown.assert_not_called()


def test_release_clients_releases_engine_before_gc_of_agent():
    # The fd leak was fd-level: prove shutdown runs while the agent is still alive.
    calls = []

    class _Engine:
        name = "lcm"

        def shutdown(self):
            calls.append(1)

    agent = _bare_agent(context_compressor=_Engine(), _context_engine_owned=True)
    agent.release_clients()
    assert calls == [1]


def test_close_after_release_clients_is_idempotent_for_engine():
    engine = Mock()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.close()
    agent.release_clients()
    engine.shutdown.assert_called_once_with()
