"""Soft eviction (release_clients) releases an agent-owned context engine exactly once.

Gateway cache eviction (gateway/run_agent_cache.py::_release_evicted_agent_soft) releases cached
agents through release_clients(), not close(). That soft path never touched the agent's
context-engine clone, so its SQLite connection + WAL/SHM fds lived until GC or process exit
(#147087, the gateway-side half of #87328's leak class). These tests pin the soft path to
agent.context_engine.shutdown_context_engine() — idempotent through that module's once-flag —
scoped to agent-owned engines only, and ordered: a session-end claim, taken before the gateway
queues its hard cleanup, keeps on_session_end ahead of a later soft eviction's shutdown().
"""

import threading
from unittest.mock import Mock

from agent.context_engine import claim_engine_session_end
from run_agent import AIAgent


class _OrderingEngine:
    """Records the lifecycle order an LCM-style engine depends on."""

    name = "lcm"

    def __init__(self):
        self.calls = []

    def on_session_end(self, session_id, messages):
        self.calls.append("on_session_end")

    def shutdown(self):
        self.calls.append("shutdown")


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


def test_session_end_claim_defers_soft_eviction_until_the_hard_path():
    # The /new shape: claim taken before the hard cleanup is queued; the 30 s wait expires; the
    # agent is soft-evicted while the job is queued or running — the soft path must not touch the
    # engine, and the hard path's on_session_end lands before its finally shuts it down.
    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    assert claim_engine_session_end(agent) is True
    agent.release_clients()
    assert engine.calls == []
    agent.shutdown_memory_provider([])
    assert engine.calls == ["on_session_end", "shutdown"]


def test_soft_release_first_still_shuts_down_exactly_once():
    # No claim: the soft eviction owns the shutdown; a hard teardown afterwards must not
    # double-release (plugins have no idempotency contract on shutdown()).
    engine = _OrderingEngine()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.release_clients()
    assert engine.calls == ["shutdown"]
    agent.shutdown_memory_provider([])
    assert engine.calls.count("shutdown") == 1


def test_claim_re_take_wins_session_end_over_a_won_soft_release():
    agent = _bare_agent(context_compressor=Mock(), _context_engine_owned=True)
    assert claim_engine_session_end(agent) is True
    assert claim_engine_session_end(agent) is True  # session-end callers may re-claim
    other = _bare_agent(context_compressor=Mock(), _context_engine_owned=True)
    other.release_clients()  # soft release claims once for itself
    assert claim_engine_session_end(other) is False  # the engine is already down


def test_child_soft_close_releases_child_engine_not_the_parents():
    child_engine, parent_engine = Mock(), Mock()
    child = _bare_agent(context_compressor=child_engine, _context_engine_owned=True)
    parent = _bare_agent(context_compressor=parent_engine, _context_engine_owned=True)
    parent._active_children = [child]
    parent._close_active_children(soft=True)
    child_engine.shutdown.assert_called_once_with()
    parent_engine.shutdown.assert_not_called()


def test_retired_agent_release_shuts_down_its_engine():
    # hermes_cli cli_agent_setup_mixin._retire_agent and managed_worker.retire_agent both drop
    # the instance right after release_clients(); the engine must go with it.
    engine = Mock()
    agent = _bare_agent(context_compressor=engine, _context_engine_owned=True)
    agent.release_clients()
    engine.shutdown.assert_called_once_with()
