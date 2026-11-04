"""Evicted agents shut down their memory providers so their SQLite connections close (MTK card #150).

``_release_evicted_agent_soft`` previously called only ``release_clients()`` (LLM
clients); the mnemosyne provider's live beam connection sat until the cyclic-garbage
collector caught up — hours in a long-lived gateway. These tests pin the new hook:
shutdown_all() is called after release_clients(), its failures are swallowed, and
agents without a memory manager (or None) are still handled.
"""
from types import SimpleNamespace

from gateway.run_agent_cache import GatewayAgentCacheMixin


class _RecordingAgent:
    def __init__(self):
        self.calls = []

    def release_clients(self):
        self.calls.append("release_clients")


def test_soft_evict_shuts_down_memory_manager():
    agent = _RecordingAgent()
    shutdowns = []
    agent._memory_manager = SimpleNamespace(shutdown_all=lambda: shutdowns.append(1))
    GatewayAgentCacheMixin._release_evicted_agent_soft(object(), agent)
    assert agent.calls == ["release_clients"]
    assert shutdowns == [1]


def test_soft_evict_swallows_provider_shutdown_errors():
    agent = _RecordingAgent()

    def boom():
        raise RuntimeError("provider already gone")

    agent._memory_manager = SimpleNamespace(shutdown_all=boom)
    # Must not raise: eviction is best-effort cleanup.
    GatewayAgentCacheMixin._release_evicted_agent_soft(object(), agent)
    assert agent.calls == ["release_clients"]


def test_soft_evict_tolerates_agent_without_memory_manager():
    agent = _RecordingAgent()
    GatewayAgentCacheMixin._release_evicted_agent_soft(object(), agent)
    assert agent.calls == ["release_clients"]


def test_soft_evict_is_noop_on_none():
    GatewayAgentCacheMixin._release_evicted_agent_soft(object(), None)