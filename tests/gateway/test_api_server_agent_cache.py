"""Tests for the API server agent cache (chat completions agent reuse).

The cache mirrors the gateway's per-session AIAgent cache
(gateway/run.py ``_agent_cache``): agents are checked OUT for the
duration of a run and checked back IN afterwards, keyed by
``gateway_session_key or session_id`` and validated by a config
signature.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter(**extra):
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra or None))


def _instrument(adapter, signature="sig-1"):
    """Stub config resolution / construction / cleanup on the adapter."""
    created = []
    cleaned = []

    def _fake_create(**kwargs):
        agent = MagicMock(name=f"agent-{len(created)}")
        created.append(agent)
        return agent

    adapter._resolve_agent_runtime = MagicMock(
        return_value={"max_iterations": 5}
    )
    adapter._agent_cache_signature = MagicMock(return_value=signature)
    adapter._create_agent = MagicMock(side_effect=_fake_create)
    adapter._cleanup_cached_agent = MagicMock(side_effect=cleaned.append)
    return created, cleaned


def _acquire(adapter, session_id="sess-1", **kwargs):
    return adapter._acquire_agent(
        ephemeral_system_prompt=kwargs.pop("ephemeral_system_prompt", None),
        session_id=session_id,
        stream_delta_callback=kwargs.pop("stream_delta_callback", None),
        **kwargs,
    )


class TestAgentReuse:
    def test_agent_reused_across_turns_of_same_session(self):
        adapter = _make_adapter()
        created, cleaned = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)
        agent2, key2, sig2 = _acquire(adapter)

        assert agent2 is agent1
        assert key2 == key
        assert len(created) == 1
        assert cleaned == []

    def test_callbacks_rebound_on_cache_hit(self):
        adapter = _make_adapter()
        _instrument(adapter)

        agent, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent, sig)

        new_cb = MagicMock(name="delta-cb")
        agent2, _, _ = _acquire(adapter, stream_delta_callback=new_cb)
        assert agent2 is agent
        assert agent2.stream_delta_callback is new_cb
        assert agent2.max_iterations == 5

    def test_different_sessions_get_different_agents(self):
        adapter = _make_adapter()
        created, _ = _instrument(adapter)

        agent1, key1, sig1 = _acquire(adapter, session_id="sess-a")
        adapter._release_agent(key1, agent1, sig1)
        agent2, key2, sig2 = _acquire(adapter, session_id="sess-b")

        assert agent2 is not agent1
        assert len(created) == 2

    def test_gateway_session_key_takes_precedence_in_cache_key(self):
        adapter = _make_adapter()
        _instrument(adapter)

        agent1, key, sig = _acquire(
            adapter, session_id="sess-a", gateway_session_key="chan-1"
        )
        assert key == "chan-1"
        adapter._release_agent(key, agent1, sig)
        # Same channel, different derived session id — still a hit.
        agent2, _, _ = _acquire(
            adapter, session_id="sess-b", gateway_session_key="chan-1"
        )
        assert agent2 is agent1


class TestInvalidation:
    def test_config_signature_change_rebuilds_agent(self):
        adapter = _make_adapter()
        created, cleaned = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)

        adapter._agent_cache_signature = MagicMock(return_value="sig-2")
        agent2, _, _ = _acquire(adapter)

        assert agent2 is not agent1
        assert cleaned == [agent1]

    def test_idle_ttl_expiry_rebuilds_agent(self):
        adapter = _make_adapter()
        created, cleaned = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)
        # Backdate the cached entry beyond the idle TTL.
        with adapter._agent_cache_lock:
            cached_agent, cached_sig, _ = adapter._agent_cache[key]
            adapter._agent_cache[key] = (
                cached_agent,
                cached_sig,
                time.time() - adapter._AGENT_CACHE_IDLE_TTL - 1,
            )

        agent2, _, _ = _acquire(adapter)
        assert agent2 is not agent1
        assert cleaned == [agent1]

    def test_discard_on_release_cleans_up_and_skips_cache(self):
        adapter = _make_adapter()
        created, cleaned = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig, discard=True)

        assert cleaned == [agent1]
        assert len(adapter._agent_cache) == 0
        agent2, _, _ = _acquire(adapter)
        assert agent2 is not agent1


class TestConcurrency:
    def test_concurrent_checkout_never_shares_an_instance(self):
        adapter = _make_adapter()
        created, _ = _instrument(adapter)

        agent1, key1, sig1 = _acquire(adapter)
        # Second request for the SAME session before the first releases —
        # must build fresh instead of sharing the checked-out agent.
        agent2, key2, sig2 = _acquire(adapter)
        assert agent2 is not agent1

        adapter._release_agent(key1, agent1, sig1)
        adapter._release_agent(key2, agent2, sig2)
        # Last one released wins the cache slot; the earlier entry for the
        # same key was overwritten, not leaked (single entry per key).
        assert len(adapter._agent_cache) == 1


class TestBounds:
    def test_lru_cap_evicts_and_cleans_oldest(self):
        adapter = _make_adapter()
        created, cleaned = _instrument(adapter)

        agents = []
        for i in range(adapter._AGENT_CACHE_MAX_SIZE + 1):
            agent, key, sig = _acquire(adapter, session_id=f"sess-{i}")
            agents.append((key, agent, sig))
        for key, agent, sig in agents:
            adapter._release_agent(key, agent, sig)

        assert len(adapter._agent_cache) == adapter._AGENT_CACHE_MAX_SIZE
        # The first-released entry is the LRU one and must be cleaned up.
        assert cleaned == [agents[0][1]]

    def test_missing_cache_key_bypasses_cache(self):
        adapter = _make_adapter()
        created, cleaned = _instrument(adapter)

        agent1, key, sig = _acquire(adapter, session_id=None)
        assert key == ""
        adapter._release_agent(key, agent1, sig)
        # Keyless agents are cleaned up, never cached.
        assert cleaned == [agent1]
        assert len(adapter._agent_cache) == 0


class TestOptOut:
    def test_agent_cache_disabled_via_extra(self):
        adapter = _make_adapter(agent_cache=False)
        created, cleaned = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)
        agent2, _, _ = _acquire(adapter)

        assert agent2 is not agent1
        assert cleaned == [agent1]

    def test_agent_cache_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_AGENT_CACHE", "false")
        adapter = _make_adapter()
        assert adapter._agent_cache_enabled is False

    def test_agent_cache_enabled_by_default(self):
        adapter = _make_adapter()
        assert adapter._agent_cache_enabled is True


class TestFailOpen:
    def test_resolution_failure_bypasses_cache(self):
        """Cache is an optimization — a config-resolution error must fall
        back to the historical build-per-request path, not fail the turn."""
        adapter = _make_adapter()
        created, cleaned = _instrument(adapter)
        adapter._resolve_agent_runtime = MagicMock(
            side_effect=RuntimeError("no provider configured")
        )

        agent, key, sig = _acquire(adapter)
        assert agent is created[0]
        assert key == ""
        assert sig == ""
        adapter._release_agent(key, agent, sig)
        # Bypassed agents are cleaned up, never cached.
        assert cleaned == [agent]
        assert len(adapter._agent_cache) == 0


class TestCleanup:
    def test_cleanup_calls_memory_shutdown_with_transcript(self):
        agent = MagicMock()
        agent._session_messages = [{"role": "user", "content": "hi"}]
        APIServerAdapter._cleanup_cached_agent(agent)
        agent.shutdown_memory_provider.assert_called_once_with(
            agent._session_messages
        )
        agent.close.assert_called_once()

    def test_cleanup_survives_agent_errors(self):
        agent = MagicMock()
        agent.shutdown_memory_provider.side_effect = RuntimeError("boom")
        agent.close.side_effect = RuntimeError("boom")
        # Must not raise.
        APIServerAdapter._cleanup_cached_agent(agent)

    def test_cleanup_handles_none(self):
        APIServerAdapter._cleanup_cached_agent(None)
