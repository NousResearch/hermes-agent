"""Tests for the API server agent cache (chat completions agent reuse).

The cache mirrors the gateway's per-session AIAgent cache
(gateway/run.py ``_agent_cache``): agents are checked OUT of the cache
for the duration of a run and checked back IN afterwards, keyed by
``gateway_session_key or session_id`` and validated by a config
signature.

Release semantics mirror the gateway's too: eviction, invalidation,
displacement and discard are SOFT (``release_clients()`` — the session
may resume and must keep its task-scoped tool state); FULL teardown
(``close()`` + memory-provider shutdown) happens only at server
shutdown.
"""

import time
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter(**extra):
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra or None))


def _instrument(adapter, signature="sig-1"):
    """Stub config resolution / construction / release on the adapter."""
    created = []
    soft_released = []

    def _fake_create(**kwargs):
        agent = MagicMock(name=f"agent-{len(created)}")
        created.append(agent)
        return agent

    adapter._resolve_agent_runtime = MagicMock(
        return_value={"max_iterations": 5}
    )
    adapter._agent_cache_signature = MagicMock(return_value=signature)
    adapter._create_agent = MagicMock(side_effect=_fake_create)
    adapter._soft_release_agent = MagicMock(side_effect=soft_released.append)
    return created, soft_released


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
        created, soft_released = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)
        agent2, key2, sig2 = _acquire(adapter)

        assert agent2 is agent1
        assert key2 == key
        assert len(created) == 1
        assert soft_released == []

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
    def test_config_signature_change_soft_releases_stale_agent(self):
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)

        adapter._agent_cache_signature = MagicMock(return_value="sig-2")
        agent2, _, _ = _acquire(adapter)

        assert agent2 is not agent1
        assert soft_released == [agent1]

    def test_idle_ttl_expiry_soft_releases_stale_agent(self):
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)

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
        assert soft_released == [agent1]

    def test_discard_on_release_soft_releases_and_skips_cache(self):
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig, discard=True)

        assert soft_released == [agent1]
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

    def test_same_key_double_release_soft_releases_displaced_agent(self):
        """Regression: two overlapping requests on one session release in
        sequence — the entry replaced by the second release must be
        deliberately soft-released, not silently overwritten (it still owns
        clients, memory-provider state and tool resources)."""
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)

        agent1, key1, sig1 = _acquire(adapter)
        agent2, key2, sig2 = _acquire(adapter)  # concurrent, builds fresh
        assert key1 == key2

        adapter._release_agent(key1, agent1, sig1)
        adapter._release_agent(key2, agent2, sig2)

        # agent1 was displaced by agent2's release and must be released.
        assert soft_released == [agent1]
        with adapter._agent_cache_lock:
            assert len(adapter._agent_cache) == 1
            assert adapter._agent_cache[key1][0] is agent2

    def test_same_agent_double_release_does_not_self_release(self):
        """Releasing the same instance twice (defensive) must not soft-release
        the very agent that stays cached."""
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)
        adapter._release_agent(key, agent1, sig)

        assert soft_released == []
        with adapter._agent_cache_lock:
            assert adapter._agent_cache[key][0] is agent1


class TestBounds:
    def test_lru_cap_evicts_and_soft_releases_oldest(self):
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)

        agents = []
        for i in range(adapter._AGENT_CACHE_MAX_SIZE + 1):
            agent, key, sig = _acquire(adapter, session_id=f"sess-{i}")
            agents.append((key, agent, sig))
        for key, agent, sig in agents:
            adapter._release_agent(key, agent, sig)

        assert len(adapter._agent_cache) == adapter._AGENT_CACHE_MAX_SIZE
        # The first-released entry is the LRU one and must be soft-released.
        assert soft_released == [agents[0][1]]

    def test_missing_cache_key_bypasses_cache(self):
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)

        agent1, key, sig = _acquire(adapter, session_id=None)
        assert key == ""
        adapter._release_agent(key, agent1, sig)
        # Keyless agents are soft-released, never cached.
        assert soft_released == [agent1]
        assert len(adapter._agent_cache) == 0


class TestOptOut:
    def test_agent_cache_disabled_via_extra(self):
        adapter = _make_adapter(agent_cache=False)
        created, soft_released = _instrument(adapter)

        agent1, key, sig = _acquire(adapter)
        adapter._release_agent(key, agent1, sig)
        agent2, _, _ = _acquire(adapter)

        assert agent2 is not agent1
        assert soft_released == [agent1]

    def test_agent_cache_enabled_by_default(self):
        adapter = _make_adapter()
        assert adapter._agent_cache_enabled is True

    def test_agent_cache_extra_accepts_string_false(self):
        adapter = _make_adapter(agent_cache="false")
        assert adapter._agent_cache_enabled is False


class TestFailOpen:
    def test_resolution_failure_bypasses_cache(self):
        """Cache is an optimization — a config-resolution error must fall
        back to the historical build-per-request path, not fail the turn."""
        adapter = _make_adapter()
        created, soft_released = _instrument(adapter)
        adapter._resolve_agent_runtime = MagicMock(
            side_effect=RuntimeError("no provider configured")
        )

        agent, key, sig = _acquire(adapter)
        assert agent is created[0]
        assert key == ""
        assert sig == ""
        adapter._release_agent(key, agent, sig)
        # Bypassed agents are soft-released, never cached.
        assert soft_released == [agent]
        assert len(adapter._agent_cache) == 0


class TestReleaseSemantics:
    def test_soft_release_frees_clients_but_preserves_session_state(self):
        agent = MagicMock()
        APIServerAdapter._soft_release_agent(agent)
        agent.release_clients.assert_called_once_with()
        agent.close.assert_not_called()
        agent.shutdown_memory_provider.assert_not_called()

    def test_soft_release_falls_back_to_full_cleanup_without_release_clients(self):
        agent = MagicMock(spec=["close", "shutdown_memory_provider"])
        agent._session_messages = None
        APIServerAdapter._soft_release_agent(agent)
        agent.close.assert_called_once()

    def test_soft_release_survives_agent_errors(self):
        agent = MagicMock()
        agent.release_clients.side_effect = RuntimeError("boom")
        APIServerAdapter._soft_release_agent(agent)  # must not raise

    def test_soft_release_handles_none(self):
        APIServerAdapter._soft_release_agent(None)

    def test_full_cleanup_calls_memory_shutdown_with_transcript(self):
        agent = MagicMock()
        agent._session_messages = [{"role": "user", "content": "hi"}]
        APIServerAdapter._cleanup_cached_agent(agent)
        agent.shutdown_memory_provider.assert_called_once_with(
            agent._session_messages
        )
        agent.close.assert_called_once()

    def test_full_cleanup_survives_agent_errors(self):
        agent = MagicMock()
        agent.shutdown_memory_provider.side_effect = RuntimeError("boom")
        agent.close.side_effect = RuntimeError("boom")
        APIServerAdapter._cleanup_cached_agent(agent)  # must not raise

    def test_full_cleanup_handles_none(self):
        APIServerAdapter._cleanup_cached_agent(None)
