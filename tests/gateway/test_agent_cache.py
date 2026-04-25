"""Integration tests for gateway AIAgent caching.

Verifies that the agent cache correctly:
- Reuses agents across messages (same config → same instance)
- Rebuilds agents when config changes (model, provider, toolsets)
- Updates reasoning_config in-place without rebuilding
- Evicts on session reset
- Evicts on fallback activation
- Preserves frozen system prompt across turns
"""

import hashlib
import json
import threading
from unittest.mock import MagicMock, patch

import pytest


def _make_runner():
    """Create a minimal GatewayRunner with just the cache infrastructure."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


class TestAgentConfigSignature:
    """Config signature produces stable, distinct keys."""

    def test_same_config_same_signature(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter", "api_mode": "chat_completions"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        assert sig1 == sig2

    def test_model_change_different_signature(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-opus-4.6", runtime, ["hermes-telegram"], "")
        assert sig1 != sig2

    def test_same_token_prefix_different_full_token_changes_signature(self):
        """Tokens sharing a JWT-style prefix must not collide."""
        from gateway.run import GatewayRunner

        rt1 = {
            "api_key": "eyJhbGci.token-for-account-a",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
        }
        rt2 = {
            "api_key": "eyJhbGci.token-for-account-b",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
        }

        assert rt1["api_key"][:8] == rt2["api_key"][:8]
        sig1 = GatewayRunner._agent_config_signature("gpt-5.3-codex", rt1, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("gpt-5.3-codex", rt2, ["hermes-telegram"], "")
        assert sig1 != sig2

    def test_provider_change_different_signature(self):
        from gateway.run import GatewayRunner

        rt1 = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter"}
        rt2 = {"api_key": "sk-test12345678", "base_url": "https://api.anthropic.com", "provider": "anthropic"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", rt1, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", rt2, ["hermes-telegram"], "")
        assert sig1 != sig2

    def test_toolset_change_different_signature(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter"}
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-discord"], "")
        assert sig1 != sig2

    def test_reasoning_not_in_signature(self):
        """Reasoning config is set per-message, not part of the signature."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "sk-test12345678", "base_url": "https://openrouter.ai/api/v1", "provider": "openrouter"}
        # Same config — signature should be identical regardless of what
        # reasoning_config the caller might have (it's not passed in)
        sig1 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        sig2 = GatewayRunner._agent_config_signature("claude-sonnet-4", runtime, ["hermes-telegram"], "")
        assert sig1 == sig2


class TestAgentCacheLifecycle:
    """End-to-end cache behavior with real AIAgent construction."""

    def test_cache_hit_returns_same_agent(self):
        """Second message with same config reuses the cached agent instance."""
        from run_agent import AIAgent

        runner = _make_runner()
        session_key = "telegram:12345"
        runtime = {"api_key": "test", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter", "api_mode": "chat_completions"}
        sig = runner._agent_config_signature("anthropic/claude-sonnet-4", runtime, ["hermes-telegram"], "")

        # First message — create and cache
        agent1 = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="telegram",
        )
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent1, sig)

        # Second message — cache hit
        with runner._agent_cache_lock:
            cached = runner._agent_cache.get(session_key)
        assert cached is not None
        assert cached[1] == sig
        assert cached[0] is agent1  # same instance

    def test_cache_miss_on_model_change(self):
        """Model change produces different signature → cache miss."""
        from run_agent import AIAgent

        runner = _make_runner()
        session_key = "telegram:12345"
        runtime = {"api_key": "test", "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter", "api_mode": "chat_completions"}

        old_sig = runner._agent_config_signature("anthropic/claude-sonnet-4", runtime, ["hermes-telegram"], "")
        agent1 = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="telegram",
        )
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent1, old_sig)

        # New model → different signature
        new_sig = runner._agent_config_signature("anthropic/claude-opus-4.6", runtime, ["hermes-telegram"], "")
        assert new_sig != old_sig

        with runner._agent_cache_lock:
            cached = runner._agent_cache.get(session_key)
        assert cached[1] != new_sig  # signature mismatch → would create new agent

    def test_evict_on_session_reset(self):
        """_evict_cached_agent removes the entry."""
        from run_agent import AIAgent

        runner = _make_runner()
        session_key = "telegram:12345"

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True,
        )
        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent, "sig123")

        runner._evict_cached_agent(session_key)

        with runner._agent_cache_lock:
            assert session_key not in runner._agent_cache

    def test_evict_does_not_affect_other_sessions(self):
        """Evicting one session leaves other sessions cached."""
        runner = _make_runner()
        with runner._agent_cache_lock:
            runner._agent_cache["session-A"] = ("agent-A", "sig-A")
            runner._agent_cache["session-B"] = ("agent-B", "sig-B")

        runner._evict_cached_agent("session-A")

        with runner._agent_cache_lock:
            assert "session-A" not in runner._agent_cache
            assert "session-B" in runner._agent_cache

    def test_reasoning_config_updates_in_place(self):
        """Reasoning config can be set on a cached agent without eviction."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True,
            reasoning_config={"enabled": True, "effort": "medium"},
        )

        # Simulate per-message reasoning update
        agent.reasoning_config = {"enabled": True, "effort": "high"}
        assert agent.reasoning_config["effort"] == "high"

        # System prompt should not be affected by reasoning change
        prompt1 = agent._build_system_prompt()
        agent._cached_system_prompt = prompt1  # simulate run_conversation caching
        agent.reasoning_config = {"enabled": True, "effort": "low"}
        prompt2 = agent._cached_system_prompt
        assert prompt1 is prompt2  # same object — not invalidated by reasoning change

    def test_system_prompt_frozen_across_cache_reuse(self):
        """The cached agent's system prompt stays identical across turns."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="telegram",
        )

        # Build system prompt (simulates first run_conversation)
        prompt1 = agent._build_system_prompt()
        agent._cached_system_prompt = prompt1

        # Simulate second turn — prompt should be frozen
        prompt2 = agent._cached_system_prompt
        assert prompt1 is prompt2  # same object, not rebuilt

    def test_callbacks_update_without_cache_eviction(self):
        """Per-message callbacks can be set on cached agent."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True, skip_context_files=True,
            skip_memory=True,
        )

        # Set callbacks like the gateway does per-message
        cb1 = lambda *a: None
        cb2 = lambda *a: None
        agent.tool_progress_callback = cb1
        agent.step_callback = cb2
        agent.stream_delta_callback = None
        agent.status_callback = None

        assert agent.tool_progress_callback is cb1
        assert agent.step_callback is cb2

        # Update for next message
        cb3 = lambda *a: None
        agent.tool_progress_callback = cb3
        assert agent.tool_progress_callback is cb3


class TestAgentCacheBoundedGrowth:
    """LRU cap and idle-TTL eviction prevent unbounded cache growth."""

    def _bounded_runner(self):
        """Runner with an OrderedDict cache (matches real gateway init)."""
        from collections import OrderedDict
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        return runner

    def _fake_agent(self, last_activity: float | None = None):
        """Lightweight stand-in; real AIAgent is heavy to construct."""
        m = MagicMock()
        if last_activity is not None:
            m._last_activity_ts = last_activity
        else:
            import time as _t
            m._last_activity_ts = _t.time()
        return m

    def test_cap_evicts_lru_when_exceeded(self, monkeypatch):
        """Inserting past _AGENT_CACHE_MAX_SIZE pops the oldest entry."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 3)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        for i in range(3):
            runner._agent_cache[f"s{i}"] = (self._fake_agent(), f"sig{i}")

        # Insert a 4th — oldest (s0) must be evicted.
        with runner._agent_cache_lock:
            runner._agent_cache["s3"] = (self._fake_agent(), "sig3")
            runner._enforce_agent_cache_cap()

        assert "s0" not in runner._agent_cache
        assert "s3" in runner._agent_cache
        assert len(runner._agent_cache) == 3

    def test_cap_respects_move_to_end(self, monkeypatch):
        """Entries refreshed via move_to_end are NOT evicted as 'oldest'."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 3)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        for i in range(3):
            runner._agent_cache[f"s{i}"] = (self._fake_agent(), f"sig{i}")

        # Touch s0 — it is now MRU, so s1 becomes LRU.
        runner._agent_cache.move_to_end("s0")

        with runner._agent_cache_lock:
            runner._agent_cache["s3"] = (self._fake_agent(), "sig3")
            runner._enforce_agent_cache_cap()

        assert "s0" in runner._agent_cache  # rescued by move_to_end
        assert "s1" not in runner._agent_cache  # now oldest → evicted
        assert "s3" in runner._agent_cache

    def test_cap_triggers_cleanup_thread(self, monkeypatch):
        """Evicted agent has release_clients() called for it (soft cleanup).

        Uses the soft path (_release_evicted_agent_soft), NOT the hard
        _cleanup_agent_resources — cache eviction must not tear down
        per-task state (terminal/browser/bg procs).
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
        runner = self._bounded_runner()

        release_calls: list = []
        cleanup_calls: list = []
        # Intercept both paths; only release_clients path should fire.
        def _soft(agent):
            release_calls.append(agent)
        runner._release_evicted_agent_soft = _soft
        runner._cleanup_agent_resources = lambda a: cleanup_calls.append(a)

        old_agent = self._fake_agent()
        new_agent = self._fake_agent()
        with runner._agent_cache_lock:
            runner._agent_cache["old"] = (old_agent, "sig_old")
            runner._agent_cache["new"] = (new_agent, "sig_new")
            runner._enforce_agent_cache_cap()

        # Cleanup is dispatched to a daemon thread; join briefly to observe.
        import time as _t
        deadline = _t.time() + 2.0
        while _t.time() < deadline and not release_calls:
            _t.sleep(0.02)
        assert old_agent in release_calls
        assert new_agent not in release_calls
        # Hard-cleanup path must NOT have fired — that's for session expiry only.
        assert cleanup_calls == []

    def test_idle_ttl_sweep_evicts_stale_agents(self, monkeypatch):
        """_sweep_idle_cached_agents removes agents idle past the TTL."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.05)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        import time as _t
        fresh = self._fake_agent(last_activity=_t.time())
        stale = self._fake_agent(last_activity=_t.time() - 10.0)
        runner._agent_cache["fresh"] = (fresh, "s1")
        runner._agent_cache["stale"] = (stale, "s2")

        evicted = runner._sweep_idle_cached_agents()
        assert evicted == 1
        assert "stale" not in runner._agent_cache
        assert "fresh" in runner._agent_cache

    def test_idle_sweep_skips_agents_without_activity_ts(self, monkeypatch):
        """Agents missing _last_activity_ts are left alone (defensive)."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        no_ts = MagicMock(spec=[])  # no _last_activity_ts attribute
        runner._agent_cache["s"] = (no_ts, "sig")

        assert runner._sweep_idle_cached_agents() == 0
        assert "s" in runner._agent_cache

    def test_plain_dict_cache_is_tolerated(self):
        """Test fixtures using plain {} don't crash _enforce_agent_cache_cap."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = {}  # plain dict, not OrderedDict
        runner._agent_cache_lock = threading.Lock()
        runner._cleanup_agent_resources = MagicMock()

        # Should be a no-op rather than raising.
        with runner._agent_cache_lock:
            for i in range(200):
                runner._agent_cache[f"s{i}"] = (MagicMock(), f"sig{i}")
            runner._enforce_agent_cache_cap()  # no crash, no eviction

        assert len(runner._agent_cache) == 200

    def test_main_lookup_updates_lru_order(self, monkeypatch):
        """Cache hit via the main-lookup path refreshes LRU position."""
        runner = self._bounded_runner()

        a0 = self._fake_agent()
        a1 = self._fake_agent()
        a2 = self._fake_agent()
        runner._agent_cache["s0"] = (a0, "sig0")
        runner._agent_cache["s1"] = (a1, "sig1")
        runner._agent_cache["s2"] = (a2, "sig2")

        # Simulate what _process_message_background does on a cache hit
        # (minus the agent-state reset which isn't relevant here).
        with runner._agent_cache_lock:
            cached = runner._agent_cache.get("s0")
            if cached and hasattr(runner._agent_cache, "move_to_end"):
                runner._agent_cache.move_to_end("s0")

        # After the hit, insertion order should be s1, s2, s0.
        assert list(runner._agent_cache.keys()) == ["s1", "s2", "s0"]


class TestAgentCacheActiveSafety:
    """Safety: eviction must not tear down agents currently mid-turn.

    AIAgent.close() kills process_registry entries for the task, cleans
    the terminal sandbox, closes the OpenAI client, and cascades
    .close() into active child subagents.  Calling it while the agent
    is still processing would crash the in-flight request.  These tests
    pin that eviction skips any agent present in _running_agents.
    """

    def _runner(self):
        from collections import OrderedDict
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        return runner

    def _fake_agent(self, idle_seconds: float = 0.0):
        import time as _t
        m = MagicMock()
        m._last_activity_ts = _t.time() - idle_seconds
        return m

    def test_cap_skips_active_lru_entry(self, monkeypatch):
        """Active LRU entry is skipped; cache stays over cap rather than
        compensating by evicting a newer entry.

        Rationale: evicting a more-recent entry just because the oldest
        slot is temporarily locked would punish the most recently-
        inserted session (which has no cache to preserve) to protect
        one that happens to be mid-turn.  Better to let the cache stay
        transiently over cap and re-check on the next insert.
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 2)
        runner = self._runner()
        runner._cleanup_agent_resources = MagicMock()

        active = self._fake_agent()
        idle_a = self._fake_agent()
        idle_b = self._fake_agent()

        # Insertion order: active (oldest), idle_a, idle_b.
        runner._agent_cache["session-active"] = (active, "sig")
        runner._agent_cache["session-idle-a"] = (idle_a, "sig")
        runner._agent_cache["session-idle-b"] = (idle_b, "sig")

        # Mark `active` as mid-turn — it's LRU, but protected.
        runner._running_agents["session-active"] = active

        with runner._agent_cache_lock:
            runner._enforce_agent_cache_cap()

        # All three remain; no eviction ran, no cleanup dispatched.
        assert "session-active" in runner._agent_cache
        assert "session-idle-a" in runner._agent_cache
        assert "session-idle-b" in runner._agent_cache
        assert runner._cleanup_agent_resources.call_count == 0

    def test_cap_evicts_when_multiple_excess_and_some_inactive(self, monkeypatch):
        """Mixed active/idle in the LRU excess window: only the idle ones go.

        With CAP=2 and 4 entries, excess=2 (the two oldest).  If the
        oldest is active and the next is idle, we evict exactly one.
        Cache ends at CAP+1, which is still better than unbounded.
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 2)
        runner = self._runner()
        runner._cleanup_agent_resources = MagicMock()

        oldest_active = self._fake_agent()
        idle_second = self._fake_agent()
        idle_third = self._fake_agent()
        idle_fourth = self._fake_agent()

        runner._agent_cache["s1"] = (oldest_active, "sig")
        runner._agent_cache["s2"] = (idle_second, "sig")  # in excess window, idle
        runner._agent_cache["s3"] = (idle_third, "sig")
        runner._agent_cache["s4"] = (idle_fourth, "sig")

        runner._running_agents["s1"] = oldest_active  # oldest is mid-turn

        with runner._agent_cache_lock:
            runner._enforce_agent_cache_cap()

        # s1 protected (active), s2 evicted (idle + in excess window),
        # s3 and s4 untouched (outside excess window).
        assert "s1" in runner._agent_cache
        assert "s2" not in runner._agent_cache
        assert "s3" in runner._agent_cache
        assert "s4" in runner._agent_cache

    def test_cap_leaves_cache_over_limit_if_all_active(self, monkeypatch, caplog):
        """If every over-cap entry is mid-turn, the cache stays over cap.

        Better to temporarily exceed the cap than to crash an in-flight
        turn by tearing down its clients.
        """
        from gateway import run as gw_run
        import logging as _logging

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
        runner = self._runner()
        runner._cleanup_agent_resources = MagicMock()

        a1 = self._fake_agent()
        a2 = self._fake_agent()
        a3 = self._fake_agent()
        runner._agent_cache["s1"] = (a1, "sig")
        runner._agent_cache["s2"] = (a2, "sig")
        runner._agent_cache["s3"] = (a3, "sig")

        # All three are mid-turn.
        runner._running_agents["s1"] = a1
        runner._running_agents["s2"] = a2
        runner._running_agents["s3"] = a3

        with caplog.at_level(_logging.WARNING, logger="gateway.run"):
            with runner._agent_cache_lock:
                runner._enforce_agent_cache_cap()

        # Cache unchanged because eviction had to skip every candidate.
        assert len(runner._agent_cache) == 3
        # _cleanup_agent_resources must NOT have been scheduled.
        assert runner._cleanup_agent_resources.call_count == 0
        # And we logged a warning so operators can see the condition.
        assert any("mid-turn" in r.message for r in caplog.records)

    def test_cap_pending_sentinel_does_not_block_eviction(self, monkeypatch):
        """_AGENT_PENDING_SENTINEL in _running_agents is treated as 'not active'.

        The sentinel is set while an agent is being CONSTRUCTED, before the
        real AIAgent instance exists.  Cached agents from other sessions
        can still be evicted safely.
        """
        from gateway import run as gw_run
        from gateway.run import _AGENT_PENDING_SENTINEL

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
        runner = self._runner()
        runner._cleanup_agent_resources = MagicMock()

        a1 = self._fake_agent()
        a2 = self._fake_agent()
        runner._agent_cache["s1"] = (a1, "sig")
        runner._agent_cache["s2"] = (a2, "sig")
        # Another session is mid-creation — sentinel, no real agent yet.
        runner._running_agents["s3-being-created"] = _AGENT_PENDING_SENTINEL

        with runner._agent_cache_lock:
            runner._enforce_agent_cache_cap()

        assert "s1" not in runner._agent_cache  # evicted normally
        assert "s2" in runner._agent_cache

    def test_idle_sweep_skips_active_agent(self, monkeypatch):
        """Idle-TTL sweep must not tear down an active agent even if 'stale'."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
        runner = self._runner()
        runner._cleanup_agent_resources = MagicMock()

        old_but_active = self._fake_agent(idle_seconds=10.0)
        runner._agent_cache["s1"] = (old_but_active, "sig")
        runner._running_agents["s1"] = old_but_active

        evicted = runner._sweep_idle_cached_agents()

        assert evicted == 0
        assert "s1" in runner._agent_cache
        assert runner._cleanup_agent_resources.call_count == 0

    def test_eviction_does_not_close_active_agent_client(self, monkeypatch):
        """Live test: evicting an active agent does NOT null its .client.

        This reproduces the original concern — if eviction fired while an
        agent was mid-turn, `agent.close()` would set `self.client = None`
        and the next API call inside the loop would crash.  With the
        active-agent skip, the client stays intact.
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
        runner = self._runner()

        # Build a proper fake agent whose close() matches AIAgent's contract.
        active = MagicMock()
        active._last_activity_ts = __import__("time").time()
        active.client = MagicMock()  # simulate an OpenAI client
        def _real_close():
            active.client = None  # mirrors run_agent.py:3299
        active.close = _real_close
        active.shutdown_memory_provider = MagicMock()

        idle = self._fake_agent()

        runner._agent_cache["active-session"] = (active, "sig")
        runner._agent_cache["idle-session"] = (idle, "sig")
        runner._running_agents["active-session"] = active

        # Real cleanup function, not mocked — we want to see whether close()
        # runs on the active agent.  (It shouldn't.)
        with runner._agent_cache_lock:
            runner._enforce_agent_cache_cap()

        # Let any eviction cleanup threads drain.
        import time as _t
        _t.sleep(0.2)

        # The ACTIVE agent's client must still be usable.
        assert active.client is not None, (
            "Active agent's client was closed by eviction — "
            "running turn would crash on its next API call."
        )


class TestAgentCacheSpilloverLive:
    """Live E2E: fill cache with real AIAgent instances and stress it."""

    def _runner(self):
        from collections import OrderedDict
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        return runner

    def _real_agent(self):
        """A genuine AIAgent; no API calls are made during these tests."""
        from run_agent import AIAgent
        return AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            platform="telegram",
        )

    def test_fill_to_cap_then_spillover(self, monkeypatch):
        """Fill to cap with real agents, insert one more, oldest evicted."""
        from gateway import run as gw_run

        CAP = 8
        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", CAP)
        runner = self._runner()

        agents = [self._real_agent() for _ in range(CAP)]
        for i, a in enumerate(agents):
            with runner._agent_cache_lock:
                runner._agent_cache[f"s{i}"] = (a, "sig")
                runner._enforce_agent_cache_cap()
        assert len(runner._agent_cache) == CAP

        # Spillover insertion.
        newcomer = self._real_agent()
        with runner._agent_cache_lock:
            runner._agent_cache["new"] = (newcomer, "sig")
            runner._enforce_agent_cache_cap()

        # Oldest (s0) evicted, cap still CAP.
        assert "s0" not in runner._agent_cache
        assert "new" in runner._agent_cache
        assert len(runner._agent_cache) == CAP

        # Clean up so pytest doesn't leak resources.
        for a in agents + [newcomer]:
            try:
                a.close()
            except Exception:
                pass

    def test_spillover_all_active_keeps_cache_over_cap(self, monkeypatch, caplog):
        """Every slot active: cache goes over cap, no one gets torn down."""
        from gateway import run as gw_run
        import logging as _logging

        CAP = 4
        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", CAP)
        runner = self._runner()

        agents = [self._real_agent() for _ in range(CAP)]
        for i, a in enumerate(agents):
            runner._agent_cache[f"s{i}"] = (a, "sig")
            runner._running_agents[f"s{i}"] = a  # every session mid-turn

        newcomer = self._real_agent()
        with caplog.at_level(_logging.WARNING, logger="gateway.run"):
            with runner._agent_cache_lock:
                runner._agent_cache["new"] = (newcomer, "sig")
                runner._enforce_agent_cache_cap()

        assert len(runner._agent_cache) == CAP + 1  # temporarily over cap
        # All existing agents still usable.
        for i, a in enumerate(agents):
            assert a.client is not None, f"s{i} got closed while active!"
        # And we warned operators.
        assert any("mid-turn" in r.message for r in caplog.records)

        for a in agents + [newcomer]:
            try:
                a.close()
            except Exception:
                pass

    def test_concurrent_inserts_settle_at_cap(self, monkeypatch):
        """Many threads inserting in parallel end with len(cache) == CAP."""
        from gateway import run as gw_run

        CAP = 16
        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", CAP)
        runner = self._runner()

        N_THREADS = 8
        PER_THREAD = 20  # 8 * 20 = 160 inserts into a 16-slot cache

        def worker(tid: int):
            for j in range(PER_THREAD):
                a = self._real_agent()
                key = f"t{tid}-s{j}"
                with runner._agent_cache_lock:
                    runner._agent_cache[key] = (a, "sig")
                    runner._enforce_agent_cache_cap()

        threads = [
            threading.Thread(target=worker, args=(t,), daemon=True)
            for t in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Worker thread hung — possible deadlock?"

        # Let daemon cleanup threads settle.
        import time as _t
        _t.sleep(0.5)

        assert len(runner._agent_cache) == CAP, (
            f"Expected exactly {CAP} entries after concurrent inserts, "
            f"got {len(runner._agent_cache)}."
        )

    def test_evicted_session_next_turn_gets_fresh_agent(self, monkeypatch):
        """After eviction, the same session_key can insert a fresh agent.

        Simulates the real spillover flow: evicted session sends another
        message, which builds a new AIAgent and re-enters the cache.
        """
        from gateway import run as gw_run

        CAP = 2
        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", CAP)
        runner = self._runner()

        a0 = self._real_agent()
        a1 = self._real_agent()
        runner._agent_cache["sA"] = (a0, "sig")
        runner._agent_cache["sB"] = (a1, "sig")

        # 3rd session forces sA (oldest) out.
        a2 = self._real_agent()
        with runner._agent_cache_lock:
            runner._agent_cache["sC"] = (a2, "sig")
            runner._enforce_agent_cache_cap()
        assert "sA" not in runner._agent_cache

        # Let the eviction cleanup thread run.
        import time as _t
        _t.sleep(0.3)

        # Now sA's user sends another message → a fresh agent goes in.
        a0_new = self._real_agent()
        with runner._agent_cache_lock:
            runner._agent_cache["sA"] = (a0_new, "sig")
            runner._enforce_agent_cache_cap()

        assert "sA" in runner._agent_cache
        assert runner._agent_cache["sA"][0] is a0_new  # the new one, not stale
        # Fresh agent is usable.
        assert a0_new.client is not None

        for a in (a0, a1, a2, a0_new):
            try:
                a.close()
            except Exception:
                pass


class TestAgentCacheIdleResume:
    """End-to-end: idle-TTL-evicted session resumes cleanly with task state.

    Real-world scenario: user leaves a Telegram session open for 2+ hours.
    Idle-TTL evicts their cached agent.  They come back and send a message.
    The new agent built for the same session_id must inherit:
      - Conversation history (from SessionStore — outside cache concern)
      - Terminal sandbox (same task_id → same _active_environments entry)
      - Browser daemon (same task_id → same browser session)
      - Background processes (same task_id → same process_registry entries)
    The ONLY thing that should reset is the LLM client pool (rebuilt fresh).
    """

    def _runner(self):
        from collections import OrderedDict
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        return runner

    def test_release_clients_does_not_touch_process_registry(self, monkeypatch):
        """release_clients must not call process_registry.kill_all for task_id."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            session_id="idle-resume-test-session",
        )

        # Spy on process_registry.kill_all — it MUST NOT be called.
        from tools import process_registry as _pr
        kill_all_calls: list = []
        original_kill_all = _pr.process_registry.kill_all
        _pr.process_registry.kill_all = lambda **kw: kill_all_calls.append(kw)
        try:
            agent.release_clients()
        finally:
            _pr.process_registry.kill_all = original_kill_all
            try:
                agent.close()
            except Exception:
                pass

        assert kill_all_calls == [], (
            f"release_clients() called process_registry.kill_all — would "
            f"kill user's bg processes on cache eviction. Calls: {kill_all_calls}"
        )

    def test_release_clients_does_not_touch_terminal_or_browser(self, monkeypatch):
        """release_clients must not call cleanup_vm or cleanup_browser."""
        from run_agent import AIAgent
        from tools import terminal_tool as _tt
        from tools import browser_tool as _bt

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            session_id="idle-resume-test-2",
        )

        vm_calls: list = []
        browser_calls: list = []
        original_vm = _tt.cleanup_vm
        original_browser = _bt.cleanup_browser
        _tt.cleanup_vm = lambda tid: vm_calls.append(tid)
        _bt.cleanup_browser = lambda tid: browser_calls.append(tid)
        try:
            agent.release_clients()
        finally:
            _tt.cleanup_vm = original_vm
            _bt.cleanup_browser = original_browser
            try:
                agent.close()
            except Exception:
                pass

        assert vm_calls == [], (
            f"release_clients() tore down terminal sandbox — user's cwd, "
            f"env, and bg shells would be gone on resume. Calls: {vm_calls}"
        )
        assert browser_calls == [], (
            f"release_clients() tore down browser session — user's open "
            f"tabs and cookies gone on resume. Calls: {browser_calls}"
        )

    def test_release_clients_closes_llm_client(self):
        """release_clients IS expected to close the OpenAI/httpx client."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
        # Clients are lazy-built; force one to exist so we can verify close.
        assert agent.client is not None  # __init__ builds it

        agent.release_clients()

        # Post-release: client reference is dropped (memory freed).
        assert agent.client is None

    def test_close_vs_release_full_teardown_difference(self, monkeypatch):
        """close() tears down task state; release_clients() does not.

        This pins the semantic contract: session-expiry path uses close()
        (full teardown — session is done), cache-eviction path uses
        release_clients() (soft — session may resume).
        """
        from run_agent import AIAgent
        import run_agent as _ra

        # Agent A: evicted from cache (soft) — terminal survives.
        # Agent B: session expired (hard) — terminal torn down.
        agent_a = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            session_id="soft-session",
        )
        agent_b = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            session_id="hard-session",
        )

        vm_calls: list = []
        # AIAgent.close() calls the ``cleanup_vm`` name bound into
        # ``run_agent`` at import time, not ``tools.terminal_tool.cleanup_vm``
        # directly — so patch the ``run_agent`` reference.
        original_vm = _ra.cleanup_vm
        _ra.cleanup_vm = lambda tid: vm_calls.append(tid)
        try:
            agent_a.release_clients()   # cache eviction
            agent_b.close()              # session expiry
        finally:
            _ra.cleanup_vm = original_vm
            try:
                agent_a.close()
            except Exception:
                pass

        # Only agent_b's task_id should appear in cleanup calls.
        assert "hard-session" in vm_calls
        assert "soft-session" not in vm_calls

    def test_idle_evicted_session_rebuild_inherits_task_id(self, monkeypatch):
        """After idle-TTL eviction, a fresh agent with the same session_id
        gets the same task_id — so tool state (terminal/browser/bg procs)
        that persisted across eviction is reachable via the new agent.
        """
        from gateway import run as gw_run
        from run_agent import AIAgent

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
        runner = self._runner()

        # Build an agent representing a stale (idle) session.
        SESSION_ID = "long-lived-user-session"
        old = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            session_id=SESSION_ID,
        )
        old._last_activity_ts = 0.0  # force idle
        runner._agent_cache["sKey"] = (old, "sig")

        # Simulate the idle-TTL sweep firing.
        runner._sweep_idle_cached_agents()
        assert "sKey" not in runner._agent_cache

        # Wait for the daemon thread doing release_clients() to finish.
        import time as _t
        _t.sleep(0.3)

        # Old agent's client is gone (soft cleanup fired).
        assert old.client is None

        # User comes back — new agent built for the SAME session_id.
        new_agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            session_id=SESSION_ID,
        )

        # Same session_id means same task_id routed to tools.  The new
        # agent inherits any per-task state (terminal sandbox etc.) that
        # was preserved across eviction.
        assert new_agent.session_id == old.session_id == SESSION_ID
        # And it has a fresh working client.
        assert new_agent.client is not None

        try:
            new_agent.close()
        except Exception:
            pass
