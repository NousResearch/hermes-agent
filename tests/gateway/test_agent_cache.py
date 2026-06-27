"""Integration tests for gateway AIAgent caching.

Verifies that the agent cache correctly:
- Reuses agents across messages (same config → same instance)
- Rebuilds agents when config changes (model, provider, toolsets)
- Updates reasoning_config in-place without rebuilding
- Evicts on session reset
- Evicts on fallback activation
- Preserves frozen system prompt across turns
"""

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

    # ---------------------------------------------------------------
    # cache_keys (compression/context config cache-busting)
    # ---------------------------------------------------------------

    def test_cache_keys_default_omitted_matches_empty(self):
        """Omitted cache_keys must produce the same signature as empty {}."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        sig_omitted = GatewayRunner._agent_config_signature("m", runtime, [], "")
        sig_empty = GatewayRunner._agent_config_signature("m", runtime, [], "", cache_keys={})
        sig_none = GatewayRunner._agent_config_signature("m", runtime, [], "", cache_keys=None)
        assert sig_omitted == sig_empty == sig_none

    def test_context_length_change_busts_cache(self):
        """Editing model.context_length in config must produce a new signature."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        sig1 = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"model.context_length": 200_000},
        )
        sig2 = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"model.context_length": 400_000},
        )
        assert sig1 != sig2

    def test_max_tokens_change_busts_cache(self):
        """Editing model.max_tokens in config must produce a new signature."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        sig1 = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"model.max_tokens": 4096},
        )
        sig2 = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"model.max_tokens": 8192},
        )
        assert sig1 != sig2

    def test_compression_threshold_change_busts_cache(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        sig1 = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"compression.threshold": 0.50},
        )
        sig2 = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"compression.threshold": 0.75},
        )
        assert sig1 != sig2

    def test_compression_enabled_toggle_busts_cache(self):
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        sig_on = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"compression.enabled": True},
        )
        sig_off = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"compression.enabled": False},
        )
        assert sig_on != sig_off

    def test_cache_keys_key_order_does_not_matter(self):
        """Signature must be stable regardless of dict key insertion order."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        sig_a = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"model.context_length": 200_000, "compression.threshold": 0.5},
        )
        sig_b = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys={"compression.threshold": 0.5, "model.context_length": 200_000},
        )
        assert sig_a == sig_b

    def test_tool_registry_generation_change_busts_cache(self):
        """MCP reloads mutate the tool registry, so cached agents must rebuild."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        sig_before = GatewayRunner._agent_config_signature(
            "m", runtime, ["telegram"], "",
            cache_keys={"tools.registry_generation": 10},
        )
        sig_after = GatewayRunner._agent_config_signature(
            "m", runtime, ["telegram"], "",
            cache_keys={"tools.registry_generation": 11},
        )

        assert sig_before != sig_after


class TestExtractCacheBustingConfig:
    """Verify _extract_cache_busting_config pulls the documented subset of
    config values that must invalidate the cached agent on change."""

    def test_reads_model_context_length(self):
        from gateway.run import GatewayRunner

        out = GatewayRunner._extract_cache_busting_config(
            {
                "model": {
                    "context_length": 272_000,
                    "max_tokens": 4096,
                    "provider": "openrouter",
                }
            }
        )
        assert out["model.context_length"] == 272_000
        assert out["model.max_tokens"] == 4096

    def test_reads_compression_subkeys(self):
        from gateway.run import GatewayRunner

        out = GatewayRunner._extract_cache_busting_config(
            {
                "compression": {
                    "enabled": False,
                    "threshold": 0.6,
                    "codex_gpt55_autoraise": False,
                    "codex_native_compaction": True,
                    "codex_responses_threshold": 0.85,
                    "target_ratio": 0.3,
                    "protect_last_n": 25,
                    "codex_app_server_auto": "hermes",
                    "some_other_key": "ignored",
                }
            }
        )
        assert out["compression.enabled"] is False
        assert out["compression.threshold"] == 0.6
        assert out["compression.codex_gpt55_autoraise"] is False
        assert out["compression.codex_native_compaction"] is True
        assert out["compression.codex_responses_threshold"] == 0.85
        assert out["compression.target_ratio"] == 0.3
        assert out["compression.protect_last_n"] == 25
        assert out["compression.codex_app_server_auto"] == "hermes"

    def test_missing_keys_yield_none(self):
        """Absent config keys must produce None values (still contribute to signature)."""
        from gateway.run import GatewayRunner

        out = GatewayRunner._extract_cache_busting_config({})
        # Every documented cache-busting key must be present, even if None
        for section, key in GatewayRunner._CACHE_BUSTING_CONFIG_KEYS:
            assert f"{section}.{key}" in out
            assert out[f"{section}.{key}"] is None

    def test_non_dict_section_treated_as_missing(self):
        from gateway.run import GatewayRunner

        # compression is a string — should not crash, all compression.* keys None
        out = GatewayRunner._extract_cache_busting_config(
            {"compression": "broken", "model": {"context_length": 100_000}}
        )
        assert out["compression.enabled"] is None
        assert out["compression.threshold"] is None
        assert out["model.context_length"] == 100_000

    def test_none_config_is_safe(self):
        from gateway.run import GatewayRunner

        out = GatewayRunner._extract_cache_busting_config(None)
        for section, key in GatewayRunner._CACHE_BUSTING_CONFIG_KEYS:
            assert out[f"{section}.{key}"] is None
        assert "tools.registry_generation" in out

    def test_extract_includes_live_tool_registry_generation(self, monkeypatch):
        from gateway.run import GatewayRunner
        from tools.registry import registry

        monkeypatch.setattr(registry, "_generation", 12345)

        out = GatewayRunner._extract_cache_busting_config({})

        assert out["tools.registry_generation"] == 12345


    def test_skips_honcho_config_read_when_provider_is_not_honcho(self, monkeypatch):
        """Non-Honcho gateways must not read/parse honcho.json on every message."""
        from gateway.run import GatewayRunner

        called = False

        def _boom():
            nonlocal called
            called = True
            raise AssertionError("should not read Honcho config")

        monkeypatch.setattr(GatewayRunner, "_extract_honcho_cache_busting_config", _boom)

        out = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "mem0"}})

        assert called is False
        assert out["honcho.peer_name"] is None
        assert out["honcho.user_peer_aliases"] is None

    def test_reads_honcho_config_only_when_provider_is_honcho(self, monkeypatch):
        from gateway.run import GatewayRunner

        calls = []

        def _fake():
            calls.append(True)
            return {
                "honcho.peer_name": "eri",
                "honcho.ai_peer": "hermes",
                "honcho.pin_peer_name": True,
                "honcho.runtime_peer_prefix": "tg_",
                "honcho.user_peer_aliases": [("123", "eri")],
            }

        monkeypatch.setattr(GatewayRunner, "_extract_honcho_cache_busting_config", _fake)

        out = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})

        assert calls == [True]
        assert out["honcho.peer_name"] == "eri"
        assert out["honcho.user_peer_aliases"] == [("123", "eri")]

    def test_memory_provider_change_busts_signature(self, monkeypatch):
        """Switching memory.provider must itself change the cache-busting
        signature, so the agent is rebuilt when a user swaps providers
        mid-gateway (independent of the honcho.json identity keys)."""
        from gateway.run import GatewayRunner

        # Neutralize honcho.json reads so the only varying input is the
        # provider value itself.
        monkeypatch.setattr(
            GatewayRunner,
            "_extract_honcho_cache_busting_config",
            classmethod(lambda cls: cls._empty_honcho_cache_busting_config()),
        )

        sig_honcho = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
        sig_mem0 = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "mem0"}})

        assert sig_honcho["memory.provider"] == "honcho"
        assert sig_mem0["memory.provider"] == "mem0"
        assert sig_honcho != sig_mem0

    def test_honcho_cache_busting_config_memoized_by_mtime(self, monkeypatch, tmp_path):
        """Repeated Honcho extraction for unchanged honcho.json should reuse parse result."""
        from types import SimpleNamespace
        from gateway.run import GatewayRunner

        config_path = tmp_path / "honcho.json"
        config_path.write_text("{}")
        parse_calls = []

        class FakeConfig:
            peer_name = "eri"
            ai_peer = "hermes"
            pin_peer_name = False
            runtime_peer_prefix = "tg_"
            user_peer_aliases = {"123": "eri"}

            @classmethod
            def from_global_config(cls, config_path=None):
                parse_calls.append(config_path)
                return cls()

        fake_client = SimpleNamespace(
            HonchoClientConfig=FakeConfig,
            resolve_config_path=lambda: config_path,
        )
        monkeypatch.setitem(__import__("sys").modules, "plugins.memory.honcho.client", fake_client)
        monkeypatch.setattr(GatewayRunner, "_HONCHO_CACHE_BUSTING_MEMO", {})

        first = GatewayRunner._extract_honcho_cache_busting_config()
        second = GatewayRunner._extract_honcho_cache_busting_config()

        assert first == second
        assert first["honcho.user_peer_aliases"] == [("123", "eri")]
        assert parse_calls == [config_path]

        config_path.write_text("{\n  \"changed\": true\n}")
        third = GatewayRunner._extract_honcho_cache_busting_config()

        assert third == first
        assert parse_calls == [config_path, config_path]

    def test_full_round_trip_busts_cache_on_real_edit(self):
        """End-to-end: simulate a config edit on main and verify the
        extracted cache_keys change produces a new signature."""
        from gateway.run import GatewayRunner

        runtime = {"api_key": "k", "base_url": "u", "provider": "p"}
        cfg_before = {
            "model": {"context_length": 200_000},
            "compression": {"threshold": 0.50, "enabled": True},
        }
        cfg_after = {
            "model": {"context_length": 200_000},
            "compression": {"threshold": 0.75, "enabled": True},  # user raised threshold
        }

        sig_before = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys=GatewayRunner._extract_cache_busting_config(cfg_before),
        )
        sig_after = GatewayRunner._agent_config_signature(
            "m", runtime, [], "",
            cache_keys=GatewayRunner._extract_cache_busting_config(cfg_after),
        )
        assert sig_before != sig_after, (
            "Editing compression.threshold in config.yaml must bust the "
            "gateway's cached agent so the new threshold takes effect."
        )


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

    def test_cap_commits_memory_before_evicting_finalizable(self, monkeypatch):
        """LRU-cap eviction of a finalizable, not-yet-expired agent commits
        on_session_end extraction before releasing.

        The agent would otherwise vanish from _agent_cache before the expiry
        watcher runs, so the watcher would never fire on_session_end() and
        memory providers would miss the transcript (#11205, LRU-cap variant).
        We hold the live agent at eviction time, so commit its memory then.
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
        runner = self._bounded_runner()

        commit_calls: list = []
        release_calls: list = []
        runner._release_evicted_agent_soft = lambda agent: release_calls.append(agent)

        # Finalizable (finite policy), not yet expired.
        runner.session_store = MagicMock()
        runner.session_store._entries = {"old": MagicMock(), "new": MagicMock()}
        runner.session_store.is_session_finalizable.return_value = True
        runner.session_store._is_session_expired.return_value = False

        old_agent = self._fake_agent()
        old_agent._memory_manager = MagicMock()  # has an external provider
        old_agent._session_messages = [{"role": "user", "content": "hi"}]
        old_agent.commit_memory_session = lambda msgs=None: commit_calls.append(msgs)
        new_agent = self._fake_agent()

        with runner._agent_cache_lock:
            runner._agent_cache["old"] = (old_agent, "sig_old")
            runner._agent_cache["new"] = (new_agent, "sig_new")
            runner._enforce_agent_cache_cap()

        import time as _t
        deadline = _t.time() + 2.0
        while _t.time() < deadline and not release_calls:
            _t.sleep(0.02)
        # Memory committed with the live transcript, THEN client released.
        assert commit_calls == [[{"role": "user", "content": "hi"}]]
        assert old_agent in release_calls

    def test_cap_skips_memory_commit_for_non_finalizable(self, monkeypatch):
        """LRU-cap eviction of a mode='none' agent does NOT commit memory.

        The expiry watcher never finalizes a mode='none' session, so there is
        no missed on_session_end boundary to compensate for. Committing here
        would fire premature/repeat extraction for a session that simply keeps
        living. The agent is released without a commit.
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_MAX_SIZE", 1)
        runner = self._bounded_runner()

        commit_calls: list = []
        release_calls: list = []
        runner._release_evicted_agent_soft = lambda agent: release_calls.append(agent)

        runner.session_store = MagicMock()
        runner.session_store._entries = {"old": MagicMock(), "new": MagicMock()}
        runner.session_store.is_session_finalizable.return_value = False  # mode='none'
        runner.session_store._is_session_expired.return_value = False

        old_agent = self._fake_agent()
        old_agent._memory_manager = MagicMock()
        old_agent._session_messages = [{"role": "user", "content": "hi"}]
        old_agent.commit_memory_session = lambda msgs=None: commit_calls.append(msgs)
        new_agent = self._fake_agent()

        with runner._agent_cache_lock:
            runner._agent_cache["old"] = (old_agent, "sig_old")
            runner._agent_cache["new"] = (new_agent, "sig_new")
            runner._enforce_agent_cache_cap()

        import time as _t
        deadline = _t.time() + 2.0
        while _t.time() < deadline and not release_calls:
            _t.sleep(0.02)
        assert commit_calls == []       # no premature extraction
        assert old_agent in release_calls  # still released

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

    def test_idle_sweep_keeps_agent_when_session_not_expired(self, monkeypatch):
        """Agents past idle TTL are kept if the session hasn't expired yet.

        In daily-reset mode the reset can fire hours after the last
        user message — evicting the agent early means the
        session-expiry watcher has nothing to call on_session_end()
        with, and memory providers miss the live transcript.
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        import time as _t
        stale = self._fake_agent(last_activity=_t.time() - 10.0)

        # Session store says the session is still alive AND is finalizable
        # (finite reset policy) — so deferring eviction is correct: the expiry
        # watcher will find this agent later and fire on_session_end().
        session_entry = MagicMock()
        runner.session_store = MagicMock()
        runner.session_store._entries = {"stale-session": session_entry}
        runner.session_store.is_session_finalizable.return_value = True
        runner.session_store._is_session_expired.return_value = False

        runner._agent_cache["stale-session"] = (stale, "sig")

        evicted = runner._sweep_idle_cached_agents()
        assert evicted == 0
        assert "stale-session" in runner._agent_cache

    def test_idle_sweep_evicts_when_session_is_expired(self, monkeypatch):
        """Agent IS evicted when past idle TTL AND session store says expired."""
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        import time as _t
        stale = self._fake_agent(last_activity=_t.time() - 10.0)

        # Session store says the session has expired.
        session_entry = MagicMock()
        runner.session_store = MagicMock()
        runner.session_store._entries = {"stale-session": session_entry}
        runner.session_store.is_session_finalizable.return_value = True
        runner.session_store._is_session_expired.return_value = True

        runner._agent_cache["stale-session"] = (stale, "sig")

        evicted = runner._sweep_idle_cached_agents()
        assert evicted == 1
        assert "stale-session" not in runner._agent_cache

    def test_idle_sweep_evicts_non_finalizable_session(self, monkeypatch):
        """A mode='none' session's idle agent IS still evicted.

        is_session_finalizable() is False for reset-policy 'none': the expiry
        watcher never finalizes such a session, so deferring eviction would
        pin the cached agent for the gateway's whole lifetime — the exact
        leak the idle sweep exists to relieve. The sweep must reap it even
        though _is_session_expired() is (and stays) False.
        """
        from gateway import run as gw_run

        monkeypatch.setattr(gw_run, "_AGENT_CACHE_IDLE_TTL_SECS", 0.01)
        runner = self._bounded_runner()
        runner._cleanup_agent_resources = MagicMock()

        import time as _t
        stale = self._fake_agent(last_activity=_t.time() - 10.0)

        session_entry = MagicMock()
        runner.session_store = MagicMock()
        runner.session_store._entries = {"never-session": session_entry}
        # mode='none' → never finalizable, never expired.
        runner.session_store.is_session_finalizable.return_value = False
        runner.session_store._is_session_expired.return_value = False

        runner._agent_cache["never-session"] = (stale, "sig")

        evicted = runner._sweep_idle_cached_agents()
        assert evicted == 1
        assert "never-session" not in runner._agent_cache

    def test_is_session_finalizable_real_predicate(self, tmp_path):
        """is_session_finalizable() reflects the real reset policy.

        Uses a real SessionStore + GatewayConfig (no mocks) so the predicate
        is exercised against actual get_reset_policy() output: True for finite
        policies (idle/daily/both), False only for mode='none'.
        """
        from datetime import datetime
        from unittest.mock import patch as _patch

        from gateway.config import GatewayConfig, Platform, SessionResetPolicy
        from gateway.session import (
            SessionEntry, SessionSource, SessionStore, build_session_key,
        )

        def _entry_for(platform: Platform) -> SessionEntry:
            src = SessionSource(
                platform=platform, user_id="u1", chat_id="c1",
                user_name="t", chat_type="dm",
            )
            return SessionEntry(
                session_key=build_session_key(src),
                session_id="s1",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                origin=src,
                platform=src.platform,
                chat_type=src.chat_type,
            )

        config = GatewayConfig()
        # Give Telegram a 'none' policy via the per-platform override; leave the
        # default policy finite ('both') for the Discord case.
        config.default_reset_policy = SessionResetPolicy(mode="both")
        config.reset_by_platform[Platform.TELEGRAM] = SessionResetPolicy(mode="none")

        with _patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None

        # mode='none' → never finalized by the watcher.
        assert store.is_session_finalizable(_entry_for(Platform.TELEGRAM)) is False
        # default 'both' → finite, will eventually expire.
        assert store.is_session_finalizable(_entry_for(Platform.DISCORD)) is True

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


_FAKE_NOW = 10_000.0  # Fixed epoch for deterministic time assertions


class TestCachedAgentInactivityReset:
    """Inactivity-clock reset must be gated on _interrupt_depth == 0.

    On interrupt-recursive turns (_interrupt_depth > 0) the clock must
    keep accumulating so the inactivity watchdog can fire when a turn is
    stuck in an interrupt loop.  Resetting unconditionally prevented the
    30-min timeout from triggering (#15654).  The depth-0 reset is still
    needed: a session idle for 29 min must not trip the watchdog before
    the new turn makes its first API call (#9051).
    """

    def _fake_agent(self, stale_seconds: float = 1800.0):
        m = MagicMock()
        m._last_activity_ts = _FAKE_NOW - stale_seconds
        m._api_call_count = 10
        m._last_activity_desc = "previous turn activity"
        return m

    def test_fresh_turn_resets_idle_clock(self):
        """interrupt_depth=0: clock resets so a post-idle turn gets a
        fresh 30-min inactivity window (guard for #9051)."""
        from gateway.run import GatewayRunner

        agent = self._fake_agent(stale_seconds=1800.0)
        old_ts = agent._last_activity_ts

        with patch("gateway.run.time") as mock_time:
            mock_time.time.return_value = _FAKE_NOW
            GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)

        assert agent._last_activity_ts == _FAKE_NOW, (
            "_last_activity_ts was not reset on a fresh turn (interrupt_depth=0)"
        )
        assert agent._last_activity_ts > old_ts, (
            "Stale idle time should be cleared so the new turn gets a fresh window"
        )

    def test_fresh_turn_resets_desc(self):
        """interrupt_depth=0: description is updated to reflect the new turn."""
        from gateway.run import GatewayRunner

        agent = self._fake_agent()

        with patch("gateway.run.time") as mock_time:
            mock_time.time.return_value = _FAKE_NOW
            GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)

        assert agent._last_activity_desc == "starting new turn (cached)"

    def test_interrupt_turn_preserves_idle_clock(self):
        """interrupt_depth=1: clock preserved so accumulated stuck-turn
        idle time is not discarded by an interrupt-recursive re-entry (#15654)."""
        from gateway.run import GatewayRunner

        agent = self._fake_agent(stale_seconds=1200.0)
        old_ts = agent._last_activity_ts

        GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=1)

        assert agent._last_activity_ts == old_ts, (
            "_last_activity_ts must not be reset on interrupt-recursive turns "
            "(interrupt_depth>0) — the watchdog needs the accumulated idle time"
        )

    def test_interrupt_turn_preserves_desc(self):
        """interrupt_depth=1: desc preserved — it is semantically paired with ts."""
        from gateway.run import GatewayRunner

        agent = self._fake_agent(stale_seconds=1200.0)

        GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=1)

        assert agent._last_activity_desc == "previous turn activity", (
            "_last_activity_desc must not change on interrupt-recursive turns; "
            "it describes the activity *at* _last_activity_ts"
        )

    def test_deep_interrupt_recursion_preserves_idle_clock(self):
        """interrupt_depth=MAX-1: clock still preserved at any non-zero depth."""
        from gateway.run import GatewayRunner

        agent = self._fake_agent(stale_seconds=600.0)
        old_ts = agent._last_activity_ts

        GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=4)

        assert agent._last_activity_ts == old_ts

    def test_fresh_turn_resets_flush_cursor(self):
        """interrupt_depth=0: _last_flushed_db_idx resets so new-turn
        messages are fully persisted to the session DB (#44327)."""
        from gateway.run import GatewayRunner

        agent = self._fake_agent()
        agent._last_flushed_db_idx = 42  # stale from previous turn

        with patch("gateway.run.time") as mock_time:
            mock_time.time.return_value = _FAKE_NOW
            GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)

        assert agent._last_flushed_db_idx == 0, (
            "_last_flushed_db_idx must be reset on a fresh turn so that "
            "_flush_messages_to_session_db starts from index 0"
        )

    def test_interrupt_turn_preserves_flush_cursor(self):
        """interrupt_depth=1: _last_flushed_db_idx preserved so an
        in-progress flush is not disrupted by interrupt re-entry."""
        from gateway.run import GatewayRunner

        agent = self._fake_agent()
        agent._last_flushed_db_idx = 42

        GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=1)

        assert agent._last_flushed_db_idx == 42, (
            "_last_flushed_db_idx must not be reset on interrupt-recursive "
            "turns — the flush cursor tracks in-progress writes"
        )

    def test_api_call_count_reset_regardless_of_depth(self):
        """_api_call_count is always reset to 0 for the new turn, at any depth."""
        from gateway.run import GatewayRunner

        agent_fresh = self._fake_agent()
        agent_interrupted = self._fake_agent()

        with patch("gateway.run.time") as mock_time:
            mock_time.time.return_value = _FAKE_NOW
            GatewayRunner._init_cached_agent_for_turn(agent_fresh, interrupt_depth=0)
        GatewayRunner._init_cached_agent_for_turn(agent_interrupted, interrupt_depth=1)

        assert agent_fresh._api_call_count == 0
        assert agent_interrupted._api_call_count == 0

    def test_watchdog_accumulation_across_recursive_turns(self):
        """Scenario: stuck turn + user interrupt → recursive turn.

        The idle time seen by the watchdog must reflect the full stuck
        duration, not restart from zero on the recursive re-entry.
        """
        from gateway.run import GatewayRunner

        STUCK_FOR = 1750.0
        agent = self._fake_agent(stale_seconds=STUCK_FOR)

        # Simulate: user sees "Still working..." and sends another message.
        # That triggers an interrupt → _run_agent recurses at depth=1.
        GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=1)

        # Watchdog sees time.time() - _last_activity_ts ≥ STUCK_FOR.
        idle_secs = _FAKE_NOW - agent._last_activity_ts
        assert idle_secs >= STUCK_FOR - 1.0, (
            f"Watchdog would see {idle_secs:.0f}s idle, expected ~{STUCK_FOR}s. "
            "Inactivity timeout could not fire for a stuck interrupted turn."
        )


class TestAgentConfigSignatureUserId:
    """Shared-thread cache must not reuse an agent across users.

    HonchoSessionManager freezes the resolved runtime user identity at
    first-message init.  When the gateway session_key omits the participant
    ID (``thread_sessions_per_user=False``), a cached AIAgent created by
    user A would otherwise be reused for user B, attributing B's writes to
    A's resolved peer.  Including ``user_id`` / ``user_id_alt`` in the
    signature forces per-user agent builds in shared threads.

    Tradeoff: cold prompt cache for each user's first turn in a shared
    thread, in exchange for correct memory attribution.
    """

    def test_signature_changes_with_user_id(self):
        from gateway.run import GatewayRunner
        runtime = {"provider": "anthropic", "api_key": "k", "base_url": "", "api_mode": "chat_completions"}
        sig_a = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "", user_id="7654321"
        )
        sig_b = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "", user_id="491827364"
        )
        assert sig_a != sig_b

    def test_signature_stable_with_same_user_id(self):
        from gateway.run import GatewayRunner
        runtime = {"provider": "anthropic", "api_key": "k", "base_url": "", "api_mode": "chat_completions"}
        sig_1 = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "", user_id="7654321"
        )
        sig_2 = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "", user_id="7654321"
        )
        assert sig_1 == sig_2

    def test_signature_changes_with_user_id_alt(self):
        from gateway.run import GatewayRunner
        runtime = {"provider": "anthropic", "api_key": "k", "base_url": "", "api_mode": "chat_completions"}
        sig_a = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "",
            user_id="7654321", user_id_alt="@igor_tg",
        )
        sig_b = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "",
            user_id="7654321", user_id_alt="@erosika_tg",
        )
        assert sig_a != sig_b

    def test_signature_omits_user_id_when_absent(self):
        """Default-None user_id must not change signatures vs unset call.

        Callers that pass no user_id kwarg must produce a signature
        byte-identical to ``user_id=None`` so in-flight caches survive
        the rollout of this fix.
        """
        from gateway.run import GatewayRunner
        runtime = {"provider": "anthropic", "api_key": "k", "base_url": "", "api_mode": "chat_completions"}
        sig_implicit = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "",
        )
        sig_explicit_none = GatewayRunner._agent_config_signature(
            "claude-sonnet-4", runtime, ["hermes-telegram"], "",
            user_id=None, user_id_alt=None,
        )
        assert sig_implicit == sig_explicit_none


class TestAgentCacheMessageCountRebaseline:
    """The cross-process coherence guard (#45966) must NOT invalidate the
    cache on this process's OWN writes.

    The guard snapshots ``message_count`` at agent-build time (before the
    turn writes its own rows) and never refreshes it on reuse.  Without a
    post-turn re-baseline, the gateway's own turn grows the count and the
    next turn sees a mismatch and rebuilds the agent — every turn, for every
    conversation — silently destroying per-conversation prompt caching.

    ``_refresh_agent_cache_message_count`` re-baselines the stored count to
    the now-current value after each turn, so the guard fires ONLY when a
    different process changed the transcript.  These tests pin both halves of
    the invariant against the REAL SessionDB + the REAL guard condition.
    """

    def _runner_with_db(self, db):
        from hermes_state import AsyncSessionDB

        runner = _make_runner()
        # The gateway holds the async facade; the production refresh awaits it.
        runner._session_db = AsyncSessionDB(db)
        return runner

    @staticmethod
    def _guard_would_reuse(runner, session_key, session_id):
        """Mirror the production cache-hit guard's reuse decision exactly.

        Reuse iff the live on-disk count equals the snapshot stored next to
        the cached agent (or either side is None / it's a legacy 2-tuple).
        """
        try:
            row = runner._session_db._db.get_session(session_id)
            live = row.get("message_count", 0) if row else None
        except Exception:
            live = None
        with runner._agent_cache_lock:
            cached = runner._agent_cache.get(session_key)
        cached_mc = cached[2] if cached and len(cached) > 2 else None
        invalidate = (
            cached_mc is not None
            and live is not None
            and live != cached_mc
        )
        return not invalidate

    @pytest.mark.asyncio
    async def test_same_process_turns_preserve_cached_agent(self, tmp_path):
        """The regression guard: consecutive same-process turns must REUSE
        the cached agent (prompt cache preserved), not rebuild every turn.

        Drives the real lifecycle: snapshot at build (before this turn's
        writes), turn appends its own rows, then the post-turn re-baseline
        runs — so the NEXT turn's guard sees no external change and reuses.
        """
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "sessions.db")
        db.create_session("s1", source="telegram")
        runner = self._runner_with_db(db)
        agent = object()

        # Turn 1: cache miss -> build. Snapshot is the count BEFORE this
        # turn's own writes (production stores _current_msg_count here).
        _row = db.get_session("s1")
        build_count = _row.get("message_count", 0) if _row else 0
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (agent, "sig", build_count)

        reuses = 0
        for _turn in range(1, 6):
            # This process's own turn flushes its user + assistant rows.
            db.append_message("s1", role="user", content="u")
            db.append_message("s1", role="assistant", content="a")
            # Post-turn re-baseline (the fix).
            await runner._refresh_agent_cache_message_count("telegram:s1", "s1")
            # Next turn's guard decision.
            if self._guard_would_reuse(runner, "telegram:s1", "s1"):
                reuses += 1

        # All 5 follow-on turns must reuse — WITHOUT the re-baseline this is 0.
        assert reuses == 5
        # The same agent instance is still cached (never rebuilt).
        with runner._agent_cache_lock:
            assert runner._agent_cache["telegram:s1"][0] is agent

    @pytest.mark.asyncio
    async def test_cross_process_write_still_invalidates(self, tmp_path):
        """After the re-baseline, a DIFFERENT process appending to the same
        session must still flip the guard to rebuild (the #45966 fix holds).
        """
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "sessions.db")
        db.create_session("s1", source="telegram")
        runner = self._runner_with_db(db)
        agent = object()

        with runner._agent_cache_lock:
            _row = db.get_session("s1")
            runner._agent_cache["telegram:s1"] = (
                agent, "sig", (_row.get("message_count", 0) if _row else 0),
            )

        # Our own turn + re-baseline -> reuse next turn.
        db.append_message("s1", role="user", content="u")
        db.append_message("s1", role="assistant", content="a")
        await runner._refresh_agent_cache_message_count("telegram:s1", "s1")
        assert self._guard_would_reuse(runner, "telegram:s1", "s1") is True

        # ANOTHER process (e.g. the desktop dashboard backend) appends a turn
        # to the SAME session in the shared DB — we have NOT re-baselined for it.
        db.append_message("s1", role="user", content="external from dashboard")

        # Guard must now reject reuse so the agent rebuilds from fresh disk.
        assert self._guard_would_reuse(runner, "telegram:s1", "s1") is False

    @pytest.mark.asyncio
    async def test_rebaseline_is_fail_safe_and_skips_legacy_and_pending(self, tmp_path):
        """Re-baseline must never crash and must leave legacy 2-tuples and
        pending-sentinel entries untouched."""
        from hermes_state import AsyncSessionDB, SessionDB
        from gateway.run import _AGENT_PENDING_SENTINEL

        db = SessionDB(db_path=tmp_path / "sessions.db")
        db.create_session("s1", source="telegram")
        db.append_message("s1", role="user", content="hi")
        runner = self._runner_with_db(db)

        # No session_db -> no-op, no crash.
        runner._session_db = None
        await runner._refresh_agent_cache_message_count("telegram:s1", "s1")
        runner._session_db = AsyncSessionDB(db)

        # Falsy session_id -> no-op.
        await runner._refresh_agent_cache_message_count("telegram:s1", "")
        await runner._refresh_agent_cache_message_count("telegram:s1", None)

        # Legacy 2-tuple is left untouched (it opts out of the guard).
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (object(), "sig")
        await runner._refresh_agent_cache_message_count("telegram:s1", "s1")
        with runner._agent_cache_lock:
            assert len(runner._agent_cache["telegram:s1"]) == 2

        # Pending sentinel entry is left untouched.
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (_AGENT_PENDING_SENTINEL, "sig", 0)
        await runner._refresh_agent_cache_message_count("telegram:s1", "s1")
        with runner._agent_cache_lock:
            assert runner._agent_cache["telegram:s1"][0] is _AGENT_PENDING_SENTINEL
            assert runner._agent_cache["telegram:s1"][2] == 0

        # A probe that raises is swallowed (no crash, snapshot unchanged).
        class _BoomDB:
            def get_session(self, _sid):
                raise RuntimeError("db locked")

        runner._session_db = AsyncSessionDB(_BoomDB())  # type: ignore[assignment]
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (object(), "sig", 5)
        await runner._refresh_agent_cache_message_count("telegram:s1", "s1")
        with runner._agent_cache_lock:
            assert runner._agent_cache["telegram:s1"][2] == 5

    @pytest.mark.asyncio
    async def test_in_band_followup_reuses_cached_agent(self, tmp_path):
        """Behavioral regression for the in-band queued (/queue) follow-up.

        #46237 re-baselines the snapshot only on the EXTERNAL-turn boundary
        (in ``_handle_message_with_agent``, after the whole ``_run_agent``
        chain unwinds).  The recursive in-band follow-up re-enters the cache
        guard MID-CHAIN — while the cache still holds the build-time snapshot
        and the first turn has already flushed its own rows — so without a
        re-baseline at the follow-up boundary the guard sees the grown count
        and rebuilds the agent on THIS process's own writes, re-introducing
        the every-turn rebuild #46237 set out to fix, on the follow-up path.

        Pins both halves at that boundary: WITHOUT the re-baseline the in-band
        follow-up would rebuild; WITH it the follow-up REUSES the warm agent.
        The guard's reuse decision (``_guard_would_reuse``) mirrors the real
        cache-hit guard, which reads ``get_session(session_id)`` with the same
        ``session_id`` the recursive ``_run_agent`` call is given.
        """
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "sessions.db")
        db.create_session("s1", source="telegram")
        runner = self._runner_with_db(db)
        agent = object()

        # First turn: cache miss -> build. Snapshot is the pre-turn count.
        _row = db.get_session("s1")
        build_count = _row.get("message_count", 0) if _row else 0
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (agent, "sig", build_count)

        # First turn flushes its own user + assistant rows.
        db.append_message("s1", role="user", content="u")
        db.append_message("s1", role="assistant", content="a")

        # Bug reproduction: re-entering the guard at the in-band follow-up
        # boundary WITHOUT the re-baseline sees the grown count and rebuilds.
        assert self._guard_would_reuse(runner, "telegram:s1", "s1") is False

        # The fix: re-baseline at the follow-up boundary.
        await runner._refresh_agent_cache_message_count("telegram:s1", "s1")

        # The in-band follow-up now REUSES the cached, warm-prefix agent.
        assert self._guard_would_reuse(runner, "telegram:s1", "s1") is True
        with runner._agent_cache_lock:
            assert runner._agent_cache["telegram:s1"][0] is agent

    def test_in_band_followup_rebaseline_precedes_recursion(self):
        """Pin the FIX PLACEMENT in the production source.

        The behavioral test above proves the re-baseline makes the in-band
        follow-up reuse the cached agent, but it calls the helper directly —
        it would still pass if the production call were deleted.  This guards
        the actual call site: the queued (/queue) follow-up recurses via
        ``followup_result = await self._run_agent(...)`` inside
        ``_run_agent_inner`` and the re-baseline MUST run BEFORE that
        recursion (running it only after, like the external-turn site, is too
        late for the in-band path — the follow-up would already have rebuilt).
        """
        import inspect
        from gateway.run import GatewayRunner

        # The recursion + pre-recursion re-baseline live in the extracted
        # ``_run_agent_inner`` (older trees had them inline in ``_run_agent``).
        src = inspect.getsource(GatewayRunner._run_agent_inner)
        marker = "followup_result = await self._run_agent("
        assert marker in src, "in-band queued follow-up recursion not found in _run_agent_inner"
        before_recursion = src[: src.index(marker)]
        assert "_refresh_agent_cache_message_count" in before_recursion, (
            "the in-band queued follow-up recursion must be preceded by a "
            "_refresh_agent_cache_message_count re-baseline, else the follow-up "
            "rebuilds the agent on this process's own first-turn writes"
        )

class TestCrossProcessInvalidationDefersCleanup:
    """#52197: cross-process cache invalidation must NOT run agent cleanup
    while holding ``_agent_cache_lock``.

    The #45966 guard popped the stale cached agent and then called the
    blocking ``_cleanup_agent_resources`` (memory-provider shutdown, socket
    teardown) *inside* the ``with _agent_cache_lock:`` block, on the gateway
    event-loop thread.  While that ran, ``_sweep_idle_cached_agents`` (driven
    by the session-expiry watcher) blocked acquiring the same lock and the
    asyncio loop stalled, tripping Discord heartbeat-blocked warnings.

    The fix mirrors the cap-enforcer / idle-sweep paths: pop under the lock,
    release it, then schedule the SOFT release (which preserves the session's
    terminal sandbox / browser / bg processes for the immediately-rebuilt
    agent) on a daemon thread.

    These tests replicate the exact eviction sequence the production guard now
    performs and pin the invariant: the lock is free while cleanup runs, and
    the hard-teardown path is never used here.
    """

    def _runner(self):
        from collections import OrderedDict
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        return runner

    @staticmethod
    def _evict_like_production(runner, session_key):
        """Run the post-#52197 cross-process eviction sequence verbatim:
        pop the stale entry under the lock, then schedule the soft release
        on a daemon thread AFTER the lock is released."""
        _xproc_evicted_agent = None
        with runner._agent_cache_lock:
            evicted = runner._agent_cache.pop(session_key, None)
            _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
            if _ev_agent is not None:
                _xproc_evicted_agent = _ev_agent
        if _xproc_evicted_agent is not None:
            threading.Thread(
                target=runner._release_evicted_agent_soft,
                args=(_xproc_evicted_agent,),
                daemon=True,
                name="agent-xproc-evict-test",
            ).start()

    def test_cleanup_runs_with_lock_released(self):
        """The cache lock must be acquirable WHILE the evicted agent's
        cleanup is running — proving cleanup is off the locked path."""
        runner = self._runner()

        cleanup_started = threading.Event()
        release_lock = threading.Event()

        def _soft(agent):
            cleanup_started.set()
            # Block here as if memory-provider shutdown / socket teardown is
            # slow.  If cleanup were still holding _agent_cache_lock, the
            # assertion below could never acquire it.
            release_lock.wait(timeout=2.0)

        runner._release_evicted_agent_soft = _soft
        runner._cleanup_agent_resources = MagicMock()

        old_agent = MagicMock()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (old_agent, "sig", 3)

        self._evict_like_production(runner, "telegram:s1")

        # Wait until the (blocking) cleanup is mid-flight.
        assert cleanup_started.wait(timeout=2.0)

        # The lock MUST be free right now — this is the heart of #52197.
        # A 0.5s acquire timeout would fire if cleanup held the lock.
        acquired = runner._agent_cache_lock.acquire(timeout=0.5)
        assert acquired, "cache lock blocked during cross-process cleanup (#52197)"
        runner._agent_cache_lock.release()

        # Let the cleanup thread finish.
        release_lock.set()

        # Stale entry was popped, hard-teardown path never used.
        assert "telegram:s1" not in runner._agent_cache
        runner._cleanup_agent_resources.assert_not_called()

    def test_soft_release_scheduled_for_evicted_agent(self):
        """The evicted agent is handed to the soft-release path, not the
        hard ``_cleanup_agent_resources`` teardown."""
        runner = self._runner()

        release_calls: list = []
        runner._release_evicted_agent_soft = lambda agent: release_calls.append(agent)
        runner._cleanup_agent_resources = MagicMock()

        old_agent = MagicMock()
        with runner._agent_cache_lock:
            runner._agent_cache["telegram:s1"] = (old_agent, "sig", 3)

        self._evict_like_production(runner, "telegram:s1")

        import time as _t
        deadline = _t.time() + 2.0
        while _t.time() < deadline and not release_calls:
            _t.sleep(0.02)

        assert release_calls == [old_agent]
        runner._cleanup_agent_resources.assert_not_called()
