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
from types import SimpleNamespace
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


    # ---------------------------------------------------------------
    # cache_keys (compression/context config cache-busting)
    # ---------------------------------------------------------------


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


class TestExtractCacheBustingConfig:
    """Verify _extract_cache_busting_config pulls the documented subset of
    config values that must invalidate the cached agent on change."""


    def test_reads_compression_subkeys(self):
        from gateway.run import GatewayRunner

        out = GatewayRunner._extract_cache_busting_config(
            {
                "compression": {
                    "enabled": False,
                    "threshold": 0.6,
                    "codex_gpt55_autoraise": False,
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
        from agent.session_activity import ActivityProvenance

        m = MagicMock()
        m._last_activity_ts = _FAKE_NOW - stale_seconds
        m._api_call_count = 10
        m._last_activity_desc = "previous turn activity"
        m._last_activity_provenance = ActivityProvenance.AGENT_COMPRESSION
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


    def test_fresh_turn_resets_provenance(self):
        """interrupt_depth=0: provenance resets with ts/desc (#72039)."""
        from agent.session_activity import ActivityProvenance
        from gateway.run import GatewayRunner

        agent = self._fake_agent()
        assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION

        with patch("gateway.run.time") as mock_time:
            mock_time.time.return_value = _FAKE_NOW
            GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=0)

        assert agent._last_activity_provenance is ActivityProvenance.UNKNOWN

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

    def test_interrupt_turn_preserves_provenance(self):
        """interrupt_depth=1: provenance preserved with ts/desc."""
        from agent.session_activity import ActivityProvenance
        from gateway.run import GatewayRunner

        agent = self._fake_agent(stale_seconds=1200.0)

        GatewayRunner._init_cached_agent_for_turn(agent, interrupt_depth=1)

        assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION

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

    @pytest.mark.asyncio
    async def test_compression_rotation_advances_cached_agent_to_child(self, tmp_path):
        """A cached agent that performed parent -> child compression stays warm."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "sessions.db")
        db.create_session("parent", source="telegram")
        db.create_session("child", source="telegram", parent_session_id="parent")
        runner = self._runner_with_db(db)
        agent = SimpleNamespace(session_id="child")
        session_key = "telegram:rotated"

        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = (agent, "sig", 0, "parent")

        db.append_message("child", role="user", content="u")
        db.append_message("child", role="assistant", content="a")
        await runner._refresh_agent_cache_message_count(
            session_key,
            "child",
            previous_session_id="parent",
            expected_agent=agent,
        )

        child_row = db.get_session("child")
        with runner._agent_cache_lock:
            cached = runner._agent_cache[session_key]
        assert cached[0] is agent
        assert cached[2] == child_row["message_count"]
        assert cached[3] == "child"
        assert self._guard_would_reuse(runner, session_key, "child") is True

    @pytest.mark.asyncio
    async def test_rotation_rebaseline_rejects_stale_parent_agent(self, tmp_path):
        """A dead-parent/self-heal artifact must still reach the eviction guard."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "sessions.db")
        db.create_session("parent", source="telegram")
        db.create_session("child", source="telegram", parent_session_id="parent")
        runner = self._runner_with_db(db)
        stale_agent = SimpleNamespace(session_id="parent")
        session_key = "telegram:stale-parent"
        original = (stale_agent, "sig", 0, "parent")

        with runner._agent_cache_lock:
            runner._agent_cache[session_key] = original

        await runner._refresh_agent_cache_message_count(
            session_key,
            "child",
            previous_session_id="parent",
            expected_agent=stale_agent,
        )

        with runner._agent_cache_lock:
            assert runner._agent_cache[session_key] == original


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

