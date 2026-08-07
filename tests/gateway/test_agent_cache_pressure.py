"""Memory-pressure eviction for the gateway agent cache (#80764).

The LRU cap counts entries and the idle sweep counts seconds, so a gateway
serving many warm sessions holds every full transcript resident until the
cgroup kills it. These tests pin the pressure valve that sheds them, and the
three things it must never shed: a mid-turn agent, the most-recently-used
sessions, and a session whose transcript has not finished reaching disk.
"""

import threading
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from gateway.agent_cache_pressure import (
    AgentCacheBounds,
    plan_pressure_evictions,
    resolve_agent_cache_bounds,
    resolve_memory_high_mb,
    transcript_persistence_caught_up,
)


class TestBoundsResolution:
    """Absent config must stay absent so gateway/run.py keeps its defaults."""

    def test_absent_section_leaves_lru_bounds_unset(self):
        bounds = resolve_agent_cache_bounds({})
        assert bounds.max_size is None
        assert bounds.idle_ttl_secs is None

    def test_configured_values_are_honoured(self):
        bounds = resolve_agent_cache_bounds(
            {
                "agent": {
                    "agent_cache": {
                        "max_size": 32,
                        "idle_ttl_secs": 600,
                        "memory_high_mb": 2048,
                        "max_evictions_per_pass": 4,
                        "protect_recent": 2,
                    }
                }
            }
        )
        assert bounds.max_size == 32
        assert bounds.idle_ttl_secs == 600.0
        assert bounds.memory_high_mb == 2048
        assert bounds.max_evictions_per_pass == 4
        assert bounds.protect_recent == 2

    def test_garbage_values_fall_back_to_defaults(self):
        """A typo in config.yaml must not disable the cache or crash startup."""
        bounds = resolve_agent_cache_bounds(
            {"agent": {"agent_cache": {"max_size": "lots", "idle_ttl_secs": -5}}}
        )
        assert bounds.max_size is None
        assert bounds.idle_ttl_secs is None
        assert bounds.max_evictions_per_pass > 0

    def test_protect_recent_zero_is_respected(self):
        """0 means "shed anything", which is distinct from "unset"."""
        bounds = resolve_agent_cache_bounds(
            {"agent": {"agent_cache": {"protect_recent": 0}}}
        )
        assert bounds.protect_recent == 0


class TestMemoryBudgetResolution:
    @pytest.mark.parametrize("setting", [0, False, None, "off", "none", ""])
    def test_falsy_settings_disable_the_pass(self, setting):
        assert resolve_memory_high_mb(setting) is None

    @pytest.mark.parametrize("setting", [4096, "4096", 4096.0])
    def test_explicit_budget_is_taken_literally(self, setting):
        assert resolve_memory_high_mb(setting) == 4096

    def test_auto_derives_a_budget_below_the_cgroup_limit(self, monkeypatch):
        """The budget must leave headroom: hitting memory.high is what makes
        the shutdown flush time out in the first place."""
        import gateway.agent_cache_pressure as acp

        limit_mb = 10 * 1024
        monkeypatch.setattr(acp, "_cgroup_limit_bytes", lambda: limit_mb * 1024 * 1024)

        budget = resolve_memory_high_mb("auto")

        assert budget is not None
        assert 0 < budget < limit_mb

    def test_auto_is_disabled_when_no_limit_is_discoverable(self, monkeypatch):
        import gateway.agent_cache_pressure as acp

        monkeypatch.setattr(acp, "_cgroup_limit_bytes", lambda: None)
        monkeypatch.setattr(acp, "_total_memory_bytes", lambda: None)

        assert resolve_memory_high_mb("auto") is None


class TestPersistenceGuard:
    """Soft eviction drops the transcript, so it may only run once the
    transcript is durable. Exercised against the real AIAgent flush."""

    def _agent(self, tmp_path, session_id):
        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "sessions.db")
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            max_iterations=5,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id=session_id,
            session_db=db,
        )
        db.create_session(session_id, source="telegram")
        agent._session_db_created = True
        return agent

    def test_fresh_agent_holds_nothing_to_lose(self, tmp_path):
        agent = self._agent(tmp_path, "fresh")
        try:
            assert transcript_persistence_caught_up(agent) is True
        finally:
            agent.close()

    def test_unflushed_turn_blocks_eviction_then_flush_unblocks_it(self, tmp_path):
        agent = self._agent(tmp_path, "lagging")
        try:
            messages = [
                {"role": "user", "content": "read the logs"},
                {"role": "assistant", "content": "done"},
            ]
            agent._session_messages = messages

            assert transcript_persistence_caught_up(agent) is False, (
                "a transcript that never reached disk must not be dropped — "
                "the session would come back with amnesia"
            )

            assert agent._flush_messages_to_session_db(messages) is True
            assert transcript_persistence_caught_up(agent) is True
        finally:
            agent.close()

    def test_unknown_shapes_are_treated_as_unsafe(self):
        assert transcript_persistence_caught_up(object()) is False
        assert transcript_persistence_caught_up(None) is False


class TestEvictionPlanner:
    def _entries(self, n):
        return [(f"s{i}", MagicMock()) for i in range(n)]

    def test_evicts_least_recently_used_first(self):
        entries = self._entries(6)
        plan = plan_pressure_evictions(
            entries, is_evictable=lambda k, a: True, max_evictions=2, protect_recent=0
        )
        assert [key for key, _ in plan] == ["s0", "s1"]

    def test_never_touches_the_protected_tail(self):
        entries = self._entries(10)
        plan = plan_pressure_evictions(
            entries, is_evictable=lambda k, a: True, max_evictions=10, protect_recent=3
        )
        assert [key for key, _ in plan] == ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]

    @pytest.mark.parametrize("size", [1, 2, 3, 5])
    def test_a_small_cache_of_large_transcripts_is_still_shedable(self, size):
        """A fixed MRU guard would protect the whole cache when a couple of
        sessions are big enough to blow the budget on their own — the gateway
        would then climb toward the OOM killer with nothing it would shed."""
        plan = plan_pressure_evictions(
            self._entries(size),
            is_evictable=lambda k, a: True,
            max_evictions=10,
            protect_recent=8,
        )
        assert plan, f"nothing evictable with {size} cached session(s)"
        assert len(plan) <= size

    def test_protection_still_keeps_the_hottest_session(self):
        plan = plan_pressure_evictions(
            self._entries(4),
            is_evictable=lambda k, a: True,
            max_evictions=10,
            protect_recent=8,
        )
        assert "s3" not in [key for key, _ in plan]

    def test_skipped_candidates_do_not_consume_the_batch(self):
        """Skipping a protected session must not shrink the batch — otherwise
        one wedged session throttles the whole pass."""
        entries = self._entries(6)
        plan = plan_pressure_evictions(
            entries,
            is_evictable=lambda k, a: k != "s0",
            max_evictions=2,
            protect_recent=0,
        )
        assert [key for key, _ in plan] == ["s1", "s2"]


class TestGatewayPressureSweep:
    """End-to-end against the real GatewayRunner method."""

    def _runner(self, bounds=None):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        runner._agent_cache_bounds_cache = bounds or AgentCacheBounds(
            memory_high_mb=1000, max_evictions_per_pass=8, protect_recent=1
        )
        return runner

    def _cached_agent(self, *, persisted=True, messages=2):
        agent = MagicMock()
        agent._session_messages = [{"role": "user", "content": "x"}] * messages
        agent._last_flushed_db_idx = messages if persisted else 0
        return agent

    def _at_rss(self, monkeypatch, mb):
        import gateway.agent_cache_pressure as acp

        monkeypatch.setattr(acp, "read_anon_rss_mb", lambda: mb)

    def test_no_eviction_below_budget(self, monkeypatch):
        runner = self._runner()
        self._at_rss(monkeypatch, 400)
        for i in range(5):
            runner._agent_cache[f"s{i}"] = (self._cached_agent(), "sig")

        assert runner._sweep_agent_cache_under_pressure() == 0
        assert len(runner._agent_cache) == 5

    def test_over_budget_sheds_lru_and_frees_the_transcript(self, monkeypatch):
        runner = self._runner()
        self._at_rss(monkeypatch, 4000)
        released: list = []
        runner._commit_then_release_soft = lambda agent, key: (
            released.append(key),
            setattr(agent, "_session_messages", []),
        )

        for i in range(4):
            runner._agent_cache[f"s{i}"] = (self._cached_agent(), "sig")
        oldest = runner._agent_cache["s0"][0]

        evicted = runner._sweep_agent_cache_under_pressure()

        assert evicted == 3  # protect_recent=1 keeps the newest
        assert "s0" not in runner._agent_cache
        assert "s3" in runner._agent_cache
        _wait_for(lambda: released == ["s0", "s1", "s2"])
        assert oldest._session_messages == []

    def test_mid_turn_session_is_never_evicted(self, monkeypatch):
        runner = self._runner()
        self._at_rss(monkeypatch, 4000)
        runner._commit_then_release_soft = lambda agent, key: None

        active = self._cached_agent()
        runner._agent_cache["s-active"] = (active, "sig")
        runner._agent_cache["s-idle"] = (self._cached_agent(), "sig")
        runner._agent_cache["s-new"] = (self._cached_agent(), "sig")
        runner._running_agents["s-active"] = active

        runner._sweep_agent_cache_under_pressure()

        assert "s-active" in runner._agent_cache, (
            "evicting a mid-turn agent tears down the clients and sandbox the "
            "running request is using"
        )
        assert "s-idle" not in runner._agent_cache

    def test_lagging_persistence_blocks_eviction(self, monkeypatch):
        runner = self._runner()
        self._at_rss(monkeypatch, 4000)
        runner._commit_then_release_soft = lambda agent, key: None

        runner._agent_cache["s-lagging"] = (
            self._cached_agent(persisted=False), "sig",
        )
        runner._agent_cache["s-durable"] = (self._cached_agent(), "sig")
        runner._agent_cache["s-new"] = (self._cached_agent(), "sig")

        runner._sweep_agent_cache_under_pressure()

        assert "s-lagging" in runner._agent_cache, (
            "dropping a transcript that never reached disk loses the "
            "conversation the FTS guard exists to protect"
        )
        assert "s-durable" not in runner._agent_cache

    def test_empty_cache_is_a_no_op(self, monkeypatch):
        """Heap pressure with nothing cached is somebody else's problem."""
        runner = self._runner()
        self._at_rss(monkeypatch, 999_999)

        assert runner._sweep_agent_cache_under_pressure() == 0

    def test_all_candidates_skipped_reports_zero_without_raising(self, monkeypatch):
        runner = self._runner()
        self._at_rss(monkeypatch, 4000)
        runner._commit_then_release_soft = lambda agent, key: None
        for i in range(3):
            runner._agent_cache[f"s{i}"] = (
                self._cached_agent(persisted=False), "sig",
            )

        assert runner._sweep_agent_cache_under_pressure() == 0
        assert len(runner._agent_cache) == 3

    def test_disabled_budget_is_a_no_op(self, monkeypatch):
        runner = self._runner(bounds=AgentCacheBounds(memory_high_mb=None))
        self._at_rss(monkeypatch, 999_999)
        runner._agent_cache["s0"] = (self._cached_agent(), "sig")

        assert runner._sweep_agent_cache_under_pressure() == 0
        assert "s0" in runner._agent_cache


class TestConfiguredBoundsReachTheCache:
    """The two existing bounds must be operator-tunable, and must keep their
    built-in values when config.yaml says nothing."""

    def _runner(self, bounds):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache_bounds_cache = bounds
        return runner

    def test_unset_config_keeps_the_built_in_defaults(self):
        from gateway import run as gw_run

        runner = self._runner(AgentCacheBounds())
        assert runner._agent_cache_cap() == gw_run._AGENT_CACHE_MAX_SIZE
        assert runner._agent_cache_idle_ttl() == gw_run._AGENT_CACHE_IDLE_TTL_SECS

    def test_configured_cap_bounds_the_real_enforcer(self):
        """A configured cap must actually shrink the cache, not just report."""
        runner = self._runner(AgentCacheBounds(max_size=2))
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        runner._release_evicted_agent_soft = lambda agent: None
        runner._commit_then_release_soft = lambda agent, key: None

        with runner._agent_cache_lock:
            for i in range(5):
                runner._agent_cache[f"s{i}"] = (MagicMock(), "sig")
            runner._enforce_agent_cache_cap()

        assert len(runner._agent_cache) == 2
        assert list(runner._agent_cache) == ["s3", "s4"]

    def test_configured_idle_ttl_drives_the_real_sweep(self):
        import time as _t

        runner = self._runner(AgentCacheBounds(idle_ttl_secs=0.01))
        runner._agent_cache = OrderedDict()
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        runner._release_evicted_agent_soft = lambda agent: None
        runner.session_store = None

        stale = MagicMock()
        stale._last_activity_ts = _t.time() - 5.0
        runner._agent_cache["s-stale"] = (stale, "sig")

        assert runner._sweep_idle_cached_agents() == 1
        assert "s-stale" not in runner._agent_cache


def _wait_for(predicate, timeout: float = 3.0) -> None:
    """Wait for a background release thread to finish its work."""
    import time as _t

    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if predicate():
            return
        _t.sleep(0.02)
    assert predicate(), "background release did not complete in time"


class TestSalvageFollowups:
    """Follow-up behaviors added while salvaging PR #80795."""

    def test_config_read_failure_still_resolves_auto_budget(self, monkeypatch):
        """A transient config-read failure must not permanently disable the
        pressure valve — the fallback resolves an empty config, whose absent
        section means memory_high_mb='auto', not None."""
        import gateway.run as gw_run
        from gateway.run import GatewayRunner

        monkeypatch.setattr(
            gw_run, "_load_gateway_config",
            lambda: (_ for _ in ()).throw(OSError("transient")),
        )
        import gateway.agent_cache_pressure as acp

        monkeypatch.setattr(acp, "_cgroup_limit_bytes", lambda: 8 * 1024**3)

        runner = GatewayRunner.__new__(GatewayRunner)
        bounds = runner._agent_cache_bounds()
        assert bounds.memory_high_mb is not None, (
            "config-read failure fell back to a disabled valve — "
            "the #80764 protection must survive a flaky config read"
        )

    def test_protect_recent_yaml_false_keeps_default(self):
        """protect_recent: false (YAML-typo bool; False == 0) must keep the
        default MRU protection, not silently disable it."""
        bounds = resolve_agent_cache_bounds(
            {"agent": {"agent_cache": {"protect_recent": False}}}
        )
        assert bounds.protect_recent > 0

    def test_release_batch_drains_plan_before_trim(self, monkeypatch):
        """The plan list must be empty when trim_memory runs, so no local
        reference pins the evicted agents during gc.collect + malloc_trim
        (otherwise the in-pass trim frees nothing and the next tick
        over-evicts another batch)."""
        from collections import OrderedDict as _OD

        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        released = []
        runner._commit_then_release_soft = lambda agent, key: released.append(key)

        plan_len_at_trim = {}

        import hermes_cli.mem_trim as mem_trim_mod

        plan = [(f"s{i}", MagicMock()) for i in range(3)]

        def fake_trim(force=False, reason=None):
            plan_len_at_trim["len"] = len(plan)
            return True

        monkeypatch.setattr(mem_trim_mod, "trim_memory", fake_trim)

        runner._release_pressure_batch(plan)

        assert sorted(released) == ["s0", "s1", "s2"]
        assert plan_len_at_trim["len"] == 0, (
            "plan still held agent references when trim_memory ran"
        )

    def test_soft_release_clears_db_flush_scan_prefix(self):
        """_db_flush_scan_prefix shallow-copies the flushed transcript and is
        populated on exactly the agents the valve targets — leaving it pins
        every message dict the eviction claims to free."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        agent = MagicMock()
        transcript = [{"role": "user", "content": "x" * 1024}]
        agent._session_messages = transcript
        agent._db_flush_scan_prefix = transcript[:]

        runner._release_evicted_agent_soft(agent)

        assert agent._session_messages == []
        assert agent._db_flush_scan_prefix is None

    def test_no_evictable_warning_distinguishes_unflushed_persistence(self, monkeypatch, caplog):
        """When everything is blocked on un-flushed persistence (e.g. the
        session DB never initialized), the warning must say so instead of
        blaming mid-turn agents."""
        import logging as _logging

        from collections import OrderedDict as _OD

        import gateway.agent_cache_pressure as acp
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._agent_cache = _OD()
        runner._agent_cache_lock = threading.Lock()
        runner._running_agents = {}
        runner._agent_cache_bounds_cache = AgentCacheBounds(
            memory_high_mb=1000, max_evictions_per_pass=8, protect_recent=0
        )
        monkeypatch.setattr(acp, "read_anon_rss_mb", lambda: 4000)

        for i in range(3):
            agent = MagicMock()
            agent._session_messages = [{"role": "user", "content": "x"}]
            agent._last_flushed_db_idx = 0  # never flushed
            runner._agent_cache[f"s{i}"] = (agent, "sig")

        with caplog.at_level(_logging.WARNING, logger="gateway.run"):
            evicted = runner._sweep_agent_cache_under_pressure()

        assert evicted == 0
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "blocked on un-flushed persistence" in joined
        assert "3 blocked" in joined
