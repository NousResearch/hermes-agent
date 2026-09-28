"""Tests for agent.memory_health — the memory-provider health state singleton."""

from unittest.mock import MagicMock

import pytest

from agent.memory_health import MemoryHealthState, get_health_state, reset_health_state
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _FakeProvider(MemoryProvider):
    """Minimal concrete provider for MemoryManager hook tests."""

    def __init__(self, name: str = "hindsight"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []


class _InitFailingProvider(_FakeProvider):
    """Provider whose ``initialize()`` raises — exercises the init-failure hook."""

    def __init__(self, name: str = "hindsight"):
        super().__init__(name)
        self.initialize_called = False

    def initialize(self, session_id: str, **kwargs) -> None:
        self.initialize_called = True
        raise RuntimeError("backend handshake failed")


class _ToolCallProvider(_FakeProvider):
    """Provider exposing one memory tool whose calls raise (or succeed)."""

    def __init__(self, name: str = "hindsight", *, fail: bool = True):
        super().__init__(name)
        self._fail = fail
        self.calls = 0

    def get_tool_schemas(self):
        return [{"name": f"{self.name}_tool", "description": "t",
                 "parameters": {"type": "object", "properties": {}}}]

    def handle_tool_call(self, tool_name: str, args, **kwargs) -> str:
        self.calls += 1
        if self._fail:
            raise RuntimeError("backend down")
        return "ok"


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset the singleton before each test."""
    reset_health_state()
    yield
    reset_health_state()


# ---------------------------------------------------------------------------
# indicator_text()
# ---------------------------------------------------------------------------


class TestIndicatorText:
    def test_no_provider_returns_empty(self):
        hs = MemoryHealthState()
        assert hs.indicator_text() == ""

    def test_active_provider_healthy(self):
        hs = MemoryHealthState(active_provider="hindsight", health="healthy")
        assert hs.indicator_text() == "hindsight ● Healthy"

    def test_active_provider_unavailable(self):
        hs = MemoryHealthState(active_provider="hindsight", health="unavailable")
        assert hs.indicator_text() == "hindsight ✖ Unavailable"

    def test_active_provider_unknown(self):
        hs = MemoryHealthState(active_provider="hindsight", health="unknown")
        assert hs.indicator_text() == "hindsight ○ Connecting"

    def test_custom_provider_name(self):
        hs = MemoryHealthState(active_provider="my_custom_mem", health="healthy")
        assert hs.indicator_text() == "my_custom_mem ● Healthy"
        assert "hindsight" not in hs.indicator_text()

    def test_fallback_to_configured_when_active_empty(self):
        hs = MemoryHealthState(configured_provider="hindsight", health="unknown")
        assert "hindsight" in hs.indicator_text()


# ---------------------------------------------------------------------------
# mark_healthy / mark_unavailable
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_starts_unknown(self):
        hs = MemoryHealthState()
        assert hs.health == "unknown"

    def test_mark_healthy(self):
        hs = MemoryHealthState()
        hs.mark_healthy()
        assert hs.health == "healthy"
        assert hs.reason == ""

    def test_mark_unavailable(self):
        hs = MemoryHealthState()
        hs.mark_unavailable("timeout")
        assert hs.health == "unavailable"
        assert hs.reason == "timeout"

    def test_recovery(self):
        hs = MemoryHealthState()
        hs.mark_unavailable("down")
        assert hs.health == "unavailable"
        hs.mark_healthy()
        assert hs.health == "healthy"
        assert hs.reason == ""

    def test_healthy_to_unavailable(self):
        hs = MemoryHealthState()
        hs.mark_healthy()
        hs.mark_unavailable("lost connection")
        assert hs.health == "unavailable"

    def test_same_provider_preserves_health(self):
        hs = MemoryHealthState(active_provider="hindsight", health="healthy")
        hs.set_active_provider("hindsight")

        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"

    def test_changed_provider_resets_health(self):
        hs = MemoryHealthState(
            active_provider="hindsight",
            health="unavailable",
            reason="connection refused",
        )
        hs._last_probe_failure_at = 123.0

        hs.set_active_provider("my_provider")

        assert hs.active_provider == "my_provider"
        assert hs.health == "unknown"
        assert hs.reason == ""
        assert hs._last_probe_failure_at == 0.0

# ---------------------------------------------------------------------------
# probe_cooldown_active()
# ---------------------------------------------------------------------------


class TestProbeCooldown:
    def test_no_cooldown_when_healthy(self):
        hs = MemoryHealthState(health="healthy")
        assert hs.probe_cooldown_active() is False

    def test_cooldown_when_probe_failed(self):
        hs = MemoryHealthState(health="unavailable")
        hs.record_probe_failure()
        assert hs.probe_cooldown_active() is True

    def test_cooldown_expires(self):
        import time
        hs = MemoryHealthState(health="unavailable")
        hs.record_probe_failure()
        hs._last_probe_failure_at = time.monotonic() - 31.0
        assert hs.probe_cooldown_active() is False

    def test_operation_failure_does_not_start_probe_cooldown(self):
        """§5/§11: only a PROBE failure arms the cooldown.  An ordinary memory
        operation failure (mark_unavailable alone) must not suppress the next
        recovery probe — the historical _last_probe_at bug class."""
        hs = MemoryHealthState()
        hs.mark_unavailable("prefetch timed out")
        assert hs.health == "unavailable"
        assert hs.probe_cooldown_active() is False

    def test_never_probed_unavailable_has_no_cooldown(self, monkeypatch):
        """§5 regression: `_last_probe_failure_at == 0.0` means "no probe
        failure ever" and must never read as cooldown — even when the monotonic
        clock is small (fresh boot, uptime < PROBE_COOLDOWN_S), where
        `monotonic() - 0.0 < 30` would otherwise misarm the cooldown and
        suppress the recovery probe of an operation-failure unavailable."""
        import time
        hs = MemoryHealthState(health="unavailable")
        monkeypatch.setattr(time, "monotonic", lambda: 5.0)  # uptime 5 s
        assert hs.probe_cooldown_active() is False


# ---------------------------------------------------------------------------
# reset_health_state()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_fields(self):
        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()
        reset_health_state()
        hs2 = get_health_state()
        assert hs2.configured_provider == ""
        assert hs2.active_provider == ""
        assert hs2.health == "unknown"


# ---------------------------------------------------------------------------
# MemoryManager failure hooks — ownership gate + no-auto-recover contract
# ---------------------------------------------------------------------------


class TestMemoryManagerHealthWrites:
    """The failure hooks write the foreground's singleton — and only that."""

    def test_failure_marks_unavailable_for_owning_manager(self):
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_healthy()
        mm = MemoryManager()
        assert mm.health_write_allowed is True  # default: foreground

        def _boom(p):
            raise RuntimeError("connection refused")

        mm._each_provider("recall failed", _boom, providers=[_FakeProvider("hindsight")])
        assert hs.health == "unavailable"
        assert "hindsight" in hs.reason

    def test_failure_ignored_for_background_manager(self):
        """Frozen §8: a background agent's manager must never write the
        foreground's process-wide health state, even for the same provider."""
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_healthy()
        mm = MemoryManager()
        mm.health_write_allowed = False  # background agent

        def _boom(p):
            raise RuntimeError("connection refused")

        mm._each_provider("recall failed", _boom, providers=[_FakeProvider("hindsight")])
        assert hs.health == "healthy"

    def test_success_does_not_auto_recover(self):
        """§6: only a runtime probe may restore healthy — a provider that
        swallows failures internally makes a failed call look successful."""
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_unavailable("down")
        mm = MemoryManager()
        mm._each_provider("recall", lambda p: "", providers=[_FakeProvider("hindsight")])
        assert hs.health == "unavailable"

    def test_foreground_prefetch_timeout_marks_unavailable_without_cooldown(self):
        """§12.2: prefetch timeout is an operation/runtime failure — it marks
        unavailable but must NOT arm the probe cooldown."""
        import threading
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        class _BlockingProvider(_FakeProvider):
            def __init__(self):
                super().__init__("hindsight")
                self.release = threading.Event()

            def prefetch(self, query, *, session_id=""):
                self.release.wait(10)
                return ""

        prov = _BlockingProvider()
        mm = MemoryManager(external_prefetch_timeout=0.05)
        try:
            assert mm._prefetch_provider(prov, "query") == ""
        finally:
            prov.release.set()
        assert hs.health == "unavailable"
        assert "prefetch timed out" in hs.reason
        assert hs.probe_cooldown_active() is False

    def test_background_prefetch_timeout_leaves_singleton_unchanged(self):
        """Frozen §13 matrix: background agents' managers never write the
        foreground's singleton, on the prefetch path either."""
        import threading
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        class _BlockingProvider(_FakeProvider):
            def __init__(self):
                super().__init__("hindsight")
                self.release = threading.Event()

            def prefetch(self, query, *, session_id=""):
                self.release.wait(10)
                return ""

        prov = _BlockingProvider()
        mm = MemoryManager(external_prefetch_timeout=0.05)
        mm.health_write_allowed = False  # background agent
        try:
            assert mm._prefetch_provider(prov, "query") == ""
        finally:
            prov.release.set()
        assert hs.health == "healthy"

    def test_handle_tool_call_failure_marks_unavailable(self):
        """Explicit memory tool calls are memory operations too (review: a
        provider must not stay 'healthy' while every tool call fails)."""
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_healthy()
        mm = MemoryManager()
        mm.add_provider(_ToolCallProvider("hindsight"))
        out = mm.handle_tool_call("hindsight_tool", {})
        assert "failed" in out
        assert hs.health == "unavailable"
        assert hs.probe_cooldown_active() is False  # op failure ≠ probe failure

    def test_handle_tool_call_failure_ignored_for_background_manager(self):
        """Frozen §8: the tool-call path respects the ownership gate too."""
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_healthy()
        mm = MemoryManager()
        mm.health_write_allowed = False  # background agent
        mm.add_provider(_ToolCallProvider("hindsight"))
        mm.handle_tool_call("hindsight_tool", {})
        assert hs.health == "healthy"

    def test_handle_tool_call_success_does_not_auto_recover(self):
        """§4: only probe_health() may restore healthy."""
        hs = get_health_state()
        hs.active_provider = "hindsight"
        hs.mark_unavailable("down")
        mm = MemoryManager()
        mm.add_provider(_ToolCallProvider("hindsight", fail=False))
        assert mm.handle_tool_call("hindsight_tool", {}) == "ok"
        assert hs.health == "unavailable"


# ---------------------------------------------------------------------------
# foreground ownership — positive identification, fail closed
# ---------------------------------------------------------------------------


class TestForegroundHealthOwnership:
    """Review: ownership is positive identification — unknown surfaces
    (api_server, acp, gateway, batch, diagnostic agents) fail closed."""

    def test_only_cli_foreground_owns(self):
        from agent.memory_health import is_foreground_health_owner
        agent = MagicMock()
        agent.side_agent = False
        agent._parent_session_id = None
        assert is_foreground_health_owner(agent, "cli") is True
        assert is_foreground_health_owner(agent, "api_server") is False
        assert is_foreground_health_owner(agent, "acp") is False
        assert is_foreground_health_owner(agent, "") is False

    def test_structural_background_markers_veto(self):
        from agent.memory_health import is_foreground_health_owner
        agent = MagicMock()
        agent.side_agent = True
        agent._parent_session_id = None
        assert is_foreground_health_owner(agent, "cli") is False
        agent.side_agent = False
        agent._parent_session_id = "parent-1"
        assert is_foreground_health_owner(agent, "cli") is False


# ---------------------------------------------------------------------------
# _init_memory: foreground/background ownership of the singleton
# ---------------------------------------------------------------------------


class TestInitMemoryOwnership:

    def test_background_cron_agent_does_not_overwrite_state(self):
        """A background cron agent without an external provider must not overwrite
        the main agent's process-wide memory health state."""
        from unittest.mock import patch as _patch

        hs = get_health_state()

        # Simulate the main agent already having a healthy external provider.
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()
        assert hs.health == "healthy"

        # Simulate a background cron agent with no external provider.
        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False  # Ensure cron, not side_agent, triggers the guard.
        mock_agent._parent_session_id = None

        mock_cfg = {"memory": {}}

        with (
            _patch(
                "tools.memory_tool.get_builtin_memory_config",
                return_value={},
            ),
            _patch(
                "plugins.memory.load_memory_provider",
                return_value=None,
            ),
            _patch(
                "hermes_cli.memory_provider_migration.recover_at_startup",
                return_value=False,
            ),
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                mock_cfg,
                skip_memory=False,
                platform="cron",
            )

        # The background agent must not overwrite the main agent's state.
        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"
    def test_background_agent_with_external_provider_does_not_overwrite_state(self):
        """A background agent with its own external provider must not overwrite
        the foreground agent's process-wide memory health state."""
        from types import SimpleNamespace
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        memory_manager = SimpleNamespace(
            providers=[SimpleNamespace(name="cron_provider")],
        )

        mock_cfg = {"memory": {}}

        with _patch(
            "tools.memory_tool.get_builtin_memory_config",
            return_value={},
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                mock_cfg,
                skip_memory=False,
                platform="cron",
                memory_manager=memory_manager,
            )

        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"
        # P2: the background agent's manager may not write the singleton either.
        assert memory_manager.health_write_allowed is False

    def test_api_server_agent_does_not_overwrite_state(self):
        """Review: gateway ``api_server`` spawns an independent agent per
        session in ONE process — ownership must fail closed for non-CLI
        surfaces, or those agents overwrite the foreground's (and each
        other's) singleton."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "api-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        with _patch("tools.memory_tool.get_builtin_memory_config", return_value={}):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {}},
                skip_memory=False,
                platform="api_server",
            )

        assert hs.configured_provider == "hindsight"
        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"

    def test_background_init_failure_cannot_mark_before_gate(self):
        """Regression (review): the ownership gate must exist BEFORE
        ``initialize_all()`` — init failures route through the same failure
        hooks, and a background agent re-initializing the same provider name
        would otherwise flip the foreground's singleton."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "bg-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        prov = _InitFailingProvider("hindsight")
        with (
            _patch("tools.memory_tool.get_builtin_memory_config",
                   return_value={"provider": "hindsight"}),
            _patch("plugins.memory.load_memory_provider", return_value=prov),
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {}},
                skip_memory=False,
                platform="curator",
            )

        assert prov.initialize_called is True  # the failure path really ran
        assert hs.health == "healthy"
        assert hs.active_provider == "hindsight"

    def test_foreground_init_failure_marks_unavailable(self):
        """§7 for the OWNER: an ``initialize()`` failure surfaces as
        ``unavailable`` on the indicator (recovery only via probe_health())."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "fg-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        prov = _InitFailingProvider("hindsight")
        with (
            _patch("tools.memory_tool.get_builtin_memory_config",
                   return_value={"provider": "hindsight"}),
            _patch("plugins.memory.load_memory_provider", return_value=prov),
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {}},
                skip_memory=False,
                platform="cli",
            )

        assert prov.initialize_called is True
        assert hs.health == "unavailable"

    def test_explicitly_skipped_memory_is_not_reported_unavailable(self):
        """``skip_memory=True`` is an intentional off, not a provider failure
        (review: avoid reporting explicitly skipped memory as unavailable) —
        the indicator must show nothing instead of a false "Unavailable"."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"

        mock_agent = MagicMock()
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        with _patch("tools.memory_tool.get_builtin_memory_config",
                    return_value={"provider": "hindsight"}):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {}},
                skip_memory=True,
                platform="cli",
            )

        assert hs.configured_provider == ""
        assert hs.active_provider == ""
        assert hs.health == "unknown"

    def test_foreground_agent_without_external_provider_clears_stale_state(self):
        """A foreground agent without an external provider must clear stale
        process-wide memory health state from a previous agent."""
        from unittest.mock import patch as _patch

        hs = get_health_state()

        # Simulate stale state left by a previous foreground agent.
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()
        assert hs.health == "healthy"

        # Simulate a foreground agent with no external provider configured.
        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        mock_cfg = {"memory": {}}

        with (
            _patch(
                "tools.memory_tool.get_builtin_memory_config",
                return_value={},
            ),
            _patch(
                "plugins.memory.load_memory_provider",
                return_value=None,
            ),
            _patch(
                "hermes_cli.memory_provider_migration.recover_at_startup",
                return_value=False,
            ),
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                mock_cfg,
                skip_memory=False,
                platform="cli",
            )

        # Foreground agent with no external provider must clear stale state.
        assert hs.active_provider == ""
        assert hs.configured_provider == ""
        assert hs.health == "unknown"

    def test_foreground_agent_manager_allows_health_writes(self):
        """The foreground agent's manager owns the singleton (init keeps health
        "unknown" — only the first probe may mark healthy, §9)."""
        from types import SimpleNamespace
        from unittest.mock import patch as _patch

        hs = get_health_state()
        mock_agent = MagicMock()
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        memory_manager = SimpleNamespace(
            providers=[SimpleNamespace(name="builtin"), SimpleNamespace(name="hindsight")],
        )
        with _patch(
            "tools.memory_tool.get_builtin_memory_config",
            return_value={"provider": "hindsight"},
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {"provider": "hindsight"}},
                skip_memory=False,
                platform="cli",
                memory_manager=memory_manager,
            )

        assert memory_manager.health_write_allowed is True
        assert hs.active_provider == "hindsight"
        assert hs.health == "unknown"  # init never marks healthy

    def test_foreground_agent_configured_but_not_loaded_marks_unavailable(self):
        """Frozen §7: configured=hindsight but nothing actually loaded →
        Unavailable ("hindsight not loaded"), never Connecting or Healthy."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        with (
            _patch(
                "tools.memory_tool.get_builtin_memory_config",
                return_value={"provider": "hindsight"},
            ),
            _patch("plugins.memory.load_memory_provider", return_value=None),
            _patch("hermes_cli.memory_provider_migration.recover_at_startup", return_value=False),
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {"provider": "hindsight"}},
                skip_memory=False,
                platform="cli",
            )

        assert hs.configured_provider == "hindsight"
        assert hs.active_provider == ""
        assert hs.health == "unavailable"
        assert "not loaded" in hs.reason

    def test_background_curator_agent_does_not_overwrite_state(self):
        """Regression: AIAgent(platform="curator") re-enters _init_memory
        in-process (agent/curator.py run_curator_review via maybe_run_curator).
        Frozen §8: it must not wipe the foreground singleton — and because the
        after-turn probe gates on active_provider, a wipe would be permanent."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None  # curator is background by platform

        with _patch(
            "tools.memory_tool.get_builtin_memory_config",
            return_value={"provider": "hindsight"},
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {"provider": "hindsight"}},
                skip_memory=True,
                platform="curator",
            )

        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"

    def test_background_review_fork_does_not_overwrite_state(self):
        """Regression: agent/background_review.py forks inherit the PARENT's
        platform ("cli"), so only parent_session_id marks them background."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "fork-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = "parent-session"  # the fork's signal

        with _patch(
            "tools.memory_tool.get_builtin_memory_config",
            return_value={"provider": "hindsight"},
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {"provider": "hindsight"}},
                skip_memory=True,
                platform="cli",  # inherited from the parent agent
            )

        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"

    def test_background_side_agent_does_not_overwrite_state(self):
        """Frozen §13 matrix row: background side_agent cannot write the
        singleton even with platform="cli"."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "side-session"
        mock_agent.side_agent = True  # the side agent's signal
        mock_agent._parent_session_id = None

        with _patch(
            "tools.memory_tool.get_builtin_memory_config",
            return_value={"provider": "hindsight"},
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {"provider": "hindsight"}},
                skip_memory=True,
                platform="cli",
            )

        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"

    def test_background_subagent_does_not_overwrite_state(self):
        """Frozen §13 matrix row: delegate_task children (platform="subagent")
        cannot write the singleton."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        hs.configured_provider = "hindsight"
        hs.active_provider = "hindsight"
        hs.mark_healthy()

        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "sub-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None  # proves the platform-set signal

        with _patch(
            "tools.memory_tool.get_builtin_memory_config",
            return_value={"provider": "hindsight"},
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {"provider": "hindsight"}},
                skip_memory=True,
                platform="subagent",
            )

        assert hs.active_provider == "hindsight"
        assert hs.health == "healthy"

    def test_foreground_agent_builtin_sentinel_not_shown(self):
        """Frozen §16: core sentinels (builtin/default/none) never become
        configured_provider and never render as an external provider."""
        from unittest.mock import patch as _patch

        hs = get_health_state()
        mock_agent = MagicMock()
        mock_agent._memory_manager = None
        mock_agent._memory_enabled = False
        mock_agent._user_profile_enabled = False
        mock_agent.enabled_toolsets = []
        mock_agent.disabled_toolsets = []
        mock_agent._session_db = None
        mock_agent.session_cwd = None
        mock_agent.session_id = "test-session"
        mock_agent.side_agent = False
        mock_agent._parent_session_id = None

        with _patch(
            "tools.memory_tool.get_builtin_memory_config",
            return_value={"provider": "builtin"},
        ):
            from agent.agent_init import _init_memory

            _init_memory(
                mock_agent,
                {"memory": {"provider": "builtin"}},
                skip_memory=False,
                platform="cli",
            )

        assert hs.configured_provider == ""
        assert hs.indicator_text() == ""
        assert hs.health == "unknown"