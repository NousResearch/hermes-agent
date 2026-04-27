"""Regression tests for gateway shutdown cleaning up cached agent memory providers (issue #11205).

When the gateway shuts down, ``stop()`` called ``_finalize_shutdown_agents()``
which only drained agents in ``_running_agents``.  Idle agents sitting in
``_agent_cache`` (LRU cache) were never cleaned up, so their
``MemoryProvider.on_session_end()`` hooks never fired.

The fix adds an explicit sweep of ``_agent_cache`` after
``_finalize_shutdown_agents`` in the ``_stop_impl`` coroutine.
"""

import asyncio
import threading
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

# Import the module (not the class) to reach stop() and helpers
import gateway.run as gw_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeGateway:
    """Minimal stand-in with just enough state for ``stop()`` to run."""

    def __init__(self):
        self._running = True
        self._draining = False
        self._restart_requested = False
        self._restart_detached = False
        self._restart_via_service = False
        self._stop_task = None
        self._exit_cleanly = False
        self._exit_with_failure = False
        self._exit_reason = None
        self._exit_code = None
        self._restart_drain_timeout = 0.01
        self._running_agents = {}
        self._running_agents_ts = {}
        self._agent_cache = OrderedDict()
        self._agent_cache_lock = threading.Lock()
        self.adapters = {}
        self._background_tasks = set()
        self._failed_platforms = []
        self._shutdown_event = asyncio.Event()
        self._pending_messages = {}
        self._pending_approvals = {}
        self._busy_ack_ts = {}

    def _running_agent_count(self):
        return len(self._running_agents)

    def _update_runtime_status(self, *_a, **_kw):
        pass

    async def _notify_active_sessions_of_shutdown(self):
        pass

    async def _drain_active_agents(self, timeout):
        return {}, False

    def _finalize_shutdown_agents(self, agents):
        for agent in agents.values():
            self._cleanup_agent_resources(agent)

    def _cleanup_agent_resources(self, agent):
        if agent is None:
            return
        try:
            if hasattr(agent, "shutdown_memory_provider"):
                agent.shutdown_memory_provider()
        except Exception:
            pass
        try:
            if hasattr(agent, "close"):
                agent.close()
        except Exception:
            pass

    def _evict_cached_agent(self, key):
        pass


def _make_mock_agent():
    a = MagicMock()
    a.shutdown_memory_provider = MagicMock()
    a.close = MagicMock()
    return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCachedAgentCleanupOnShutdown:
    """Verify that ``stop()`` calls ``_cleanup_agent_resources`` on idle
    cached agents, triggering ``shutdown_memory_provider()`` (which calls
    ``on_session_end``)."""

    @pytest.mark.asyncio
    async def test_cached_agent_memory_provider_shut_down(self):
        """A cached agent's shutdown_memory_provider is called during gateway stop."""
        gw = _FakeGateway()
        agent = _make_mock_agent()
        gw._agent_cache["session-1"] = (agent, "sig-123")

        # Call the real stop() from GatewayRunner
        await gw_mod.GatewayRunner.stop(gw)

        agent.shutdown_memory_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_cleared_after_shutdown(self):
        """The _agent_cache dict is cleared after stop."""
        gw = _FakeGateway()
        agent = _make_mock_agent()
        gw._agent_cache["s1"] = (agent, "sig1")

        await gw_mod.GatewayRunner.stop(gw)

        assert len(gw._agent_cache) == 0

    @pytest.mark.asyncio
    async def test_no_cached_agents_no_error(self):
        """stop() works fine when _agent_cache is empty."""
        gw = _FakeGateway()

        await gw_mod.GatewayRunner.stop(gw)  # Should not raise

        assert len(gw._agent_cache) == 0

    @pytest.mark.asyncio
    async def test_multiple_cached_agents_all_cleaned(self):
        """All cached agents get cleaned up."""
        gw = _FakeGateway()
        agents = []
        for i in range(5):
            a = _make_mock_agent()
            agents.append(a)
            gw._agent_cache[f"s{i}"] = (a, f"sig{i}")

        await gw_mod.GatewayRunner.stop(gw)

        for a in agents:
            a.shutdown_memory_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_survives_agent_exception(self):
        """An exception from one agent's shutdown doesn't prevent others."""
        gw = _FakeGateway()

        bad = _make_mock_agent()
        bad.shutdown_memory_provider.side_effect = RuntimeError("boom")
        bad.close.side_effect = RuntimeError("boom")

        good = _make_mock_agent()

        gw._agent_cache["bad"] = (bad, "sig-bad")
        gw._agent_cache["good"] = (good, "sig-good")

        await gw_mod.GatewayRunner.stop(gw)

        # The good agent should still be cleaned up
        good.shutdown_memory_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_plain_agent_not_tuple(self):
        """Cache entries that aren't tuples (just bare agents) are also cleaned."""
        gw = _FakeGateway()
        agent = _make_mock_agent()
        gw._agent_cache["s1"] = agent  # Not a tuple

        await gw_mod.GatewayRunner.stop(gw)

        agent.shutdown_memory_provider.assert_called_once()
        assert len(gw._agent_cache) == 0

    @pytest.mark.asyncio
    async def test_none_entry_skipped(self):
        """A None cache entry doesn't cause errors."""
        gw = _FakeGateway()
        gw._agent_cache["s1"] = None

        await gw_mod.GatewayRunner.stop(gw)

        assert len(gw._agent_cache) == 0


class TestRunningAgentsNotDoubleCleaned:
    """Verify behavior when agents appear in both _running_agents and _agent_cache."""

    @pytest.mark.asyncio
    async def test_running_and_cached_agent_cleaned_at_least_once(self):
        """An agent in both _running_agents and _agent_cache gets
        shutdown_memory_provider called at least once."""
        gw = _FakeGateway()
        shared = _make_mock_agent()

        gw._running_agents["s1"] = shared
        gw._agent_cache["s1"] = (shared, "sig1")

        await gw_mod.GatewayRunner.stop(gw)

        # Called at least once — either from _finalize_shutdown_agents
        # or from the cache sweep (or both)
        assert shared.shutdown_memory_provider.call_count >= 1
