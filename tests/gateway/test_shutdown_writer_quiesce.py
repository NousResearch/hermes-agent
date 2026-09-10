"""Tests for cron/deferred worker quiesce before closing state.db."""

import asyncio
import concurrent.futures
import threading
import time
from collections import OrderedDict
from unittest import mock

import pytest

import gateway.run as gw_mod
import gateway.run_shutdown as gw_shutdown_mod


class _FakeSessionDB:
    def __init__(self, events, name):
        self._events = events
        self._name = name

    def close(self):
        self._events.append(f"close:{self._name}")


class _FakeGateway:
    def __init__(self, events):
        self._events = events
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
        self._executor_lock = threading.Lock()
        self._executor_closing = False
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="quiesce-test"
        )
        self._session_db = _FakeSessionDB(events, "session_db")
        self.session_store = None
        
        # State for our tests
        self.cron_count = 0
        self.deferred_count = 0

    def _running_agent_count(self):
        return len(self._running_agents)

    def _active_cron_job_count(self):
        return self.cron_count

    def _active_deferred_agent_worker_count(self):
        return self.deferred_count

    def _active_api_run_count(self):
        return 0

    def _update_runtime_status(self, *_a, **_kw):
        pass

    def _clear_plugin_message_injector(self):
        pass

    async def _run_in_executor_with_context(self, func, *args):
        return func(*args)

    async def _cleanup_agent_resources_off_loop(self, agent, *, context=""):
        self._cleanup_agent_resources(agent)

    async def _notify_active_sessions_of_shutdown(self):
        pass

    async def _cancel_secondary_profile_reconnect_tasks(self):
        pass

    async def _drain_active_agents(self, timeout, cron_timeout=None):
        return {}, False

    async def _finalize_shutdown_agents(self, agents):
        pass

    def _cleanup_agent_resources(self, agent):
        pass

    def _evict_cached_agent(self, key):
        pass

    def _release_running_agent_state(self, session_key, **_kwargs):
        self._running_agents.pop(session_key, None)
        self._running_agents_ts.pop(session_key, None)
        return False

    def close_all_session_db_handles(self):
        pass


@pytest.mark.asyncio
async def test_stuck_cron_writer_skips_the_session_db_close():
    events = []
    gw = _FakeGateway(events)
    gw.cron_count = 1

    with mock.patch("gateway.run_shutdown.resolve_cron_drain_budget", return_value=0.0):
        await gw_mod.GatewayRunner.stop(gw)

    assert "close:session_db" not in events, "SessionDB closed despite stuck cron worker"


@pytest.mark.asyncio
async def test_stuck_deferred_worker_skips_the_session_db_close():
    events = []
    gw = _FakeGateway(events)
    gw.deferred_count = 1

    with mock.patch("gateway.run_shutdown.resolve_cron_drain_budget", return_value=0.0):
        await gw_mod.GatewayRunner.stop(gw)

    assert "close:session_db" not in events, "SessionDB closed despite stuck deferred worker"


@pytest.mark.asyncio
async def test_writers_finish_before_timeout_allows_session_db_close():
    events = []
    gw = _FakeGateway(events)
    gw.cron_count = 1
    gw.deferred_count = 1
    
    def finish_work():
        time.sleep(0.05)
        gw.cron_count = 0
        gw.deferred_count = 0
        
    threading.Thread(target=finish_work, daemon=True).start()

    with mock.patch("gateway.run_shutdown.resolve_cron_drain_budget", return_value=1.0):
        await gw_mod.GatewayRunner.stop(gw)

    assert "close:session_db" in events, "SessionDB was not closed even though writers finished"
