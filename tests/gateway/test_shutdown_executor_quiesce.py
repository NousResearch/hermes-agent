"""Gateway shutdown quiesces its thread pool before closing state.db (#101093).

``_shutdown_executor()`` used to run *after* the SessionDB close block in
``_stop_impl``, and it never waited: ``cancel_futures`` only drops work that has
not started, and cancelling the awaiting task does not stop the worker thread
behind a ``run_in_executor`` future.  So blocking DB work could still be running
when ``SessionDB.close()`` checkpointed the WAL and let SQLite unlink the
sidecar.  The late write then reopens the handle (#94736) and mints a fresh WAL
generation behind that checkpoint, leaving teardown to checkpoint the same file
a second time from a connection the shutdown log never accounts for -- the
close-time page-write damage in #101093 and the split WAL generation in #101064.

The order is now: quiesce (bounded) -> close.
"""

import asyncio
import concurrent.futures
import threading
import time
from collections import OrderedDict

import pytest

import gateway.run as gw_mod


class _FakeSessionDB:
    """Records when the gateway closed it, on a shared event log."""

    def __init__(self, events, name):
        self._events = events
        self._name = name

    def close(self):
        self._events.append(f"close:{self._name}")


class _FakeGateway:
    """Minimal stand-in with just enough state for ``stop()`` to run."""

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

    # -- shutdown collaborators the real stop() reaches into ---------------

    def _running_agent_count(self):
        return len(self._running_agents)

    def _active_cron_job_count(self):
        return 0

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
async def test_running_executor_work_finishes_before_session_db_close():
    """A future already running when stop() begins writes before the close."""
    events = []
    gw = _FakeGateway(events)
    started = threading.Event()

    def _blocking_db_write():
        started.set()
        # Longer than the rest of the shutdown tail (~0.4s), shorter than the
        # 2s quiesce ceiling: without the wait the close lands first.
        time.sleep(1.0)
        events.append("worker_write")

    future = gw._executor.submit(_blocking_db_write)
    assert started.wait(2.0), "worker never started"

    await gw_mod.GatewayRunner.stop(gw)
    future.result(timeout=5)

    assert "worker_write" in events, "worker never ran"
    assert "close:session_db" in events, "SessionDB was never closed"
    assert events.index("worker_write") < events.index("close:session_db"), (
        f"state.db was closed while a worker was still writing: {events}"
    )


@pytest.mark.asyncio
async def test_executor_refuses_new_work_before_session_db_close():
    """``_executor_closing`` is set before the close, so no fresh pool is minted."""
    events = []
    gw = _FakeGateway(events)

    real_close = gw._session_db.close

    def _close_and_probe():
        # The flag must already be set by the time the DB is closed, or a
        # coroutine reaching _get_executor() here would spin up a new pool and
        # run more blocking DB work against the handle being torn down.
        events.append(f"closing_flag:{gw._executor_closing}")
        real_close()

    gw._session_db.close = _close_and_probe

    await gw_mod.GatewayRunner.stop(gw)

    assert "closing_flag:True" in events, events
    with pytest.raises(RuntimeError):
        gw_mod.GatewayRunner._get_executor(gw)


def test_shutdown_executor_defaults_to_no_wait():
    """The no-argument call keeps the historical fire-and-forget contract."""
    gw = _FakeGateway([])
    release = threading.Event()
    started = threading.Event()

    def _slow():
        started.set()
        release.wait(5.0)

    future = gw._executor.submit(_slow)
    assert started.wait(2.0)

    began = time.monotonic()
    still_live = gw_mod.GatewayRunner._shutdown_executor(gw)
    elapsed = time.monotonic() - began

    assert elapsed < 0.5, f"default call waited {elapsed:.2f}s"
    assert still_live == 1
    release.set()
    future.result(timeout=5)


def test_shutdown_executor_reports_a_stuck_worker():
    """A worker that outlives the budget is reported, not waited on forever."""
    gw = _FakeGateway([])
    release = threading.Event()
    started = threading.Event()

    def _stuck():
        started.set()
        release.wait(5.0)

    future = gw._executor.submit(_stuck)
    assert started.wait(2.0)

    began = time.monotonic()
    still_live = gw_mod.GatewayRunner._shutdown_executor(gw, drain_timeout=0.2)
    elapsed = time.monotonic() - began

    assert still_live == 1
    assert 0.15 <= elapsed < 2.0, f"budget not honoured: {elapsed:.2f}s"
    release.set()
    future.result(timeout=5)


def test_shutdown_executor_without_executor_returns_zero():
    gw = _FakeGateway([])
    gw._executor.shutdown(wait=True)
    gw._executor = None
    assert gw_mod.GatewayRunner._shutdown_executor(gw, drain_timeout=1.0) == 0
