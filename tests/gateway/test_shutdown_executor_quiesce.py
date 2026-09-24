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
from contextlib import suppress

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

    # Real hook + counter: 0 while ``adapters`` is empty, the API-server count once a fake adapter is in.
    _api_server_hook = gw_mod.GatewayShutdownMixin._api_server_hook
    _mark_api_runs_shutdown_requested = gw_mod.GatewayShutdownMixin._mark_api_runs_shutdown_requested
    _active_api_run_count = gw_mod.GatewayShutdownMixin._active_api_run_count
    _active_api_worker_count = gw_mod.GatewayShutdownMixin._active_api_worker_count
    _active_deferred_agent_worker_count = gw_mod.GatewayShutdownMixin._active_deferred_agent_worker_count

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


@pytest.mark.asyncio
async def test_stuck_worker_skips_the_session_db_close():
    """A worker that outlives the quiesce budget must not be raced by close().

    Reporting the live worker with a "may reopen state.db" warning is not
    enough: the close()/checkpoint itself is the operation that raced the
    late write and produced the wrong-page-number corruption in #101093,
    so the close path has to be skipped whenever a worker survives the
    budget, not merely logged around.
    """
    events = []
    gw = _FakeGateway(events)
    release = threading.Event()
    started = threading.Event()

    def _stuck():
        started.set()
        release.wait(5.0)
        events.append("worker_write")

    future = gw._executor.submit(_stuck)
    assert started.wait(2.0), "worker never started"

    # Force the quiesce budget to 0 so the worker is deterministically still
    # alive when `_shutdown_executor` returns, without sleeping through the
    # real 2s ceiling.
    original_timeout = gw_mod._EXECUTOR_QUIESCE_TIMEOUT
    gw_mod._EXECUTOR_QUIESCE_TIMEOUT = 0.0
    try:
        await gw_mod.GatewayRunner.stop(gw)
    finally:
        gw_mod._EXECUTOR_QUIESCE_TIMEOUT = original_timeout

    assert "close:session_db" not in events, (
        f"SessionDB was closed/checkpointed while a worker was still alive: {events}"
    )

    release.set()
    future.result(timeout=5)
    assert "worker_write" in events, "worker never finished"


def _arm_cron(gw):
    gw._active_cron_job_count = lambda: 1


def _arm_api(gw):
    # Through the real hook: the adapter map is cleared one phase before the close gate, so the
    # gate must use the count taken before the clear, not a live lookup.
    from gateway.config import Platform

    class _ApiAdapter:
        def active_agent_work_count(self):
            return 1

    async def _teardown(adapter, platform, *, profile=None):
        pass  # the run keeps going on the default executor after the transport is torn down

    gw.adapters[Platform.API_SERVER] = _ApiAdapter()
    gw._bounded_adapter_teardown = _teardown


def _arm_deferred(gw):
    # A hygiene worker on the loop's default executor, never finished.
    gw._deferred_agent_workers = {asyncio.get_event_loop().create_future(): object()}


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", [_arm_cron, _arm_api, _arm_deferred], ids=["cron", "api", "deferred"])
async def test_live_writer_outside_the_executor_skips_the_session_db_close(monkeypatch, arm):
    """A cron job, API-server run or deferred worker that outlived the drain must not have state.db
    closed under it (#102198).

    None of them run on ``self._executor`` (scheduler pool / loop default executor), so the executor
    join above the close block never sees them; the close has to consult their counters too.
    The executor must still be sealed on this path (#101118).
    """
    import hermes_state_registry

    events = []
    gw = _FakeGateway(events)
    arm(gw)
    monkeypatch.setattr(
        hermes_state_registry, "close_all", lambda: events.append("close_all") or 0
    )

    await gw_mod.GatewayRunner.stop(gw)

    assert "close:session_db" not in events and "close_all" not in events, (
        f"SessionDB closed despite a live {arm.__name__[5:]} writer: {events}"
    )
    assert gw._executor_closing is True, "executor left unsealed on the outside-writer path"


@pytest.mark.asyncio
async def test_cancelled_api_handler_worker_still_blocks_session_db_close(monkeypatch):
    """A cancelled request handler must not let state.db be closed under its live worker (#116535).

    The handler-side ``_inflight_agent_runs`` count drops in the handler's ``finally`` when the
    handler task is cancelled, while the executor thread behind ``run_in_executor`` is still
    blocked in the turn. Only the worker-scoped count still sees that thread, so the SessionDB
    close gate must consult it alongside the handler snapshot -- and must not close until the
    worker itself exits.
    """
    import hermes_state_registry
    from gateway.platforms import api_server_runs as api_runs

    events = []
    gw = _FakeGateway(events)
    monkeypatch.setattr(
        hermes_state_registry, "close_all", lambda: events.append("close_all") or 0
    )

    release = threading.Event()
    worker_started = threading.Event()

    def _blocked_turn():
        worker_started.set()
        assert release.wait(5.0), "test tore down while the worker was still blocked"
        events.append("worker_done")

    loop = asyncio.get_running_loop()

    async def _handler():
        # Same shape as the api_server call sites: the worker-scoped count is taken before
        # run_in_executor and released in the worker's own finally, so cancelling this task
        # drops the handler side while the thread keeps holding the worker side.
        return await api_runs._submit_api_worker(loop, _blocked_turn)

    task = asyncio.ensure_future(_handler())
    assert await loop.run_in_executor(None, worker_started.wait, 5.0), "worker never started"
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert api_runs.api_worker_live_count() == 1, (
        "cancelled handler took the worker count with it"
    )

    await gw_mod.GatewayRunner.stop(gw)

    assert "close:session_db" not in events and "close_all" not in events, (
        f"SessionDB closed while the API worker was still alive: {events}"
    )

    release.set()
    for _ in range(100):
        if "worker_done" in events:
            break
        await asyncio.sleep(0.05)
    assert "worker_done" in events, "worker never finished"
    assert api_runs.api_worker_live_count() == 0, "worker exit did not release the count"


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


