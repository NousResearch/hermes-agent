"""Regression coverage for #63529 API-server shutdown draining.

API-server work is adapter-owned rather than tracked by
``GatewayRunner._running_agents``. The shutdown drain must account for the
same live state as the API concurrency limiter, including a ``/v1/runs`` task
that exists before its agent has been constructed, and it must refuse new API
turns once the gateway starts draining.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import _INTERRUPT_REASON_GATEWAY_SHUTDOWN
from hermes_state import SessionDB
from tests.gateway.restart_test_helpers import attach_real_stop, make_restart_runner

# Safety net so a regression parks the executor thread forever instead of
# hanging CI.  No assertion below depends on elapsed time.
_TURN_UNBLOCK_TIMEOUT = 30.0


class _RunTask:
    def __init__(self, done: bool = False):
        self._done = done

    def done(self) -> bool:
        return self._done


def _make_api_adapter(*, inflight: int = 0, queued_ids=()):
    tasks = {run_id: _RunTask() for run_id in queued_ids}
    adapter = SimpleNamespace(
        platform=Platform.API_SERVER,
        _inflight_agent_runs=inflight,
        _active_run_tasks=tasks,
    )

    def active_agent_work_count() -> int:
        return int(getattr(adapter, "_pending_agent_requests", 0)) + int(
            adapter._inflight_agent_runs
        ) + sum(not task.done() for task in adapter._active_run_tasks.values())

    adapter.active_agent_work_count = active_agent_work_count
    return adapter


def _make_admission_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream
    )
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


class TestActiveApiRunCount:
    def test_zero_when_no_api_adapter(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {}
        assert runner._active_api_run_count() == 0


class TestAPIServerAdapterWorkCount:

    @pytest.mark.asyncio
    async def test_concurrency_limit_excludes_current_pending_admission(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._max_concurrent_runs = 1
        app = _make_admission_app(adapter)

        async with TestClient(TestServer(app)) as client:
            with patch.object(adapter, "_run_agent", new=AsyncMock(return_value=({}, {}))):
                response = await client.post(
                    "/api/sessions/s/chat",
                    json={"message": "hello"},
                )

        assert response.status == 404


    def test_counts_live_run_task_before_agent_creation(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._inflight_agent_runs = 2
        adapter._active_run_tasks = {
            "queued": _RunTask(),
            "finished": _RunTask(done=True),
        }
        adapter._active_run_agents = {}

        assert adapter.active_agent_work_count() == 3

    def test_does_not_double_count_started_run_agent(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._inflight_agent_runs = 0
        adapter._active_run_tasks = {"run-1": _RunTask()}
        adapter._active_run_agents = {"run-1": object()}

        assert adapter.active_agent_work_count() == 1

    def test_interrupt_active_runs_interrupts_adapter_owned_agents(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        agent = MagicMock()
        adapter._active_run_agents = {"run-1": agent}

        assert adapter.interrupt_active_runs("gateway shutdown") == 1

        agent.interrupt.assert_called_once_with("gateway shutdown")


class TestDrainWaitsForApiWork:

    @pytest.mark.asyncio
    async def test_drain_waits_for_real_queued_run_before_agent_creation(self):
        """A live /v1/runs task must block drain before it has an agent."""
        runner, _adapter = make_restart_runner()
        api = APIServerAdapter(PlatformConfig(enabled=True))
        runner.adapters = {Platform.API_SERVER: api}
        app = _make_admission_app(api)
        original_create_task = asyncio.create_task
        task_started = asyncio.Event()
        allow_task = asyncio.Event()

        def delayed_create_task(coro):
            async def delayed():
                task_started.set()
                await allow_task.wait()
                return await coro

            return original_create_task(delayed())

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "done"}
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        with patch(
            "gateway.platforms.api_server.asyncio.create_task",
            side_effect=delayed_create_task,
        ), patch.object(api, "_create_agent", return_value=mock_agent):
            async with TestClient(TestServer(app)) as client:
                response = await client.post("/v1/runs", json={"input": "hello"})
                assert response.status == 202
                await task_started.wait()

                assert api._active_run_agents == {}
                assert runner._active_api_run_count() == 1
                drain_task = original_create_task(runner._drain_active_agents(2.0))
                await asyncio.sleep(0.1)
                assert not drain_task.done()

                allow_task.set()
                _snapshot, timed_out = await drain_task

        assert timed_out is False

    @pytest.mark.asyncio
    async def test_drain_times_out_if_api_run_outlives_the_window(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {Platform.API_SERVER: _make_api_adapter(queued_ids=["run-1"])}

        _snapshot, timed_out = await runner._drain_active_agents(0.1)

        assert timed_out is True

    def test_shutdown_interrupt_reaches_api_server_runs(self):
        runner, _adapter = make_restart_runner()
        api = APIServerAdapter(PlatformConfig(enabled=True))
        agent = MagicMock()
        api._active_run_agents = {"run-1": agent}
        runner.adapters = {Platform.API_SERVER: api}

        runner._interrupt_running_agents("gateway shutdown")

        agent.interrupt.assert_called_once_with("gateway shutdown")

    @pytest.mark.asyncio
    async def test_drain_still_waits_for_chat_cron_and_api_work(self):
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        runner._running_agents = {"session-1": MagicMock()}
        sched._running_job_ids.add("job-1")
        runner.adapters = {Platform.API_SERVER: _make_api_adapter(queued_ids=["run-1"])}

        async def finish_all():
            await asyncio.sleep(0.12)
            runner._running_agents.clear()
            sched._running_job_ids.discard("job-1")
            runner.adapters[Platform.API_SERVER]._active_run_tasks.clear()

        task = asyncio.create_task(finish_all())
        try:
            _snapshot, timed_out = await runner._drain_active_agents(2.0)
        finally:
            await task
            sched._running_job_ids.discard("job-1")

        assert timed_out is False


class TestDrainAdmission:
    @pytest.mark.asyncio
    async def test_drain_refuses_every_agent_start_endpoint(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        runner = SimpleNamespace(_draining=True, _external_drain_active=False)
        app = _make_admission_app(adapter)
        paths = (
            "/api/sessions/missing/chat",
            "/api/sessions/missing/chat/stream",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/runs",
        )

        with patch("gateway.run._gateway_runner_ref", lambda: runner):
            async with TestClient(TestServer(app)) as client:
                for path in paths:
                    response = await client.post(path, json={})
                    payload = await response.json()

                    assert response.status == 503
                    assert response.headers["Retry-After"] == "1"
                    assert payload["error"]["code"] == "gateway_draining"


# ---------------------------------------------------------------------------
# Shutdown interrupt coverage (#63529)
#
# The drain ACCOUNTS for every API turn (`active_agent_work_count()` sums
# `_pending_agent_requests` + `_inflight_agent_runs` + live `_active_run_tasks`)
# but `GatewayRunner._interrupt_running_agents()` only walked
# `self._running_agents`, which no API turn ever enters.  So an API turn held
# the drain open for the full timeout and was then amputated by
# `_kill_tool_subprocesses("post-interrupt")` with no cooperative interrupt.
#
# `/v1/runs` is only one of seven API agent-entry points.  The other six all
# funnel through `_run_agent()` — both session-chat routes and
# `/v1/chat/completions` + `/v1/responses` in streaming and non-streaming form
# — and none of them has a run_id, so `_active_run_agents` cannot reach them.
# ---------------------------------------------------------------------------


def _parked_agent(loop, started: asyncio.Event, release: threading.Event) -> MagicMock:
    """A mock agent whose turn parks inside ``run_conversation`` until released.

    ``request_hard_interrupt`` falls back to ``agent.interrupt(reason)`` for an
    unspecced ``MagicMock`` — ``inspect.getattr_static`` refuses to invent
    ``hard_interrupt`` on a ``__getattr__`` proxy — which is exactly the ABI
    teknium1's review asked this regression to verify.
    """
    agent = MagicMock()
    agent.session_id = None
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent._last_compaction_in_place = False
    agent._hermes_api_runtime = {}

    def _park(user_message=None, conversation_history=None, task_id=None):
        loop.call_soon_threadsafe(started.set)
        release.wait(_TURN_UNBLOCK_TIMEOUT)
        return {"final_response": "done", "messages": [], "api_calls": 0, "tools": []}

    agent.run_conversation.side_effect = _park
    # A real agent unwinds its turn on interrupt; releasing here models that so
    # the parked executor thread can finish.
    agent.interrupt.side_effect = lambda *_a, **_k: release.set()
    return agent


class _SettlingApiAdapter:
    """API adapter double whose work clears a few polls AFTER it is interrupted.

    The poll count is the deterministic quantity under test: it makes "the
    settle window kept polling API work" observable without timing anything.
    """

    def __init__(self, polls_to_settle: int = 3):
        self._polls_to_settle = polls_to_settle
        self.interrupt_reasons: list = []

    def active_agent_work_count(self) -> int:
        if not self.interrupt_reasons:
            return 1
        if self._polls_to_settle > 0:
            self._polls_to_settle -= 1
            return 1
        return 0

    def interrupt_active_runs(self, reason: str) -> int:
        self.interrupt_reasons.append(reason)
        return 1

    @property
    def settled(self) -> bool:
        """Non-consuming view of the same state, safe to read from a spy."""
        return bool(self.interrupt_reasons) and self._polls_to_settle == 0


def _make_async_noop():
    async def _noop(*args, **kwargs):
        return None

    return _noop


class TestRunAgentRegistersForShutdownInterrupt:
    @pytest.mark.asyncio
    async def test_run_agent_registers_and_unregisters_the_agent(self):
        """One registration inside ``_run_agent`` covers all six of its callers.

        Only two callers pass ``agent_ref``, and that lands in a caller-local
        list rather than any registry, so it is not a usable hook.
        """
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        agent = MagicMock()
        agent.session_id = None
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent._last_compaction_in_place = False
        observed = {}

        def _record(user_message=None, conversation_history=None, task_id=None):
            observed["during"] = dict(adapter._shutdown_interruptible_agents)
            return {"final_response": "done", "messages": [], "api_calls": 0, "tools": []}

        agent.run_conversation.side_effect = _record

        with patch.object(adapter, "_create_agent", return_value=agent):
            await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="s1",
            )

        assert list(observed["during"].values()) == [agent]
        assert adapter._shutdown_interruptible_agents == {}

    @pytest.mark.asyncio
    async def test_agent_is_unregistered_when_the_turn_raises(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        agent = MagicMock()
        agent.run_conversation.side_effect = RuntimeError("boom")

        with patch.object(adapter, "_create_agent", return_value=agent):
            with pytest.raises(RuntimeError):
                await adapter._run_agent(
                    user_message="hello",
                    conversation_history=[],
                    session_id="s1",
                )

        assert adapter._shutdown_interruptible_agents == {}


class TestInterruptActiveRuns:
    def test_interrupts_v1_runs_agents(self):
        """The ``/v1/runs`` coverage #63963 established stays green."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        agent = MagicMock()
        adapter._active_run_agents = {"run-1": agent}

        assert adapter.interrupt_active_runs("gateway shutdown") == 1
        agent.interrupt.assert_called_once_with("gateway shutdown")

    def test_interrupts_each_agent_exactly_once_across_both_registries(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        shared = MagicMock()
        run_only = MagicMock()
        turn_only = MagicMock()
        adapter._active_run_agents = {"run-1": run_only, "run-2": shared}
        adapter._shutdown_interruptible_agents = {
            id(shared): shared,
            id(turn_only): turn_only,
        }

        assert adapter.interrupt_active_runs("gateway shutdown") == 3
        shared.interrupt.assert_called_once_with("gateway shutdown")
        run_only.interrupt.assert_called_once_with("gateway shutdown")
        turn_only.interrupt.assert_called_once_with("gateway shutdown")

    def test_one_bad_agent_does_not_strand_the_others(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        exploding = MagicMock()
        exploding.interrupt.side_effect = RuntimeError("already torn down")
        no_abi = object()  # exposes neither hard_interrupt nor interrupt
        healthy = MagicMock()
        adapter._shutdown_interruptible_agents = {
            id(exploding): exploding,
            id(no_abi): no_abi,
            id(healthy): healthy,
        }

        assert adapter.interrupt_active_runs("gateway shutdown") == 1
        healthy.interrupt.assert_called_once_with("gateway shutdown")


class TestShutdownInterruptReachesEveryApiTurn:
    @pytest.mark.asyncio
    async def test_chat_completions_turn_is_interrupted(self):
        """A non-``/v1/runs`` API turn, end to end through the real handler.

        This is teknium1's named acceptance criterion on #63963: the drain
        counts this turn, so the shutdown interrupt must reach it.
        """
        runner, _adapter = make_restart_runner()
        api = APIServerAdapter(PlatformConfig(enabled=True))
        runner.adapters = {Platform.API_SERVER: api}
        app = _make_admission_app(api)

        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        agent = _parked_agent(loop, started, release)

        try:
            with patch.object(api, "_create_agent", return_value=agent):
                async with TestClient(TestServer(app)) as client:
                    request = asyncio.ensure_future(
                        client.post(
                            "/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": "hi"}]},
                        )
                    )
                    await asyncio.wait_for(started.wait(), _TURN_UNBLOCK_TIMEOUT)

                    # The drain sees this turn ...
                    assert runner._active_api_run_count() == 1
                    # ... and it is not in _running_agents, so only the API
                    # hook can reach it.
                    assert runner._running_agents == {}

                    runner._interrupt_running_agents(_INTERRUPT_REASON_GATEWAY_SHUTDOWN)

                    agent.interrupt.assert_called_once_with(
                        _INTERRUPT_REASON_GATEWAY_SHUTDOWN
                    )
                    response = await asyncio.wait_for(request, _TURN_UNBLOCK_TIMEOUT)
                    assert response.status == 200
        finally:
            release.set()

        assert api._shutdown_interruptible_agents == {}

    @pytest.mark.asyncio
    async def test_session_chat_sse_turn_is_interrupted(self, tmp_path):
        """The SSE session-chat route is a second, differently shaped caller."""
        runner, _adapter = make_restart_runner()
        api = APIServerAdapter(PlatformConfig(enabled=True))
        session_db = SessionDB(tmp_path / "state.db")
        api._session_db = session_db
        runner.adapters = {Platform.API_SERVER: api}
        app = _make_admission_app(api)
        session_id = session_db.create_session("sse-session", "api_server")

        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        agent = _parked_agent(loop, started, release)

        try:
            with patch.object(api, "_create_agent", return_value=agent):
                async with TestClient(TestServer(app)) as client:
                    request = asyncio.ensure_future(
                        client.post(
                            f"/api/sessions/{session_id}/chat/stream",
                            json={"message": "hi"},
                        )
                    )
                    await asyncio.wait_for(started.wait(), _TURN_UNBLOCK_TIMEOUT)

                    assert runner._active_api_run_count() == 1
                    assert runner._running_agents == {}

                    runner._interrupt_running_agents(_INTERRUPT_REASON_GATEWAY_SHUTDOWN)

                    agent.interrupt.assert_called_once_with(
                        _INTERRUPT_REASON_GATEWAY_SHUTDOWN
                    )
                    response = await asyncio.wait_for(request, _TURN_UNBLOCK_TIMEOUT)
                    assert response.status == 200
                    await asyncio.wait_for(response.text(), _TURN_UNBLOCK_TIMEOUT)
        finally:
            release.set()
            close = getattr(session_db, "close", None)
            if callable(close):
                close()

        assert api._shutdown_interruptible_agents == {}

    def test_interrupt_running_agents_is_a_noop_without_an_api_adapter(self):
        """The hook is duck-typed — an adapterless runner must not raise."""
        runner, _adapter = make_restart_runner()
        runner.adapters = {}

        runner._interrupt_running_agents(_INTERRUPT_REASON_GATEWAY_SHUTDOWN)

        assert runner._interrupt_api_server_runs("x") == 0


class TestShutdownSettleWindow:
    @pytest.mark.asyncio
    async def test_settle_window_waits_for_interrupted_api_work(self, monkeypatch):
        """The interrupt is cooperative, so the settle window must poll API work.

        Otherwise the window closes the instant ``_running_agents`` is empty —
        which it always is for API turns — and the post-interrupt tool kill
        lands on a turn that was asked to stop microseconds earlier.
        """
        import tools.browser_tool as _bt
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt

        runner, adapter = make_restart_runner()
        attach_real_stop(runner)
        runner._restart_drain_timeout = 0.01  # force the drain-timeout path
        adapter.disconnect = _make_async_noop()
        api = _SettlingApiAdapter()
        runner.adapters = {Platform.TELEGRAM: adapter, Platform.API_SERVER: api}

        settled_at_kill: list = []

        def _spy_kill_all(task_id=None):
            settled_at_kill.append(api.settled)
            return 0

        monkeypatch.setattr(_pr.process_registry, "kill_all", _spy_kill_all)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        with patch("gateway.status.remove_pid_file"), \
             patch("gateway.status.write_runtime_status"), \
             patch("cron.scheduler.mark_job_run"):
            await runner.stop()

        assert api.interrupt_reasons == [_INTERRUPT_REASON_GATEWAY_SHUTDOWN]
        assert settled_at_kill, "post-interrupt tool kill never ran"
        assert settled_at_kill[0] is True, (
            "post-interrupt tool kill ran while the interrupted API turn was "
            "still unwinding"
        )

    @pytest.mark.asyncio
    async def test_api_work_still_live_at_settle_exit_is_reinterrupted(
        self, monkeypatch
    ):
        """A /v1/runs agent can materialize AFTER the one-shot interrupt.

        The task is counted via ``_active_run_tasks`` from admission, but
        ``_active_run_agents[run_id]`` is populated only once ``_create_agent``
        returns — an agent landing in that window missed the single interrupt
        and previously went straight to the tool-subprocess kill. The settle
        loop must re-signal when API work is still live at exit.
        """
        import tools.browser_tool as _bt
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt

        runner, adapter = make_restart_runner()
        attach_real_stop(runner)
        runner._restart_drain_timeout = 0.01
        adapter.disconnect = _make_async_noop()
        api = _SettlingApiAdapter(polls_to_settle=10_000)  # never settles
        runner.adapters = {Platform.TELEGRAM: adapter, Platform.API_SERVER: api}

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 0)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        # Accelerate the loop clock: each time() call advances 1s of virtual
        # time, so the 5s settle deadline expires after a handful of polls
        # instead of 5 real seconds. Relative deadline math is preserved.
        loop = asyncio.get_running_loop()
        _real_time = type(loop).time
        _skew = [0.0]

        def _fast_time(self):
            _skew[0] += 1.0
            return _real_time(self) + _skew[0]

        monkeypatch.setattr(type(loop), "time", _fast_time)
        try:
            with patch("gateway.status.remove_pid_file"), \
                 patch("gateway.status.write_runtime_status"), \
                 patch("cron.scheduler.mark_job_run"):
                await runner.stop()
        finally:
            monkeypatch.undo()

        # One shot from _interrupt_running_agents + one re-signal at settle
        # exit because API work was still live.
        assert api.interrupt_reasons == [
            _INTERRUPT_REASON_GATEWAY_SHUTDOWN,
            _INTERRUPT_REASON_GATEWAY_SHUTDOWN,
        ]


