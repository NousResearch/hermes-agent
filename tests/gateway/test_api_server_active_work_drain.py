"""Regression coverage for #63529 API-server shutdown draining.

API-server work is adapter-owned rather than tracked by
``GatewayRunner._running_agents``. The shutdown drain must account for the
same live state as the API concurrency limiter, including a ``/v1/runs`` task
that exists before its agent has been constructed, and it must refuse new API
turns once the gateway starts draining.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.restart_test_helpers import make_restart_runner


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

    def test_delegates_to_primary_api_adapter(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {
            Platform.API_SERVER: _make_api_adapter(inflight=2, queued_ids=["r1"])
        }
        assert runner._active_api_run_count() == 3

    def test_ignores_non_api_platforms(self):
        runner, _adapter = make_restart_runner()
        other = SimpleNamespace(
            platform=Platform.DISCORD,
            active_agent_work_count=lambda: 99,
        )
        runner.adapters = {Platform.DISCORD: other}
        assert runner._active_api_run_count() == 0

    def test_never_raises_on_broken_adapter(self):
        runner, _adapter = make_restart_runner()

        class Bad:
            platform = Platform.API_SERVER

            @staticmethod
            def active_agent_work_count() -> int:
                raise RuntimeError("boom")

        runner.adapters = {Platform.API_SERVER: Bad()}
        assert runner._active_api_run_count() == 0


class TestAPIServerAdapterWorkCount:
    def test_concurrency_limit_counts_other_pending_admissions(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._max_concurrent_runs = 1
        adapter._pending_agent_requests = 1

        response = adapter._concurrency_limited_response()

        assert response is not None
        assert response.status == 429

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

    def test_counts_pending_admission_before_agent_bookkeeping(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._pending_agent_requests = 1

        assert adapter.active_agent_work_count() == 1

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


class TestDrainWaitsForApiWork:
    @pytest.mark.asyncio
    async def test_drain_returns_immediately_when_nothing_active(self):
        runner, _adapter = make_restart_runner()
        runner.adapters = {}

        _snapshot, timed_out = await runner._drain_active_agents(5.0)

        assert timed_out is False

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

    @pytest.mark.asyncio
    async def test_external_drain_refuses_every_agent_start_endpoint(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        runner = SimpleNamespace(_draining=False, _external_drain_active=True)
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
                    assert payload["error"]["code"] == "gateway_draining"

    @pytest.mark.asyncio
    async def test_admitted_request_blocks_drain_before_agent_bookkeeping(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        runner, _adapter = make_restart_runner()
        runner.adapters = {Platform.API_SERVER: adapter}
        app = _make_admission_app(adapter)
        body_read_started = asyncio.Event()
        allow_body_read = asyncio.Event()

        async def delayed_read_json(_request):
            body_read_started.set()
            await allow_body_read.wait()
            return {"message": "hello"}, None

        with patch.object(
            adapter,
            "_get_existing_session_or_404",
            return_value=({}, None),
        ), patch.object(
            adapter,
            "_read_json_body",
            side_effect=delayed_read_json,
        ), patch.object(
            adapter,
            "_run_agent",
            new=AsyncMock(return_value=({"final_response": "done"}, {})),
        ):
            async with TestClient(TestServer(app)) as client:
                request_task = asyncio.create_task(
                    client.post("/api/sessions/missing/chat", json={})
                )
                await body_read_started.wait()

                assert adapter._pending_agent_requests == 1
                drain_task = asyncio.create_task(runner._drain_active_agents(2.0))
                await asyncio.sleep(0.1)
                assert not drain_task.done()

                allow_body_read.set()
                response = await request_task
                assert response.status == 200
                _snapshot, timed_out = await drain_task

        assert timed_out is False
