"""A run owns admission and control state until its actual execution settles."""

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms import api_server_runs
from tools import approval, approval_gateway_wait


def _request(body=None, *, key=None):
    request = make_mocked_request("POST", "/v1/runs", headers={"Idempotency-Key": key} if key else {})
    request.json = AsyncMock(return_value=body or {"input": "hello"})
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before_start", "shutdown_before_start", "running", "approval_wait", "worker_cancelled"])
async def test_cancellation_releases_run_only_after_execution_settles(when):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._max_concurrent_runs = 1
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    approval_released = asyncio.Event()
    release = threading.Event()
    agent = MagicMock()

    def run(**kwargs):
        if when == "worker_cancelled":
            raise asyncio.CancelledError
        if when == "approval_wait":
            decision = approval_gateway_wait._await_gateway_decision(
                run_id, lambda _data: loop.call_soon_threadsafe(ready.set), {"command": "probe"})
            assert decision["choice"] is None
            assert decision["cancelled"]  # Teardown wakes the waiter without granting approval.
            assert not approval.has_blocking_approval(run_id)
            loop.call_soon_threadsafe(approval_released.set)
        else:
            loop.call_soon_threadsafe(ready.set)
        assert release.wait(timeout=10)
        return {"final_response": "finished"}

    agent.run_conversation.side_effect = run
    with patch.object(adapter, "_create_agent", return_value=agent):
        response = await adapter._handle_runs(_request(key="lifetime"))
        assert response.status == 202
        run_id = json.loads(response.text)["run_id"]
        task = adapter._active_run_tasks[run_id]
        try:
            if when in {"running", "approval_wait"}:
                await asyncio.wait_for(ready.wait(), timeout=5)
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)  # Deliver cancellation on the next task step.
                    adapter._set_run_status(run_id, "waiting_for_approval")
                    assert adapter._run_statuses[run_id]["status"] == "stopping"
                    assert adapter.active_agent_work_count() == 1
                if when == "approval_wait":
                    await asyncio.wait_for(approval_released.wait(), timeout=5)
                replay = await adapter._handle_runs(_request(key="lifetime"))
                assert json.loads(replay.text)["run_id"] == run_id
                assert (await adapter._handle_runs(_request())).status == 429
            elif when in {"before_start", "shutdown_before_start"}:
                if when == "shutdown_before_start":
                    adapter.interrupt_active_runs("Gateway shutdown")
                task.cancel()
            release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
            expected = "interrupted" if when == "shutdown_before_start" else "cancelled"
            assert adapter._run_statuses[run_id]["status"] == expected
            assert run_id not in adapter._active_run_tasks
            assert run_id not in adapter._active_run_agents
            assert run_id not in adapter._run_approval_sessions
            assert adapter.active_agent_work_count() == 0
            if when in {"before_start", "shutdown_before_start"}:
                agent.run_conversation.assert_not_called()
            app = web.Application()
            app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
            async with TestClient(TestServer(app)) as client:
                events_response = await client.get(f"/v1/runs/{run_id}/events")
                text = await asyncio.wait_for(events_response.text(), timeout=5)
                events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
                assert events[-1]["event"] == f"run.{expected}"
        finally:
            release.set()
            approval.clear_session(run_id)
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
            adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_live_owner_lookup_keeps_admission_reserved():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._max_concurrent_runs = 1
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()

    async def lookup(*args):
        lookup_started.set()
        await release_lookup.wait()
        return None

    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "finished"}
    with patch.object(adapter, "_create_agent", return_value=agent), patch.object(
        api_server_runs, "_resolve_live_session_id", AsyncMock(return_value="session")
    ), patch.object(adapter, "_conversation_history_for_session", AsyncMock(return_value=[])), patch.object(
        adapter, "_admit_to_live_bot_chat", lookup
    ):
        admission = asyncio.create_task(adapter._handle_runs(_request({"input": "hello", "session_id": "session"})))
        try:
            await asyncio.wait_for(lookup_started.wait(), timeout=5)
            assert adapter.active_agent_work_count() == 1
            assert (await adapter._handle_runs(_request())).status == 429
        finally:
            release_lookup.set()
            await admission
            await asyncio.gather(*adapter._active_run_tasks.values(), return_exceptions=True)
            adapter._run_idempotency_store.close()
