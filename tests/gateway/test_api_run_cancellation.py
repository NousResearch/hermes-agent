"""Lost admission responses must not turn cancellation into a second submission."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms.api_server_runs import _http_routes


def test_cancel_fences_delayed_admission_across_connections_and_restart(tmp_path, monkeypatch):
    path = str(tmp_path / "runs.db")
    first, second = RunIdempotencyStore(path), RunIdempotencyStore(path)
    try:
        assert first.lookup("owner", "key", "payload")[0] == "missing"
        outcome, cancellation = second.cancel_key("owner", "key")
        assert outcome == "prevented"
        assert cancellation["expires_at"] >= time.time() + 86390
        assert first.reserve("owner", "key", "payload", "late", {"status": "queued"})[0] == "cancelled"
        assert first.reserve("owner", "key", "different", "late", {"status": "queued"})[0] == "cancelled"
        assert first.reserve("another-owner", "key", "payload", "other", {"status": "queued"})[0] == "created"
    finally:
        first.close()
        second.close()
    reopened = RunIdempotencyStore(path)
    try:
        assert reopened.lookup("owner", "key", "payload")[0] == "cancelled"
        assert reopened.cancel_key("owner", "key")[1]["expires_at"] >= cancellation["expires_at"]
        with ThreadPoolExecutor(2) as executor:
            cancel = executor.submit(reopened.cancel_key, "owner", "race")
            reserve = executor.submit(reopened.reserve, "owner", "race", "payload", "race-run", {"status": "queued"})
            outcomes = (cancel.result()[0], reserve.result()[0])
        assert outcomes in {("prevented", "cancelled"), ("admitted", "created")}
        # Cancellation protection is independent of receipt pruning and only expires at its horizon.
        now = time.time()
        horizon = now + 2 * reopened.RETENTION_SECONDS
        assert reopened.cancel_key("owner", "expiry", retention_until=horizon)[1]["expires_at"] == horizon
        assert reopened.cancel_key("owner", "expiry")[1]["expires_at"] == horizon
        monkeypatch.setattr(time, "time", lambda: horizon - 1)
        assert reopened.lookup("owner", "expiry", "payload")[0] == "cancelled"
        monkeypatch.setattr(time, "time", lambda: horizon)
        assert reopened.lookup("owner", "expiry", "payload")[0] == "missing"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_http_cancel_never_launches_and_fails_closed_during_admission(tmp_path, monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "owner-secret"}))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "http.db"))
    app = web.Application()
    for method, path, handler in _http_routes(adapter):
        app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        headers = {"Authorization": "Bearer owner-secret", "Idempotency-Key": "lost-response"}
        denied = await client.post("/v1/runs/cancel", headers={"Idempotency-Key": "lost-response"})
        assert denied.status == 401
        prevented = await client.post("/v1/runs/cancel", headers=headers)
        assert prevented.status == 200
        assert (await prevented.json())["status"] == "prevented"
        delayed = await client.post("/v1/runs", headers=headers, json={"input": "Do not execute this"})
        assert delayed.status == 409
        assert (await delayed.json())["error"]["code"] == "idempotency_key_cancelled"
        assert not adapter._active_run_tasks
        assert not adapter._run_streams
        # Suspend the real admission path after reservation but before execution registration.
        entered, release = asyncio.Event(), asyncio.Event()
        original_admit = adapter._admit_to_live_bot_chat

        async def paused_admit(*args):
            entered.set()
            await release.wait()
            return await original_admit(*args)

        monkeypatch.setattr(adapter, "_admit_to_live_bot_chat", paused_admit)
        racing_headers = {**headers, "Idempotency-Key": "suspended-admission"}
        submission = asyncio.create_task(client.post(
            "/v1/runs", headers=racing_headers,
            json={"input": "Do not execute this", "session_id": "cancel-race-session"}))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            uncertain = await client.post("/v1/runs/cancel", headers=racing_headers)
            assert uncertain.status == 409
            original_id = (await uncertain.json())["run_id"]
        finally:
            release.set()
        accepted = await submission
        assert accepted.status == 202
        assert (await accepted.json())["run_id"] == original_id
        from tests.waiting import async_wait_until
        await async_wait_until(
            lambda: adapter._run_statuses[original_id]["status"] == "cancelled", timeout=5)
        assert not adapter._active_run_agents
        terminal = await client.post("/v1/runs/cancel", headers=racing_headers)
        assert (await terminal.json())["status"] == "cancelled"
        # The real store contains an admitted receipt, while no execution task is registered.
        # This is also the state during the asynchronous live-owner mailbox admission.
        from aiohttp.test_utils import make_mocked_request
        request = make_mocked_request("POST", "/v1/runs/cancel", headers=headers)
        scope = adapter._run_idempotency_scope(request)
        adapter._run_idempotency_store.reserve(
            scope, "admitted", "payload", "run-original", {"run_id": "run-original", "status": "queued"},
            owner_pid=adapter._run_owner_pid, owner_started=adapter._run_owner_started)
        headers["Idempotency-Key"] = "admitted"
        pending = await client.post("/v1/runs/cancel", headers=headers)
        assert pending.status == 409
        assert (await pending.json())["error"]["code"] == "run_not_active"
        # Once the original local task exists, cancellation addresses it without creating work.
        task = asyncio.create_task(asyncio.Event().wait())
        adapter._active_run_tasks["run-original"] = task
        try:
            stopping = await client.post("/v1/runs/cancel", headers=headers)
            assert stopping.status == 200
            assert await stopping.json() == {"run_id": "run-original", "status": "stopping"}
            adapter._set_run_status("run-original", "running", delivery_id="external-mailbox")
            external = await client.post("/v1/runs/cancel", headers=headers)
            assert external.status == 409
            assert (await external.json())["error"]["code"] == "run_not_interruptible"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    adapter._run_idempotency_store.close()


def test_cancel_by_key_uses_native_queued_admission(tmp_path):
    from tests.gateway.test_api_authority_execution import _probe
    _probe(tmp_path, runs="cancel-by-key")
