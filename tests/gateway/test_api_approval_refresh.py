"""Approval refreshes retain durable state and the originating stream identity."""

import asyncio
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_run_approval_races import _prepare_run
from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _make_adapter,
    _make_slow_agent,
    _read_sse_frame,
    _use_idempotency_db,
)
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


@asynccontextmanager
async def _pending_approvals(run_id, notify):
    loop = asyncio.get_running_loop()
    workers, decisions, errors = [], [], []
    secret = "ghp_" + "B" * 36
    try:
        for index in range(2):
            published = loop.create_future()

            def observed_notify(data, done=published):
                notify(data)
                # This follows the bridge's publication on the same event loop.
                loop.call_soon_threadsafe(done.set_result, None)

            def wait_for_approval(index=index, observed_notify=observed_notify):
                try:
                    decisions.append(_await_gateway_decision(run_id, observed_notify, {
                        "request_id": f"approval-{index}",
                        "command": f"rm -r refresh-fixture-{index}; printf {secret}",
                        "allow_session": False, "allow_permanent": False,
                    }))
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=wait_for_approval, daemon=True)
            workers.append(worker)
            worker.start()
            await asyncio.wait_for(published, 5)
        yield secret
    finally:
        approval.resolve_gateway_approval(run_id, "deny", resolve_all=True)
        for worker in workers:
            await asyncio.to_thread(worker.join, 5)
        assert all(not worker.is_alive() for worker in workers)
        assert not errors
        assert len(decisions) == len(workers)


@pytest.mark.asyncio
async def test_partial_approval_refresh_survives_durable_replay(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    path = tmp_path / "runs.db"
    adapter: Any = _make_adapter()
    _use_idempotency_db(adapter, path)
    agent, ready, interrupted = _make_slow_agent()
    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: agent)
    restarted: Any = None
    run_id = None
    try:
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            body, headers = {"input": "approval refresh"}, {"Idempotency-Key": "refresh"}
            started = await client.post("/v1/runs", json=body, headers=headers)
            assert started.status == 202, await started.text()
            run_id = (await started.json())["run_id"]
            assert await asyncio.to_thread(ready.wait, 5)
            notify = approval._gateway_notify_cb(run_id)
            assert callable(notify)
            async with _pending_approvals(run_id, notify) as secret:
                initial = await (await client.get(f"/v1/runs/{run_id}")).json()
                assert initial["approval"]["request_id"] == "approval-0"
                answered = await client.post(f"/v1/runs/{run_id}/approval", json={
                    "choice": "once", "request_id": "approval-0"})
                assert answered.status == 200, await answered.text()
                live = await (await client.get(f"/v1/runs/{run_id}")).json()
                assert live["status"] == initial["status"] == "waiting_for_approval"
                assert live["approval"]["request_id"] == "approval-1"
                assert secret not in live["approval"]["command"]

                # Reopen the actual SQLite file in a fresh adapter without live
                # statuses. The original worker still owns this nonterminal run.
                _use_idempotency_db(adapter, path)
                restarted = cast(Any, _make_adapter())
                _use_idempotency_db(restarted, path)
                async with TestClient(TestServer(_create_runs_app(restarted))) as replay_client:
                    replay = await replay_client.post("/v1/runs", json=body, headers=headers)
                    assert replay.status == 202, await replay.text()
                    assert replay.headers["Idempotency-Replayed"] == "true"
                    assert (await replay.json())["run_id"] == run_id
                    hydrated = await (await replay_client.get(f"/v1/runs/{run_id}")).json()
                assert hydrated["status"] == "waiting_for_approval"
                assert hydrated["approval"] == live["approval"]

                answered = await client.post(f"/v1/runs/{run_id}/approval", json={
                    "choice": "deny", "request_id": "approval-1"})
                assert answered.status == 200, await answered.text()
                durable = adapter._run_idempotency_store.status_for_run(adapter._run_owners[run_id], run_id)
                assert durable["status"]["status"] == "running"
                assert "approval" not in durable["status"]
    finally:
        task = adapter._active_run_tasks.get(run_id)
        interrupted.set()
        if task is not None:
            await asyncio.wait_for(task, 5)
        adapter._run_idempotency_store.close()
        if restarted is not None:
            restarted._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["run", "session", "completion"])
@pytest.mark.parametrize("control", ["approval", "steer"])
async def test_control_refresh_preserves_approval_surface_identity(surface, control, monkeypatch):
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    run_id = f"run-refresh-{surface}-{control}"
    adapter, stream, notify, _ = _prepare_run(run_id, surface)
    try:
        async with AsyncExitStack() as pending:
            async with TestClient(TestServer(_create_runs_app(adapter))) as client:
                if control == "approval":
                    secret = await pending.enter_async_context(_pending_approvals(run_id, notify))
                else:
                    secret = "ghp_" + "B" * 36
                if surface == "run":
                    expected_identity = {}
                elif surface == "session":
                    expected_identity = {"message_id": "message-approval-race"}
                else:
                    expected_identity = {"session_id": "session-approval-race"}
                events = await client.get(f"/v1/runs/{run_id}/events", headers={
                    "Last-Event-ID": str(stream.next_seq - 1)})
                if control == "approval":
                    initial = await (await client.get(f"/v1/runs/{run_id}")).json()
                    assert all(initial["approval"].get(key) == value for key, value in expected_identity.items())
                    response = await client.post(f"/v1/runs/{run_id}/approval", json={
                        "choice": "once", "request_id": "approval-0"})
                else:
                    reading_body = asyncio.Event()
                    read_json = adapter._read_json_body

                    async def observed_read(request):
                        reading_body.set()
                        return await read_json(request)

                    monkeypatch.setattr(adapter, "_read_json_body", observed_read)

                    async def steer_body():
                        yield b'{"input":'
                        # Steering admits only running runs. The approval arrives
                        # during the asynchronous body read, as in a real turn.
                        await asyncio.wait_for(reading_body.wait(), 5)
                        await pending.enter_async_context(_pending_approvals(run_id, notify))
                        initial = await (await client.get(f"/v1/runs/{run_id}")).json()
                        assert all(initial["approval"].get(key) == value for key, value in expected_identity.items())
                        yield b'"continue after approval"}'

                    response = await client.post(f"/v1/runs/{run_id}/steer", data=steer_body(),
                                                 headers={"Content-Type": "application/json"})
                assert response.status == 200, await response.text()
                status = await (await client.get(f"/v1/runs/{run_id}")).json()
                expected_request = "approval-1" if control == "approval" else "approval-0"
                assert status["status"] == "waiting_for_approval"
                assert status["approval"]["request_id"] == expected_request
                assert all(status["approval"].get(key) == value for key, value in expected_identity.items())
                assert status["approval"]["choices"] == ["once", "deny"]
                assert secret not in status["approval"]["command"]
                _, event = await asyncio.wait_for(_read_sse_frame(events), 5)
                while event["event"] == "approval.request":
                    _, event = await asyncio.wait_for(_read_sse_frame(events), 5)
                if control == "approval":
                    assert event["event"] == "approval.responded"
                    assert event["approval"] == status["approval"]
                else:
                    assert event["event"] == "run.steered"
                    assert "approval" not in event
                events.close()
    finally:
        approval.unregister_gateway_notify(run_id)
        stream.put_nowait(None)
        adapter._run_idempotency_store.close()
