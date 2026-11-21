"""Regression for #147302: startup admission stays pending until its replay finishes."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter
from tests.gateway.restart_test_helpers import make_restart_runner
from tests.waiting import async_wait_until


def make_a2a_runner(monkeypatch, queued, reply):
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    runner, _ = make_restart_runner(adapter)
    runner.adapters = {adapter.platform: adapter}
    runner._startup_restore_in_progress = queued
    runner._startup_restore_queue = []
    monkeypatch.setattr(
        runner,
        "_hm_pre_gateway_dispatch_hook",
        AsyncMock(side_effect=lambda event, source: event),
    )
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    monkeypatch.setattr(runner, "_admit_bot_message_for_source", lambda source: True)
    monkeypatch.setattr(runner, "_hm_estop_gate", lambda *args: None)
    monkeypatch.setattr(
        runner, "_hm_pending_reply_intercepts", AsyncMock(return_value=None)
    )
    executed = []

    async def execute_turn(event, source, session_key, is_internal):
        executed.append(event.message_id)
        if isinstance(reply, Exception):
            raise reply
        return True, reply

    # The model boundary is deterministic; admission, replay, delivery and task storage are real.
    monkeypatch.setattr(runner, "_hm_authority_admission", execute_turn)
    adapter.set_message_handler(runner._handle_message)
    return runner, adapter, executed


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [True, False])
@pytest.mark.parametrize("reply", ["executed once", None, RuntimeError("turn failed")])
async def test_a2a_startup_replay_owns_terminal_result(monkeypatch, queued, reply):
    runner, adapter, executed = make_a2a_runner(monkeypatch, queued, reply)
    adapter._loop = asyncio.get_running_loop()
    params = {"message": protocol.text_message(protocol.ROLE_USER, "answer once")}
    task, pending = adapter._prepare_task(params, "startup-peer")
    assert task is None
    task_id = pending["task_id"]
    try:
        if queued:
            await async_wait_until(
                lambda: runner._startup_restore_queue and not adapter._background_tasks,
                what="startup admission to finish parking the event",
            )
            assert executed == []
            assert not pending["future"].done()
            assert (
                adapter._rpc_tasks_get(1, {"id": task_id})["result"]["status"]["state"]
                == protocol.STATE_WORKING
            )
            assert await runner._drain_startup_restore_queue() == 1
        state, text = await asyncio.wait_for(asyncio.wrap_future(pending["future"]), 10)
        adapter._finalize_task(pending, state, text)
        await async_wait_until(
            lambda: not adapter._background_tasks, what="processing cleanup"
        )
        assert executed == [task_id]
        stored = adapter._rpc_tasks_get(2, {"id": task_id})["result"]
        if isinstance(reply, Exception):
            assert stored["status"]["state"] == protocol.STATE_FAILED
        else:
            assert stored["status"]["state"] == protocol.STATE_COMPLETED
            assert adapter.tasks.get(task_id)["reply"] == (reply or "")
        assert runner._startup_restore_queue == []
    finally:
        await adapter.cancel_background_tasks()
        adapter._pop_pending(task_id)


@pytest.mark.asyncio
async def test_startup_queued_http_request_returns_replayed_result(monkeypatch):
    runner, adapter, executed = make_a2a_runner(monkeypatch, True, "verified result")
    assert await adapter.connect()
    url = f"http://127.0.0.1:{adapter._httpd.server_port}"

    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:

        async def rpc(method, params):
            response = await client.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            )
            response.raise_for_status()
            return response.json()["result"]

        request = asyncio.create_task(
            rpc(
                "message/send",
                {
                    "message": protocol.text_message(protocol.ROLE_USER, "answer once"),
                },
            )
        )
        try:
            await async_wait_until(
                lambda: runner._startup_restore_queue and not adapter._background_tasks,
                what="HTTP input to reach startup queue",
            )
            task_id = runner._startup_restore_queue[0].message_id
            parked = await rpc("tasks/get", {"id": task_id})
            assert parked["status"]["state"] == protocol.STATE_WORKING
            assert not request.done()
            assert await runner._drain_startup_restore_queue() == 1
            completed = await request
            assert completed["status"]["state"] == protocol.STATE_COMPLETED
            stored = await rpc("tasks/get", {"id": task_id})
            assert stored["artifacts"][0]["parts"][0]["text"] == "verified result"
            assert executed == [task_id]
        finally:
            await adapter.cancel_background_tasks()
            await adapter.disconnect()
            await asyncio.gather(request, return_exceptions=True)
