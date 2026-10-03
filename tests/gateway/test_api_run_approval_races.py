"""Approval publication must agree with control requests on the owning event loop."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server, api_server_runs
from gateway.platforms.api_server import _SessionEventQueue, ThreadSafeAsyncQueue
from gateway.platforms.api_server_runs import _RunStream
from tests.gateway.test_api_server_runs import _claim_run, _create_runs_app, _make_adapter, _read_sse_frame
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


def _prepare_run(run_id, surface):
    adapter = _make_adapter()
    adapter._set_run_status(run_id, "running")
    adapter._active_run_agents[run_id] = SimpleNamespace(
        steer=lambda text: bool(text), interrupt=lambda *args, **kwargs: None)
    stream = adapter._run_streams[run_id] = _RunStream()
    _claim_run(adapter, run_id)
    if surface == "session":
        events = _SessionEventQueue("session-approval-race", run_id)
        notify = adapter._register_session_stream_approval(run_id, events, "message-approval-race")
        approval_events = events.queue
    elif surface == "completion":
        approval_events = ThreadSafeAsyncQueue()
        request = MagicMock(headers={})
        notify, _ = adapter._register_stream_approval(
            request, run_id, approval_events, "session-approval-race")
    else:
        adapter._run_approval_sessions[run_id] = run_id
        notify = api_server_runs._make_approval_notify(
            adapter, run_id, enqueue_event=stream.put_nowait, _api_server=api_server)
        approval_events = None
    return adapter, stream, notify, approval_events


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["run", "session", "completion"])
async def test_approval_arriving_after_control_snapshot_is_published(surface, monkeypatch):
    # The real wait enqueues from a worker thread precisely after the handler
    # snapshots an empty queue, before the handler writes its running status.
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    run_id = f"run-snapshot-race-{surface}"
    adapter, stream, notify, approval_events = _prepare_run(run_id, surface)
    snapshot_taken, notified = threading.Event(), threading.Event()
    decisions, errors = [], []
    secret = "ghp_" + "A" * 36
    list_pending = approval.list_gateway_approvals

    def snapshot_with_barrier(session_key):
        pending = list_pending(session_key)
        if session_key == run_id and not snapshot_taken.is_set():
            assert not pending
            snapshot_taken.set()
            assert notified.wait(5), "approval worker did not reach its notifier"
        return pending

    monkeypatch.setattr(approval, "list_gateway_approvals", snapshot_with_barrier)

    def observed_notify(data):
        try:
            notify(data)
        finally:
            notified.set()

    def wait_for_approval():
        try:
            assert snapshot_taken.wait(5), "control handler did not snapshot approvals"
            decisions.append(_await_gateway_decision(run_id, observed_notify, {
                "request_id": "approval-after-snapshot", "command": f"rm -r approval-fixture; printf {secret}",
                "description": "fixture approval", "allow_session": False, "allow_permanent": False,
            }))
        except BaseException as exc:
            errors.append(exc)
            notified.set()

    worker = threading.Thread(target=wait_for_approval, daemon=True)
    worker.start()
    try:
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            events_response = await client.get(f"/v1/runs/{run_id}/events")
            response = await client.post(f"/v1/runs/{run_id}/steer", json={"input": "continue"})
            assert response.status == 200, await response.text()
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "waiting_for_approval"
            assert status["approval"]["request_id"] == "approval-after-snapshot"
            assert status["approval"]["choices"] == ["once", "deny"]
            assert secret not in status["approval"]["command"]
            _, steered = await asyncio.wait_for(_read_sse_frame(events_response), 5)
            assert steered["event"] == "run.steered"
            assert "approval" not in steered
            if approval_events is None:
                _, request = await asyncio.wait_for(_read_sse_frame(events_response), 5)
            else:
                name, request = await asyncio.wait_for(approval_events.get(), 5)
                assert name == ("__approval__" if surface == "completion" else "approval.request")
                assert request["session_id"] == "session-approval-race"
                if surface == "session":
                    assert request["message_id"] == "message-approval-race"
            assert request["event"] == "approval.request"
            assert request["run_id"] == run_id
            assert request["request_id"] == status["approval"]["request_id"]
            assert secret not in request["command"]
            response = await client.post(f"/v1/runs/{run_id}/approval", json={
                "choice": "once", "request_id": request["request_id"]})
            assert response.status == 200, await response.text()
            assert (await response.json())["resolved"] == 1
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "running"
            assert "approval" not in status
            _, responded = await asyncio.wait_for(_read_sse_frame(events_response), 5)
            assert responded["event"] == "approval.responded"
            assert "approval" not in responded
            events_response.close()
    finally:
        snapshot_taken.set()
        notified.set()
        approval.resolve_gateway_approval(run_id, "deny", resolve_all=True)
        approval.unregister_gateway_notify(run_id)
        stream.put_nowait(None)
        await asyncio.to_thread(worker.join, 5)
    assert not worker.is_alive()
    assert not errors
    assert decisions and decisions[0]["choice"] == "once"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["run", "session", "completion"])
@pytest.mark.parametrize("action", ["resolve", "stop", "completed", "failed", "cancelled", "interrupted"])
async def test_delayed_approval_notification_cannot_revive_resolved_or_finished_run(surface, action, monkeypatch):
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    run_id = f"run-delayed-approval-{surface}-{action}"
    adapter, stream, notify, approval_events = _prepare_run(run_id, surface)
    entered, release = threading.Event(), threading.Event()
    decisions, errors = [], []
    loop = asyncio.get_running_loop()
    published = loop.create_future()

    def delayed_notify(data):
        entered.set()
        assert release.wait(5), "control request did not release the notifier"
        try:
            notify(data)
        finally:
            # Queued after publication, so awaiting this is a deterministic
            # loop barrier rather than a timing-dependent sleep.
            loop.call_soon_threadsafe(published.set_result, None)

    def wait_for_approval():
        try:
            decisions.append(_await_gateway_decision(run_id, delayed_notify, {
                "request_id": "approval-delayed", "command": "rm -r delayed-approval-fixture",
            }))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=wait_for_approval, daemon=True)
    worker.start()
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            if action == "resolve":
                response = await client.post(f"/v1/runs/{run_id}/approval", json={
                    "choice": "deny", "request_id": "approval-delayed"})
                assert response.status == 200, await response.text()
                assert (await response.json())["resolved"] == 1
                expected_status = "running"
            elif action == "stop":
                response = await client.post(f"/v1/runs/{run_id}/stop")
                assert response.status == 200, await response.text()
                expected_status = "stopping"
            else:
                adapter._set_run_status(run_id, action, last_event=f"run.{action}")
                expected_status = action
            release.set()
            await asyncio.wait_for(published, 5)
            status = await (await client.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == expected_status
            assert "approval" not in status
            assert all(event["event"] != "approval.request" for _, event in stream.backlog)
            if approval_events is not None:
                assert approval_events.empty()
            if action != "resolve":
                # The late response itself must also preserve the stop/terminal
                # state while acknowledging that the real waiter was released.
                response = await client.post(f"/v1/runs/{run_id}/approval", json={
                    "choice": "deny", "request_id": "approval-delayed"})
                assert response.status == 200, await response.text()
                assert (await response.json())["resolved"] == 1
                status = await (await client.get(f"/v1/runs/{run_id}")).json()
                assert status["status"] == expected_status
                assert "approval" not in status
    finally:
        release.set()
        approval.resolve_gateway_approval(run_id, "deny", resolve_all=True)
        approval.unregister_gateway_notify(run_id)
        stream.put_nowait(None)
        await asyncio.to_thread(worker.join, 5)
    assert not worker.is_alive()
    assert not errors
    assert decisions and decisions[0]["choice"] == "deny"
