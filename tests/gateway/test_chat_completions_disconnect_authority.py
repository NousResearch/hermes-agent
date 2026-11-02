"""Regression for #145750: chat-completions disconnects stop the canonical API turn."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web

from gateway.platforms.api_server import ThreadSafeAsyncQueue
from hermes_state_runtime import list_session_admissions
from tests.waiting import async_wait_until


class _ControlledAgent:
    def __init__(self):
        self.started = asyncio.Event()
        self.continue_work = asyncio.Event()
        self.interrupted = False
        self.tool_calls = []

    def hard_interrupt(self, _message=None, **_kwargs):
        self.interrupted = True
        self.continue_work.set()


def _spawn_stream(api, stream_q, completion_id, admission_future, *, session_id, user_message):
    request = SimpleNamespace(headers={})
    notify, on_done = api._register_stream_approval(request, completion_id, stream_q, session_id)
    return api._spawn_stream_agent(
        stream_q,
        on_done=on_done,
        approval_notify_callback=notify,
        approval_session_key=completion_id,
        admission_future=admission_future,
        user_message=user_message,
        conversation_history=[],
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_chat_completion_disconnect_interrupts_canonical_turn(api, owner, monkeypatch):
    """Cancelling the HTTP observer must stop the authority-owned execution, not only its task."""
    from gateway import session_finite

    controlled = _ControlledAgent()

    async def execute(authority, ref, row):
        authority.adopt_agent(ref.session_id, row["generation"], controlled)
        controlled.tool_calls.append("first")
        controlled.started.set()
        await controlled.continue_work.wait()
        if not controlled.interrupted:
            controlled.tool_calls.append("second")
        result = {
            "final_response": "interrupted" if controlled.interrupted else "completed",
            "messages": [],
            "interrupted": controlled.interrupted,
        }
        authority.pending_results[row["admission_id"]] = {"result": result, "usage": {}}
        return result["final_response"]

    monkeypatch.setattr(session_finite, "execute_finite_admission", execute)
    stream_q = ThreadSafeAsyncQueue()
    completion_id = "chatcmpl-authority-disconnect"
    admission_future = asyncio.get_running_loop().create_future()
    agent_task, agent_ref = _spawn_stream(
        api, stream_q, completion_id, admission_future,
        session_id="authority-disconnect", user_message="run controlled work",
    )
    response = AsyncMock(spec=web.StreamResponse)
    response.prepare = AsyncMock()
    writes = 0

    async def disconnect_on_delta(_data):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise ConnectionResetError("client disconnected")

    response.write = AsyncMock(side_effect=disconnect_on_delta)
    request = SimpleNamespace(headers={})
    with patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
        writer = asyncio.create_task(api._write_sse_chat_completion(
            request, completion_id, "test-model", 1, stream_q, agent_task, agent_ref,
            admission_future=admission_future,
        ))
        await asyncio.wait_for(controlled.started.wait(), 5)
        stream_q.put_nowait("first delta")
        await asyncio.wait_for(writer, 5)

    interrupted = controlled.interrupted
    controlled.continue_work.set()
    live_task = owner.sessions["authority-disconnect"].task
    if live_task is not None:
        await asyncio.wait_for(live_task, 5)
    await async_wait_until(lambda: not owner.api_observers,
                           timeout=5, what="the API observer was removed")

    assert interrupted, "the canonical agent kept running after its SSE observer disconnected"
    assert controlled.tool_calls == ["first"]
    assert completion_id not in api._run_admission_aliases
    assert completion_id not in api._stopping_run_ids


@pytest.mark.asyncio
async def test_disconnect_during_admission_cancels_only_its_queued_turn(api, owner, monkeypatch):
    """An early disconnect waits for commit, cancels that row, and leaves the lane head alone."""
    from gateway import session_finite

    blocker = _ControlledAgent()
    executed = []

    async def execute(authority, ref, row):
        text = row["payload"]["text"]
        executed.append(text)
        if text == "blocker":
            authority.adopt_agent(ref.session_id, row["generation"], blocker)
            blocker.started.set()
            await blocker.continue_work.wait()
        result = {"final_response": text, "messages": []}
        authority.pending_results[row["admission_id"]] = {"result": result, "usage": {}}
        return text

    monkeypatch.setattr(session_finite, "execute_finite_admission", execute)
    blocker_task = asyncio.create_task(api._run_agent(
        user_message="blocker", conversation_history=[], session_id="queued-disconnect",
    ))
    await asyncio.wait_for(blocker.started.wait(), 5)

    admission_lock = owner._api_admissions = asyncio.Lock()
    await admission_lock.acquire()
    stream_q = ThreadSafeAsyncQueue()
    completion_id = "chatcmpl-queued-disconnect"
    admission_future = asyncio.get_running_loop().create_future()
    queued_task, agent_ref = _spawn_stream(
        api, stream_q, completion_id, admission_future,
        session_id="queued-disconnect", user_message="queued",
    )
    response = AsyncMock(spec=web.StreamResponse)
    response.prepare = AsyncMock()
    disconnected = asyncio.Event()

    async def disconnect_before_commit(_data):
        disconnected.set()
        raise ConnectionResetError("client disconnected during admission")

    response.write = AsyncMock(side_effect=disconnect_before_commit)
    request = SimpleNamespace(headers={})
    with patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
        writer = asyncio.create_task(api._write_sse_chat_completion(
            request, completion_id, "test-model", 1, stream_q, queued_task, agent_ref,
            admission_future=admission_future,
        ))
        await asyncio.wait_for(disconnected.wait(), 5)
        assert not writer.done(), "disconnect cleanup skipped the in-flight admission"
        admission_lock.release()
        await asyncio.wait_for(writer, 5)

    blocker.continue_work.set()
    await asyncio.wait_for(blocker_task, 5)
    rows = {row["payload"]["text"]: row for row in list_session_admissions(
        owner.db, session_id="queued-disconnect", pending_only=False)}
    assert executed == ["blocker"]
    assert not blocker.interrupted
    assert rows["blocker"]["outcome"] == "completed"
    assert rows["queued"]["status"] == "terminal" and rows["queued"]["outcome"] == "cancelled"
    assert completion_id not in api._run_admission_aliases
    assert completion_id not in api._stopping_run_ids


@pytest.mark.asyncio
async def test_disconnect_completion_race_keeps_completed_outcome(api, owner, monkeypatch, caplog):
    """A completion that wins the disconnect race stays completed and is not logged as interrupted."""
    from gateway import session_finite

    controlled = _ControlledAgent()

    async def execute(authority, ref, row):
        authority.adopt_agent(ref.session_id, row["generation"], controlled)
        controlled.started.set()
        await controlled.continue_work.wait()
        result = {"final_response": "done", "messages": []}
        authority.pending_results[row["admission_id"]] = {"result": result, "usage": {}}
        return "done"

    monkeypatch.setattr(session_finite, "execute_finite_admission", execute)
    caplog.set_level("INFO", logger="gateway.platforms.api_server")
    stream_q = ThreadSafeAsyncQueue()
    completion_id = "chatcmpl-completion-race"
    admission_future = asyncio.get_running_loop().create_future()
    agent_task, agent_ref = _spawn_stream(
        api, stream_q, completion_id, admission_future,
        session_id="completion-race", user_message="finish first",
    )
    response = AsyncMock(spec=web.StreamResponse)
    response.prepare = AsyncMock()
    writes = 0

    async def complete_then_disconnect(_data):
        nonlocal writes
        writes += 1
        if writes == 2:
            controlled.continue_work.set()
            await async_wait_until(
                lambda: list_session_admissions(owner.db, session_id="completion-race", pending_only=False)[0][
                    "status"] == "terminal",
                timeout=5, what="the canonical turn completed",
            )
            raise ConnectionResetError("late disconnect")

    response.write = AsyncMock(side_effect=complete_then_disconnect)
    request = SimpleNamespace(headers={})
    with patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
        writer = asyncio.create_task(api._write_sse_chat_completion(
            request, completion_id, "test-model", 1, stream_q, agent_task, agent_ref,
            admission_future=admission_future,
        ))
        await asyncio.wait_for(controlled.started.wait(), 5)
        stream_q.put_nowait("final delta")
        await asyncio.wait_for(writer, 5)

    row = list_session_admissions(owner.db, session_id="completion-race", pending_only=False)[0]
    assert row["status"] == "terminal" and row["outcome"] == "completed"
    assert not controlled.interrupted
    assert "requested interruption" not in caplog.text
    assert "ended without a new interruption request" in caplog.text
    assert completion_id not in api._run_admission_aliases
    assert completion_id not in api._stopping_run_ids


@pytest.mark.asyncio
async def test_explicit_stop_still_interrupts_streamed_authority_turn(api, owner, monkeypatch):
    """The existing run Stop path and its generation fence remain the disconnect primitive."""
    from gateway import session_finite
    from gateway.platforms.api_server_authority_runs import stop_run

    controlled = _ControlledAgent()

    async def execute(authority, ref, row):
        authority.adopt_agent(ref.session_id, row["generation"], controlled)
        controlled.started.set()
        await controlled.continue_work.wait()
        result = {"final_response": "", "messages": [], "interrupted": controlled.interrupted}
        authority.pending_results[row["admission_id"]] = {"result": result, "usage": {}}
        return ""

    monkeypatch.setattr(session_finite, "execute_finite_admission", execute)
    stream_q = ThreadSafeAsyncQueue()
    completion_id = "chatcmpl-explicit-stop"
    admission_future = asyncio.get_running_loop().create_future()
    agent_task, _ = _spawn_stream(
        api, stream_q, completion_id, admission_future,
        session_id="explicit-stop", user_message="stop me",
    )
    await asyncio.wait_for(controlled.started.wait(), 5)
    projection = await stop_run(api, completion_id)
    await asyncio.wait_for(agent_task, 5)
    await async_wait_until(lambda: completion_id not in api._stopping_run_ids,
                           timeout=5, what="the explicit stop marker was removed")

    assert projection["status"] == "stopping"
    assert controlled.interrupted
    assert completion_id not in api._run_admission_aliases
    assert completion_id not in api._stopping_run_ids


@pytest.mark.asyncio
async def test_detached_observer_still_leaves_canonical_turn_running(api, owner, monkeypatch):
    """Observer cancellation alone remains resumable; only the HTTP disconnect path requests Stop."""
    from contextlib import suppress
    from gateway import session_finite

    controlled = _ControlledAgent()

    async def execute(authority, ref, row):
        authority.adopt_agent(ref.session_id, row["generation"], controlled)
        controlled.started.set()
        await controlled.continue_work.wait()
        result = {"final_response": "done", "messages": []}
        authority.pending_results[row["admission_id"]] = {"result": result, "usage": {}}
        return "done"

    monkeypatch.setattr(session_finite, "execute_finite_admission", execute)
    observer = asyncio.create_task(api._run_agent(
        user_message="detach observer", conversation_history=[], session_id="detached-observer",
    ))
    await asyncio.wait_for(controlled.started.wait(), 5)
    observer.cancel()
    with suppress(asyncio.CancelledError):
        await observer

    assert not controlled.interrupted
    controlled.continue_work.set()
    live_task = owner.sessions["detached-observer"].task
    if live_task is not None:
        await asyncio.wait_for(live_task, 5)
    row = list_session_admissions(owner.db, session_id="detached-observer", pending_only=False)[0]
    assert row["status"] == "terminal" and row["outcome"] == "completed"
    assert not owner.api_observers
