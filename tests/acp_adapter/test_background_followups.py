"""ACP keeps the prompt RPC open until background delegation results are ingested."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from acp.schema import PromptResponse

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionState


def _server() -> HermesACPAgent:
    server = HermesACPAgent.__new__(HermesACPAgent)
    server.session_manager = SimpleNamespace(save_session=lambda _sid: None)
    server._conn = None
    server._send_usage_update = AsyncMock()
    return server


def _state() -> SessionState:
    return SessionState(
        session_id="acp-session",
        agent=SimpleNamespace(session_id="hermes-head"),
        history=[],
        cancel_event=threading.Event(),
        is_running=True,
        runtime_lock=threading.Lock(),
        current_prompt_text="do work",
    )


@pytest.mark.asyncio
async def test_finish_turn_waits_for_real_delegation_and_ingests_its_completion() -> None:
    from tools.async_delegation import (
        _reset_for_tests,
        dispatch_async_delegation,
        get_durable_delegation,
    )

    server, state = _server(), _state()
    runner_started = threading.Event()
    release_runner = threading.Event()
    running_during_followup: list[bool] = []

    def runner():
        runner_started.set()
        assert release_runner.wait(timeout=3)
        return {"status": "completed", "summary": "child done"}

    async def fake_prompt(*, prompt, session_id, **_kwargs):
        assert session_id == "acp-session"
        assert "child done" in prompt[0].text
        running_during_followup.append(state.is_running)
        return PromptResponse(stop_reason="end_turn")

    server.prompt = fake_prompt  # type: ignore[method-assign]
    handle = dispatch_async_delegation(
        goal="prove ACP follow-up delivery",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="acp-session",
        parent_session_id="acp-session",
        runner=runner,
    )
    assert handle["status"] == "dispatched"
    assert await asyncio.to_thread(runner_started.wait, 2)

    try:
        finish = asyncio.create_task(
            server._finish_turn(
                state, "acp-session", None, {"messages": [], "final_response": ""}, "hermes-head", False
            )
        )
        await asyncio.sleep(0.05)
        assert not finish.done()
        assert state.is_running is True

        release_runner.set()
        response = await asyncio.wait_for(finish, timeout=3)

        assert response.stop_reason == "end_turn"
        assert running_during_followup == [True]
        assert state.is_running is False
        durable = get_durable_delegation(handle["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "delivered"
    finally:
        release_runner.set()
        _reset_for_tests()


@pytest.mark.asyncio
async def test_finish_turn_returns_immediately_when_no_delegation_is_live() -> None:
    from tools.async_delegation import _reset_for_tests

    server, state = _server(), _state()
    try:
        response = await asyncio.wait_for(
            server._finish_turn(
                state, "acp-session", None, {"messages": [], "final_response": ""}, "hermes-head", False
            ),
            timeout=2,
        )
        assert response.stop_reason == "end_turn"
        assert state.is_running is False
    finally:
        _reset_for_tests()


@pytest.mark.asyncio
async def test_followup_prompt_runs_before_finish_turn_releases_the_session() -> None:
    """The follow-up inherits the outer RPC's lease (is_running stays True) and a
    queued user prompt waits until after the completion turn, not before it."""
    from tools.async_delegation import _reset_for_tests

    server, state = _server(), _state()
    order: list[str] = []

    async def fake_prompt(*, prompt, session_id, **_kwargs):
        order.append(f"followup:{state.is_running}")
        return PromptResponse(stop_reason="end_turn")

    server.prompt = fake_prompt  # type: ignore[method-assign]
    state.queued_prompts = ["user typed while child ran"]

    async def fake_drain(_state, _sid, _conn):
        order.append("drain")

    notification = (
        {"type": "async_delegation", "delegation_id": "deleg-x", "session_key": "acp-session"},
        "child done",
    )
    drain_state = {"calls": 0}

    def fake_drain_notifications(**_kwargs):
        # Deliver once, then the queue is empty (every subsequent drain).
        drain_state["calls"] += 1
        return [notification] if drain_state["calls"] == 1 else []

    with patch.object(server, "_drain_queued_prompts", fake_drain), \
         patch("tools.async_delegation.has_live_for_session", return_value=False), \
         patch(
             "tools.process_registry.process_registry.drain_notifications",
             side_effect=fake_drain_notifications,
         ), \
         patch("tools.async_delegation.claim_event_delivery", return_value="claim"), \
         patch("tools.async_delegation.complete_event_delivery"):
        await server._finish_turn(
            state, "acp-session", None, {"messages": [], "final_response": ""}, "hermes-head", False
        )

    assert order == ["followup:True", "drain"]
    _reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "cancelled"])
async def test_unaccepted_followup_releases_claim_and_requeues_completion(outcome: str) -> None:
    server, state = _server(), _state()
    event = {
        "type": "async_delegation",
        "delegation_id": f"deleg-{outcome}",
        "session_key": "acp-session",
    }

    async def unaccepted_prompt(**_kwargs):
        if outcome == "error":
            raise RuntimeError("follow-up failed")
        state.cancel_event.set()
        return PromptResponse(stop_reason="cancelled")

    server.prompt = unaccepted_prompt  # type: ignore[method-assign]
    with (
        patch(
            "tools.process_registry.process_registry.drain_notifications",
            return_value=[(event, "child done")],
        ),
        patch("tools.process_registry.process_registry.completion_queue.put") as requeue,
        patch("tools.async_delegation.claim_event_delivery", return_value="claim") as claim,
        patch("tools.async_delegation.release_event_delivery") as release,
        patch("tools.async_delegation.complete_event_delivery") as complete,
    ):
        if outcome == "cancelled":
            response = await server._finish_turn(
                state, "acp-session", None, {"messages": [], "final_response": ""}, "hermes-head", False
            )
            assert response.stop_reason == "cancelled"
        else:
            await server._run_background_followups(state, "acp-session")

    claim.assert_called_once_with(event, "acp-background-followup")
    release.assert_called_once_with(event, "claim")
    requeue.assert_called_once_with(event)
    complete.assert_not_called()
    assert state.queued_prompts == []


@pytest.mark.asyncio
async def test_followup_on_cancelled_session_never_touches_the_queue() -> None:
    """Cancellation wins the race: with the session already cancelled, no completion
    is even drained — the event stays queued untouched for the next accepted prompt."""
    server, state = _server(), _state()
    state.cancel_event.set()
    event = {"type": "async_delegation", "delegation_id": "deleg-1", "session_key": "acp-session"}
    prompted: list[str] = []

    async def should_not_run(**_kwargs):
        prompted.append("ran")
        return PromptResponse(stop_reason="end_turn")

    server.prompt = should_not_run  # type: ignore[method-assign]
    with (
        patch(
            "tools.process_registry.process_registry.drain_notifications",
            return_value=[(event, "child done")],
        ) as drain,
        patch("tools.process_registry.process_registry.completion_queue.put") as requeue,
        patch("tools.async_delegation.claim_event_delivery", return_value="claim") as claim,
        patch("tools.async_delegation.release_event_delivery") as release,
        patch("tools.async_delegation.complete_event_delivery") as complete,
    ):
        await server._run_background_followups(state, "acp-session")

    assert prompted == []
    drain.assert_not_called()
    claim.assert_not_called()
    release.assert_not_called()
    requeue.assert_not_called()
    complete.assert_not_called()
