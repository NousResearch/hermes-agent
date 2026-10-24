"""Approval buttons resume Discord typing only when they release real pending work."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.adapter import ExecApprovalView, discord
from tests.gateway import test_discord_typing_pause as typing_tests
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


typing_fixture = typing_tests.typing


@pytest.mark.asyncio
async def test_button_resolution_preserves_next_worker_pause(typing_fixture, monkeypatch):
    adapter, _request, _clock = typing_fixture
    loop = asyncio.get_running_loop()
    first_published = asyncio.Event()
    second_published = threading.Event()
    key = "discord:typing-test"
    original_resolve = approval.resolve_gateway_approval

    def notify_first(data):
        adapter.pause_typing_for_chat("77")
        loop.call_soon_threadsafe(first_published.set)

    def notify_second(data):
        adapter.pause_typing_for_chat("77")
        second_published.set()

    def sequential_waits():
        first = _await_gateway_decision(
            key, notify_first, {"command": "first-command", "pattern_key": "first"}
        )
        assert first["resolved"] and first["choice"] == "once"
        return _await_gateway_decision(
            key, notify_second, {"command": "second-command", "pattern_key": "second"}
        )

    def resolve_then_allow_worker_to_continue(*args, **kwargs):
        count = original_resolve(*args, **kwargs)
        # Force a valid schedule: the awakened worker publishes its next approval
        # before the UI callback continues after the resolver call.
        assert second_published.wait(5), "worker did not publish its second approval"
        return count

    waiter = asyncio.create_task(asyncio.to_thread(sequential_waits))
    try:
        await asyncio.wait_for(first_published.wait(), 5)
        view = await approval_view(adapter)
        monkeypatch.setattr(approval, "resolve_gateway_approval", resolve_then_allow_worker_to_continue)
        await view._resolve(interaction(), "once", None, "gateway.exec_approval.action_once")
        assert approval.has_blocking_approval(key)
        assert "77" in adapter._typing_paused, "first decision erased the next approval's pause"
    finally:
        original_resolve(key, "deny", resolve_all=True)
        await asyncio.wait_for(waiter, 5)
        await adapter.stop_typing("77")


async def approval_view(adapter):
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=123)))
    adapter._client.get_channel = lambda channel_id: channel
    adapter._allowed_user_ids = {"user"}
    result = await adapter.send_exec_approval(
        chat_id="77", command="test-command", session_key="discord:typing-test",
        metadata={"thread_id": "88"},
    )
    assert result.success
    view = channel.send.await_args.kwargs["view"]
    assert isinstance(view, ExecApprovalView)
    return view


def interaction(user="user", edit=None):
    return SimpleNamespace(
        user=SimpleNamespace(id=user, display_name=user, roles=[]),
        channel_id=88, message=SimpleNamespace(embeds=[]),
        response=SimpleNamespace(send_message=AsyncMock(), edit_message=edit or AsyncMock()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["once", "session", "always", "deny"])
async def test_button_decision_resumes_before_ui_ack(typing_fixture, choice):
    adapter, request, clock = typing_fixture
    loop = asyncio.get_running_loop()
    published = asyncio.Event()
    editing = asyncio.Event()
    release_edit = asyncio.Event()

    def notify(data):
        adapter.pause_typing_for_chat("77")
        loop.call_soon_threadsafe(published.set)

    async def edit(**kwargs):
        editing.set()
        await release_edit.wait()

    waiter = asyncio.create_task(asyncio.to_thread(
        _await_gateway_decision, "discord:typing-test", notify,
        {"command": "test-command", "pattern_key": "test"},
    ))
    clicked = None
    try:
        await asyncio.wait_for(published.wait(), 5)
        # A different chat must not be resumed just because the card was routed there.
        adapter.pause_typing_for_chat("88")
        view = await approval_view(adapter)
        clicked = asyncio.create_task(view._resolve(
            interaction(edit=edit), choice, None, "gateway.exec_approval.action_once"))
        await asyncio.wait_for(editing.wait(), 5)
        decision = await asyncio.wait_for(asyncio.shield(waiter), 5)
        assert decision["resolved"] and decision["choice"] == choice
        assert "77" not in adapter._typing_paused
        assert "88" in adapter._typing_paused
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == 1
        assert not clicked.done(), "UI ack must still be stalled"
        # A duplicate card click must not unpause a later approval wait.
        adapter.pause_typing_for_chat("77")
        await view._resolve(interaction(), choice, None, "gateway.exec_approval.action_once")
        assert "77" in adapter._typing_paused
    finally:
        release_edit.set()
        if clicked:
            await asyncio.gather(clicked, return_exceptions=True)
        approval.resolve_gateway_approval("discord:typing-test", "deny")
        await asyncio.wait_for(waiter, 5)
        await adapter.stop_typing("77")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unauthorized", "expired", "resolver-error", "view-timeout"])
async def test_unsettled_approval_keeps_typing_paused(typing_fixture, monkeypatch, outcome):
    adapter, request, _clock = typing_fixture
    monkeypatch.setattr(discord.Color, "dark_grey", lambda: 0, raising=False)
    adapter.pause_typing_for_chat("77")
    view = await approval_view(adapter)
    if outcome == "resolver-error":
        def fail(*args, **kwargs):
            raise RuntimeError("queue unavailable")
        monkeypatch.setattr(approval, "resolve_gateway_approval", fail)
    if outcome == "view-timeout":
        await view.on_timeout()
    else:
        await view._resolve(interaction("stranger" if outcome == "unauthorized" else "user"),
                            "once", None, "gateway.exec_approval.action_once")
    try:
        assert "77" in adapter._typing_paused
        await adapter.send_typing("77")
        await asyncio.sleep(0)
        assert request.await_count == 0
    finally:
        await adapter.stop_typing("77")
