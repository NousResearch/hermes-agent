"""Typed selection must not turn inferred FIFO IDs into mandatory consent."""
import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.gateway.test_approval_requester_binding import _runner
from tests.hermes_cli import test_mandatory_native_approval as native_approval
from tools import approval, approval_gateway_wait

native = native_approval.native


def route(native, text):
    runner = _runner()
    platform = Platform(native.platform)
    runner.config.platforms[platform] = PlatformConfig(enabled=True, token="test-token")
    runner.adapters[platform] = native.adapter
    event = MessageEvent(text=text, message_id="typed-choice", source=SessionSource(
        platform=platform, chat_id="123", chat_type="dm", user_id="123", user_name="Operator"))
    approval.unregister_gateway_notify(native.session)
    native.session = runner._session_key_for_source(event.source)
    native.runner._ctx.session_key = native.session
    approval.register_gateway_notify(native.session, native.notify)
    return runner, event


def request_ids(native):
    return [entry["request_id"] for entry in approval.list_gateway_approvals(native.session)]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/approve", "/approve all"])
async def test_typed_grant_cannot_infer_mandatory_request(native, command):
    n = native
    runner, event = route(n, command)
    first = await n.start("A", 1)
    second = await n.start("B", 2)
    try:
        pending = request_ids(n)
        await runner._handle_approve_command(event)
        # Queue removal is synchronous: checking identity avoids an execution-thread race.
        assert request_ids(n) == pending
        assert n.executed == n.middleware == []
        await n.tap(1)
        assert (await asyncio.wait_for(second, 10))["ok"] == "B"
        assert n.executed == n.middleware == ["B"]
        await n.tap(0, "deny")
        assert "error" in await asyncio.wait_for(first, 10)
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(first, second)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/approve", "/approve all"])
async def test_typed_input_after_expiry_cannot_select_new_request(native, monkeypatch, command):
    n = native
    runner, event = route(n, command)
    with monkeypatch.context() as m:
        m.setattr(approval_gateway_wait, "_poll_event", lambda *a, **kw: "timeout")
        assert "error" in await asyncio.to_thread(n.dispatch, "expired")
    current = await n.start("new", 2)
    try:
        pending = request_ids(n)
        await runner._handle_approve_command(event)
        assert request_ids(n) == pending
        assert n.executed == n.middleware == []
        await n.tap(1)
        assert (await asyncio.wait_for(current, 10))["ok"] == "new"
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(current)


@pytest.mark.asyncio
async def test_typed_all_preserves_ordinary_grants_without_granting_mandatory(native):
    n = native
    runner, event = route(n, "/approve all")
    mandatory = await n.start("mandatory", 1)
    pending = request_ids(n)
    ordinary = approval_gateway_wait._ApprovalEntry({"command": "ordinary", "request_id": "ordinary"})
    approval._gateway_queues[n.session].append(ordinary)
    try:
        await runner._handle_approve_command(event)
        assert ordinary.event.is_set() and ordinary.result == "once"
        assert request_ids(n) == pending
        assert n.executed == n.middleware == []
        await n.tap(0)
        assert (await asyncio.wait_for(mandatory, 10))["ok"] == "mandatory"
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(mandatory)


@pytest.mark.asyncio
async def test_generic_deny_can_still_cancel_mandatory(native):
    n = native
    runner, event = route(n, "/deny all")
    task = await n.start("cancelled", 1)
    try:
        await runner._handle_deny_command(event)
        assert request_ids(n) == []
        assert "error" in await asyncio.wait_for(task, 10)
        assert n.executed == n.middleware == []
    finally:
        approval.unregister_gateway_notify(n.session)
        await asyncio.gather(task)
