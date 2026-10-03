"""Regressions reproduced by the request capability review."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_request_lifecycle import NativeTransport, Transport, event, runner, scoped  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["gateway_request_lifecycle", "gateway_request_control"])
async def test_sync_request_callback_is_bounded_without_freezing_loop(scoped, monkeypatch, hook):
    from hermes_cli.plugins import get_plugin_manager
    from hermes_constants import get_hermes_home
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    home = get_hermes_home()
    observed = []

    def blocking(**kwargs):
        observed.append(get_hermes_home())
        started.set()
        try:
            release.wait(1)
            return "late result"
        finally:
            finished.set()

    scoped.register_hook(hook, blocking)
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05)
    task = asyncio.create_task(get_plugin_manager().ainvoke_hook(hook, turn_id="request"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        # The callback remains blocked while this coroutine can make progress.
        assert not finished.is_set()
        assert await task == []
        assert observed == [home]
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
        await task


@pytest.mark.asyncio
async def test_async_request_callback_keeps_admission_loop(scoped):
    from hermes_cli.plugins import get_plugin_manager
    loop = asyncio.get_running_loop()

    async def callback(**kwargs):
        assert asyncio.get_running_loop() is loop
        await asyncio.sleep(0)
        return "on loop"

    scoped.register_hook("gateway_request_lifecycle", callback)
    assert await get_plugin_manager().ainvoke_hook("gateway_request_lifecycle") == ["on loop"]


@pytest.mark.asyncio
async def test_sync_factory_awaitable_keeps_admission_loop(scoped):
    from hermes_cli.plugins import get_plugin_manager
    loop = asyncio.get_running_loop()

    def callback(**kwargs):
        async def result():
            assert asyncio.get_running_loop() is loop
            await asyncio.sleep(0)
            return "on loop"
        return result()

    scoped.register_hook("gateway_request_lifecycle", callback)
    assert await get_plugin_manager().ainvoke_hook("gateway_request_lifecycle") == ["on loop"]


@pytest.mark.asyncio
async def test_overlapping_finals_keep_each_request_formatter(scoped):
    from gateway.request_lifecycle import admit_request, begin_request, final_response
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    started, release = threading.Event(), threading.Event()

    def final(request, response):
        if request.facts.requester_id == "A":
            started.set()
            assert release.wait(2)
        return response + request.facts.requester_id

    scoped.register_hook("gateway_request_final", final)
    obj = runner(Transport())
    for owner in ("A", "B"):
        evt = event(user=owner)
        await admit_request(obj, evt, owner)
        begin_request(obj, evt, 4)
    first = asyncio.create_task(asyncio.to_thread(final_response, obj, "A", 4, "answer "))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert await asyncio.to_thread(final_response, obj, "B", 4, "answer ") == "answer B"
    finally:
        release.set()
        assert await first == "answer A"


@pytest.mark.asyncio
@pytest.mark.parametrize("same_owner", [False, True])
async def test_full_photo_queue_rejects_new_owner_but_keeps_album_merge(scoped, same_owner):
    from gateway.platforms.event import MessageType
    from gateway.request_lifecycle import admit_request
    from gateway.run_busy import GatewayBusySessionMixin
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = Transport()
    obj = runner(adapter)
    overflow = [event("already queued")]
    obj._BUSY_QUEUE_MAX_PENDING = 2
    obj._session_state = lambda key: SimpleNamespace(conversation=SimpleNamespace(queued_events=overflow))
    obj._overflow_queue = lambda key: overflow
    head = event(user="A", message="head")
    incoming = event("caption", user="A" if same_owner else "B", message="incoming")
    head.message_type = incoming.message_type = MessageType.PHOTO
    head.media_urls, incoming.media_urls = ["first.jpg"], ["second.jpg"]
    await admit_request(obj, head, "session")
    request = await admit_request(obj, incoming, "session")
    adapter._pending_messages["session"] = head
    GatewayBusySessionMixin._queue_or_replace_pending_event(obj, "session", incoming)
    assert obj._queue_depth("session", adapter=adapter) == 2
    assert incoming._gateway_accepted is same_owner
    assert request.active is same_owner
    assert head.media_urls == (["first.jpg", "second.jpg"] if same_owner else ["first.jpg"])


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [True, False])
async def test_successful_final_reconcile_edit_records_delivery_without_send(scoped, queued):
    from gateway.request_lifecycle import admit_request, finish_request
    scoped.register_hook("gateway_request_lifecycle", lambda **kwargs: None)
    adapter = NativeTransport()
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    obj = runner(adapter)
    evt = event()
    request = await admit_request(obj, evt, "session")
    finish_request(request, "final_ready")
    stream = SimpleNamespace(adapter=adapter, message_id="stream", final_content_delivered=True,
                             delivered_final_matches=lambda text: False)
    obj._deliver_media_from_response = AsyncMock()
    if queued:
        assert await obj._deliver_queued_first_response(
            "complete answer", evt.source, adapter, stream_consumer=stream,
            request_context=request) is True
    else:
        response = {"final_response": "complete answer", "response_transformed": True}
        ctx = SimpleNamespace(source=evt.source, session_key="session",
                              stream_consumer_holder=[stream], request_context=request)
        await obj._run_agent_mark_streamed_delivery(response, ctx)
        assert response["already_sent"] is True
    assert adapter.sent == []
    adapter.edit_message.assert_awaited_once()
    assert len(request.final_receipts) == 1
    receipt = request.final_receipts[0]
    assert receipt.success is True
    assert receipt.initiated_at is not None
    assert receipt.delivered_at >= receipt.initiated_at


@pytest.mark.asyncio
@pytest.mark.parametrize("provided_ids", [False, True])
async def test_parallel_registry_dispatch_keeps_all_request_observations(scoped, monkeypatch, provided_ids):
    import model_tools
    from agent.tool_execution_observer import observe_tool_execution
    from gateway.request_lifecycle import admit_request, begin_request, tool_observer_for_run
    from tools.registry import registry
    started, release = threading.Event(), threading.Event()
    observed = []

    def callback(request, stage, **kwargs):
        observed.append((request.facts.requester_id, stage, kwargs.get("tool_call_id")))
        if request.facts.requester_id == "A" and stage == "completed":
            started.set()
            assert release.wait(2)

    scoped.register_hook("gateway_request_tool", callback)
    obj = runner(Transport())
    for owner in ("A", "B"):
        evt = event(user=owner)
        await admit_request(obj, evt, owner)
        begin_request(obj, evt, 4)
    name = "request_review_fixture"
    registry.register(name=name, toolset="fixture", schema={"name": name, "parameters": {"type": "object"}},
                      handler=lambda args, **kwargs: "exact result")
    monkeypatch.setattr(model_tools, "_pre_dispatch_guards", lambda *args: (args[1], None))

    def execute(owner):
        with observe_tool_execution(tool_observer_for_run(obj, owner, 4)):
            return model_tools.handle_function_call(
                name, {}, tool_call_id=owner if provided_ids else None,
                skip_tool_request_middleware=True, skip_tool_execution_middleware=True)

    first = asyncio.create_task(asyncio.to_thread(execute, "A"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert await asyncio.to_thread(execute, "B") == "exact result"
        assert [(owner, stage) for owner, stage, _ in observed] == [
            ("A", "started"), ("A", "completed"), ("B", "started"), ("B", "completed")]
        identities = {owner: {identity for who, _, identity in observed if who == owner} for owner in ("A", "B")}
        assert all(len(values) == 1 and None not in values for values in identities.values())
        assert identities["A"] != identities["B"]
    finally:
        release.set()
        assert await first == "exact result"
        registry._tools.pop(name, None)
