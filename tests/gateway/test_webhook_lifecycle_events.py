"""Webhook delivery lifecycle on ``gateway_platform_event``: ``delivery_received`` before each agent
run, ``delivery_completed`` once at its true end, with the body before and after the route script."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import ProcessingOutcome
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH

ORIGINAL = {"title": "Disk alert", "details": {"severity": "high"}}
TRANSFORMED = {"title": "Disk alert", "summary": "Investigate disk"}


def _make_adapter(route_extra=None, *, handler=True):
    route = {"secret": _INSECURE_NO_AUTH, "prompt": "Investigate {title}", **(route_extra or {})}
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "rate_limit": 100, "routes": {"alarm": route}}))
    events = []
    if handler:
        async def _observe(envelope, source):
            events.append((envelope, source))

        adapter.set_platform_event_handler(_observe)
    order = []

    async def _handle(event):
        order.append(("run", event.message_id))

    adapter.handle_message = AsyncMock(side_effect=_handle)
    return adapter, events, order


def _request(body, delivery_id):
    raw = json.dumps(body).encode()
    request = MagicMock()
    request.headers = {"X-GitHub-Delivery": delivery_id}
    request.content_length = len(raw)
    request.match_info = {"route_name": "alarm"}
    request.method = "POST"

    async def _read():
        return raw

    request.read = _read
    return request


async def _settle_tasks(adapter):
    await asyncio.gather(*list(adapter._background_tasks))


def _transform_with(adapter, result):
    async def _inline(function, *args):
        return function(*args)

    adapter._route_processor.run_route_script = lambda _script, payload: (True, dict(result))
    return _inline


@pytest.mark.asyncio
async def test_received_precedes_run_and_carries_body_before_and_after_script(monkeypatch):
    adapter, events, order = _make_adapter({"script": "transform.py"})
    monkeypatch.setattr("gateway.platforms.webhook.asyncio.to_thread", _transform_with(adapter, TRANSFORMED))
    original_observe = adapter._platform_event_handler

    async def _observe(envelope, source):
        order.append(("event", envelope["event_type"]))
        await original_observe(envelope, source)

    adapter.set_platform_event_handler(_observe)

    response = await adapter._handle_webhook(_request(ORIGINAL, "d1"))
    assert response.status == 202
    await _settle_tasks(adapter)

    assert order == [("event", "delivery_received"), ("run", "d1")]
    [(envelope, source)] = events
    assert envelope["platform"] == "webhook" and envelope["event_type"] == "delivery_received"
    payload = envelope["payload"]
    assert payload["route"] == "alarm" and payload["delivery_id"] == "d1"
    # The chat id is the adapter's opaque per-delivery identity; observers key on it, never parse it.
    assert payload["chat_id"] == source.chat_id and payload["chat_id"].startswith("webhook:")
    assert payload["webhook_event_type"] == "unknown" and payload["profile"] is None
    assert payload["prompt"] == "Investigate Disk alert"
    assert payload["payload"] == TRANSFORMED and payload["raw_payload"] == ORIGINAL
    assert payload["created_at"].endswith("+00:00")
    assert payload["version"] == 1
    assert "secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_without_script_raw_payload_is_none():
    adapter, events, _order = _make_adapter()
    await adapter._handle_webhook(_request(ORIGINAL, "d2"))
    await _settle_tasks(adapter)
    [(envelope, _source)] = events
    assert envelope["payload"]["payload"] == ORIGINAL
    assert envelope["payload"]["raw_payload"] is None


@pytest.mark.asyncio
async def test_completed_fires_once_with_outcome():
    adapter, events, _order = _make_adapter()
    await adapter._handle_webhook(_request(ORIGINAL, "d3"))
    await _settle_tasks(adapter)
    event = adapter.handle_message.await_args.args[0]

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert [e["event_type"] for e, _ in events] == ["delivery_received", "delivery_completed"]
    completed = events[1][0]["payload"]
    assert completed["outcome"] == "failure"
    assert completed["delivery_id"] == "d3" and completed["chat_id"] == events[0][0]["payload"]["chat_id"]


@pytest.mark.asyncio
async def test_completion_survives_delivery_info_pruning():
    adapter, events, _order = _make_adapter()
    await adapter._handle_webhook(_request(ORIGINAL, "d4"))
    await _settle_tasks(adapter)
    event = adapter.handle_message.await_args.args[0]
    adapter._delivery_info.clear()  # a long run can outlive the idempotency TTL

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert events[-1][0]["event_type"] == "delivery_completed"
    assert events[-1][0]["payload"]["outcome"] == "success"


@pytest.mark.asyncio
async def test_observer_copy_is_isolated_from_the_run():
    adapter, _events, _order = _make_adapter()

    async def _mutate(envelope, source):
        envelope["payload"]["payload"]["title"] = "mutated"

    adapter.set_platform_event_handler(_mutate)
    await adapter._handle_webhook(_request(ORIGINAL, "d5"))
    await _settle_tasks(adapter)
    event = adapter.handle_message.await_args.args[0]
    assert event.raw_message["title"] == "Disk alert"
    assert event.metadata["webhook_lifecycle"]["payload"]["title"] == "Disk alert"


@pytest.mark.asyncio
async def test_failing_observer_never_blocks_the_run_or_completion():
    adapter, _events, _order = _make_adapter()

    async def _boom(envelope, source):
        raise RuntimeError("observer down")

    adapter.set_platform_event_handler(_boom)
    await adapter._handle_webhook(_request(ORIGINAL, "d6"))
    await _settle_tasks(adapter)
    adapter.handle_message.assert_awaited_once()
    await adapter.on_processing_complete(adapter.handle_message.await_args.args[0], ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_no_handler_is_a_no_op():
    adapter, _events, _order = _make_adapter(handler=False)
    task = adapter._spawn_agent_run(ORIGINAL, "p", "d7", 0.0, route_config={"prompt": "p"}, route_name="alarm",
                                    profile=None, event_type="push")
    assert isinstance(task, asyncio.Task)  # the coalescer's synchronous, Task-returning interface
    await task
    adapter.handle_message.assert_awaited_once()
    await adapter.on_processing_complete(adapter.handle_message.await_args.args[0], ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_coalesced_settlement_reports_the_settled_event(monkeypatch):
    adapter, events, _order = _make_adapter(
        {"script": "transform.py", "coalesce": {"key": "title", "window_seconds": 0.05}})
    monkeypatch.setattr("gateway.platforms.webhook.asyncio.to_thread", _transform_with(adapter, TRANSFORMED))
    first = {**ORIGINAL, "details": {"severity": "low"}}
    for delivery_id, body in (("c1", first), ("c2", ORIGINAL)):
        response = await adapter._handle_webhook(_request(body, delivery_id))
        assert json.loads(response.text)["status"] == "coalesced"
    assert events == []
    await asyncio.sleep(0.15)
    await _settle_tasks(adapter)

    [(envelope, _source)] = events
    assert envelope["event_type"] == "delivery_received"
    assert envelope["payload"]["delivery_id"] == "c2"
    assert envelope["payload"]["raw_payload"] == ORIGINAL  # the newest event's body as received


@pytest.mark.asyncio
async def test_deliver_only_routes_publish_no_lifecycle():
    adapter, events, _order = _make_adapter({"deliver_only": True, "deliver": "log"})
    adapter._direct_deliver = AsyncMock(return_value=MagicMock(success=True))
    response = await adapter._handle_webhook(_request(ORIGINAL, "d8"))
    assert response.status == 200
    assert events == []


@pytest.mark.asyncio
async def test_completion_reports_the_ingress_body_even_if_the_run_mutates_it():
    adapter, events, _order = _make_adapter()
    await adapter._handle_webhook(_request(ORIGINAL, "d9"))
    await _settle_tasks(adapter)
    event = adapter.handle_message.await_args.args[0]
    event.raw_message["title"] = "changed during run"

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    received, completed = (envelope["payload"] for envelope, _ in events)
    assert received["payload"]["title"] == completed["payload"]["title"] == "Disk alert"
    assert received["created_at"] == completed["created_at"]
