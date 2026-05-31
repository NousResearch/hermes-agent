import asyncio
import time

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.platforms.webhook import WebhookAdapter


def _adapter(tmp_path, monkeypatch):
    from gateway import webhook_queue

    monkeypatch.setattr(webhook_queue, "get_hermes_home", lambda: tmp_path)
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "secret": "INSECURE_NO_AUTH",
                "routes": {
                    "notion-sync": {
                        "secret": "INSECURE_NO_AUTH",
                        "prompt": "Notion event: {type}",
                        "deliver": "log",
                    }
                },
                "queue_retry_delay_seconds": 5,
            },
        )
    )


def _event(adapter, chat_id="webhook:notion-sync:delivery-1"):
    return MessageEvent(
        text="process",
        message_type=MessageType.TEXT,
        source=adapter.build_source(
            chat_id=chat_id,
            chat_name="webhook/notion-sync",
            chat_type="webhook",
            user_id="webhook:notion-sync",
            user_name="notion-sync",
        ),
        raw_message={"id": "delivery-1", "type": "comment.created"},
        message_id="delivery-1",
    )


@pytest.mark.asyncio
async def test_webhook_request_is_enqueued_before_agent_dispatch(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway import webhook_queue

    adapter = _adapter(tmp_path, monkeypatch)
    captured_events = []

    async def _capture(event):
        captured_events.append(event)

    adapter.handle_message = _capture
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/webhooks/notion-sync",
            json={"id": "delivery-1", "type": "comment.created"},
            headers={"X-Request-ID": "delivery-1"},
        )
        assert response.status == 202

    await asyncio.sleep(0.05)

    assert len(captured_events) == 1
    assert adapter._queue_record_by_chat_id["webhook:notion-sync:delivery-1"]
    stats = webhook_queue.stats()
    assert stats["inflight"] == 1
    assert stats["total"] == 1


@pytest.mark.asyncio
async def test_webhook_queue_done_only_on_processing_success(tmp_path, monkeypatch):
    from gateway import webhook_queue

    adapter = _adapter(tmp_path, monkeypatch)
    record = webhook_queue.make_record(
        route_name="notion-sync",
        delivery_id="delivery-1",
        event_type="comment.created",
        payload={"id": "delivery-1"},
        prompt="process",
        deliver_config={"deliver": "log", "payload": {}},
    )
    queue_id = webhook_queue.enqueue(record)
    adapter._queue_record_by_chat_id["webhook:notion-sync:delivery-1"] = queue_id

    await adapter.on_processing_complete(_event(adapter), ProcessingOutcome.SUCCESS)

    assert webhook_queue.stats()["total"] == 0


@pytest.mark.asyncio
async def test_webhook_queue_retry_on_processing_failure(tmp_path, monkeypatch):
    from gateway import webhook_queue

    adapter = _adapter(tmp_path, monkeypatch)
    record = webhook_queue.make_record(
        route_name="notion-sync",
        delivery_id="delivery-1",
        event_type="comment.created",
        payload={"id": "delivery-1"},
        prompt="process",
        deliver_config={"deliver": "log", "payload": {}},
    )
    queue_id = webhook_queue.enqueue(record)
    webhook_queue.mark_inflight(queue_id, now=time.time())
    adapter._queue_record_by_chat_id["webhook:notion-sync:delivery-1"] = queue_id

    await adapter.on_processing_complete(_event(adapter), ProcessingOutcome.FAILURE)

    stats = webhook_queue.stats()
    assert stats["pending"] == 1
    assert stats["total"] == 1
    claimed = webhook_queue.claim_due(now=time.time() + 10)
    assert len(claimed) == 1
    assert claimed[0]["id"] == queue_id


def test_log_delivery_failure_markers_are_not_success(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)

    failed, error, classification = adapter._classify_protocol_failure(
        "Notion comment handling failed: validation_error parent.discussion_id is not allowed"
    )

    assert failed is True
    assert classification == "terminal"
    assert "validation_error" in error


def test_log_delivery_normal_content_is_success(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)

    failed, error, classification = adapter._classify_protocol_failure("Done: replied in Notion.")

    assert failed is False
    assert error is None
    assert classification == "success"


@pytest.mark.asyncio
async def test_terminal_protocol_failure_moves_to_dead_letter(tmp_path, monkeypatch):
    from gateway import webhook_queue

    adapter = _adapter(tmp_path, monkeypatch)
    record = webhook_queue.make_record(
        route_name="notion-sync",
        delivery_id="delivery-1",
        event_type="comment.created",
        payload={"id": "delivery-1"},
        prompt="process",
        deliver_config={"deliver": "log", "payload": {}},
    )
    queue_id = webhook_queue.enqueue(record)
    webhook_queue.mark_inflight(queue_id, now=time.time())
    event = _event(adapter)
    setattr(event, "_hermes_delivery_error", "terminal: Notion validation_error parent.discussion_id is not allowed")
    adapter._queue_record_by_chat_id["webhook:notion-sync:delivery-1"] = queue_id

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert webhook_queue.stats()["total"] == 0
    dead = webhook_queue.dead_letter_path()
    assert dead.exists()
    text = dead.read_text()
    assert "validation_error" in text
    assert "dead_letter" in text


def test_transient_protocol_failure_is_retryable(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)

    failed, error, classification = adapter._classify_protocol_failure(
        "Notion comment handling failed: 503 service_unavailable"
    )

    assert failed is True
    assert classification == "retryable"
    assert "503" in error
