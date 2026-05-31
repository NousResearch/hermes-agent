import json

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _adapter(tmp_path, monkeypatch):
    import hermes_constants
    import gateway.platforms.webhook as webhook_module

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(webhook_module, "get_hermes_home", lambda: tmp_path, raising=False)
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
                        "notion_routing": "comment.created/comment.updated trigger agent immediately; comment.deleted ignored; all other events are queued in ~/.hermes/data/notion-sync/notion-sync-queue.jsonl for hourly batch processing",
                    }
                },
            },
        )
    )


@pytest.mark.asyncio
async def test_notion_non_comment_event_is_queued_without_agent_dispatch(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    adapter = _adapter(tmp_path, monkeypatch)
    captured_events = []

    async def _capture(event):
        captured_events.append(event)

    adapter.handle_message = _capture
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)

    payload = {
        "id": "notion-delivery-1",
        "type": "page.content_updated",
        "timestamp": "2026-05-25T00:00:00.000Z",
        "workspace_id": "workspace-1",
        "subscription_id": "subscription-1",
        "attempt_number": 1,
        "entity": {"id": "page-1", "type": "page"},
        "data": {
            "parent": {"id": "database-1", "type": "database", "data_source_id": "ds-1"},
            "updated_blocks": [{"id": "block-1", "type": "block"}],
        },
    }

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/webhooks/notion-sync",
            json=payload,
            headers={"X-Request-ID": "delivery-1"},
        )
        body = await response.json()

    assert response.status == 202
    assert body["status"] == "queued"
    assert captured_events == []

    queue_path = tmp_path / "data" / "notion-sync" / "notion-sync-queue.jsonl"
    queued = [json.loads(line) for line in queue_path.read_text().splitlines()]
    assert len(queued) == 1
    assert queued[0]["type"] == "page.content_updated"
    assert queued[0]["page_id"] == "page-1"
    assert queued[0]["parent_id"] == "database-1"
    assert queued[0]["reason"] == "non_comment_event_context_sync_only"


@pytest.mark.asyncio
async def test_notion_deleted_comment_is_ignored_without_agent_dispatch(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

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
            json={"id": "notion-delivery-2", "type": "comment.deleted", "entity": {"id": "comment-1", "type": "comment"}},
            headers={"X-Request-ID": "delivery-2"},
        )
        body = await response.json()

    assert response.status == 200
    assert body["status"] == "ignored"
    assert captured_events == []
    assert not (tmp_path / "data" / "notion-sync" / "notion-sync-queue.jsonl").exists()



def test_notion_subscription_prompt_requires_block_parent_comment_fallback(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    subs_path = Path.home() / ".hermes" / "webhook_subscriptions.json"
    if not subs_path.exists():
        pytest.skip("local webhook subscription config not present")
    route = json.loads(subs_path.read_text())["notion-sync"]
    prompt = route.get("comment_prompt") or route.get("prompt", "")

    assert "payload.data.parent.id" in prompt
    assert "parent.type == block" in prompt
    assert "page_id-only fallback" in prompt
    assert route.get("skills") == ["productivity/notion"]


@pytest.mark.asyncio
async def test_notion_created_comment_still_dispatches_to_agent(tmp_path, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway import webhook_queue

    monkeypatch.setattr(webhook_queue, "get_hermes_home", lambda: tmp_path)
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
            json={"id": "notion-delivery-3", "type": "comment.created", "entity": {"id": "comment-1", "type": "comment"}},
            headers={"X-Request-ID": "delivery-3"},
        )
        body = await response.json()

    assert response.status == 202
    assert body["status"] == "accepted"
    assert len(captured_events) == 1
    assert "comment.created" in captured_events[0].text
    assert not (tmp_path / "data" / "notion-sync" / "notion-sync-queue.jsonl").exists()
