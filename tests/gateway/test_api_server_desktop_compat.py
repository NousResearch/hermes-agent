"""Desktop compatibility endpoints on the API-server platform.

These routes are consumed by agente-desktop's bundled Hermes sidecar.  They
are intentionally covered here because a missing route degrades Desktop chat
tool authoring into HTTP 404s before Hermes' native tools ever run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter(api_key: str | None = None) -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/tools", adapter._handle_list_tools)
    app.router.add_post("/api/tools/{tool_name}", adapter._handle_dispatch_tool)
    app.router.add_post("/rpc", adapter._handle_rpc)
    return app


def _workflow_registry():
    import tools.workflow_routine_tools  # noqa: F401
    from tools.registry import registry

    return registry


@pytest.mark.asyncio
async def test_api_tools_lists_desktop_workflow_tools(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_tool_registry_loaded", _workflow_registry)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get("/api/tools")

    assert response.status == 200
    body = await response.json()
    names = {tool["name"] for tool in body["tools"]}
    assert "save_workflow" in names
    assert "create_routine" in names


@pytest.mark.asyncio
async def test_save_workflow_accepts_desktop_proxy_shape(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_tool_registry_loaded", _workflow_registry)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.post(
            "/api/tools/save_workflow",
            json={
                "arguments": {
                    "name": "סריקת וואטסאפ",
                    "description": "סריקת הודעות וואטסאפ ופתיחת כרטיסים",
                    "steps": [
                        {
                            "action": "create_ticket",
                            "title_template": "בקשת לקוח מוואטסאפ",
                        }
                    ],
                }
            },
        )

    assert response.status == 200
    body = await response.json()
    assert body["success"] is True
    workflow = body["workflow"]
    assert workflow["name_he"] == "סריקת וואטסאפ"
    assert workflow["description_he"] == "סריקת הודעות וואטסאפ ופתיחת כרטיסים"
    assert workflow["trigger"]["kind"] == "manual"
    assert workflow["actions"][0]["kind"] == "create_ticket"
    assert (Path(os.environ["HERMES_HOME"]) / "workflows" / f"{workflow['id']}.yaml").is_file()


@pytest.mark.asyncio
async def test_create_routine_accepts_desktop_proxy_shape(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_tool_registry_loaded", _workflow_registry)

    import tools.workflow_routine_tools as workflow_routine_tools

    monkeypatch.setattr(
        workflow_routine_tools,
        "_create_cron_job_for_routine",
        lambda *_args, **_kwargs: "cron-test-123",
    )

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.post(
            "/api/tools/create_routine",
            json={
                "arguments": {
                    "workflow_id": "wf-whatsapp",
                    "name": "בדיקה שעתית",
                    "cron": "0 * * * *",
                    "timezone": "Asia/Jerusalem",
                }
            },
        )

    assert response.status == 200
    body = await response.json()
    assert body["success"] is True
    routine = body["routine"]
    assert routine["workflow_id"] == "wf-whatsapp"
    assert routine["name_he"] == "בדיקה שעתית"
    assert routine["cron_schedule"] == "0 * * * *"
    assert routine["cron_job_id"] == "cron-test-123"
    assert (Path(os.environ["HERMES_HOME"]) / "routines" / f"{routine['id']}.yaml").is_file()


@pytest.mark.asyncio
async def test_rpc_unknown_method_returns_json_rpc_error_not_http_404():
    adapter = _make_adapter()

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.post(
            "/rpc",
            json={"jsonrpc": "2.0", "id": "r1", "method": "missing.method", "params": {}},
        )

    assert response.status == 200
    body = await response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "r1"
    assert body["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_rpc_accepts_desktop_request_id_alias_for_oauth(monkeypatch):
    adapter = _make_adapter()
    captured: dict[str, object] = {}

    class FakeTuiServer:
        @staticmethod
        def handle_request(body):
            captured.update(json.loads(json.dumps(body)))
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"status": "ok"}}

    monkeypatch.setattr(adapter, "_load_tui_rpc_server", lambda: FakeTuiServer)

    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.post(
            "/rpc",
            json={
                "jsonrpc": "2.0",
                "id": "oauth-1",
                "method": "auth.submit_oauth_code",
                "params": {"request_id": "req-123", "code": "abc"},
            },
        )

    assert response.status == 200
    body = await response.json()
    assert body["result"]["status"] == "ok"
    assert captured["method"] == "auth.submit_oauth_code"
    assert captured["params"]["session_id"] == "req-123"
