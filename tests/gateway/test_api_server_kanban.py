"""Tests for /v1/kanban/* gateway routes and kanban_http handlers."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.kanban_api_routes import register_kanban_routes
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_http as kh


def _adapter(key: str = "test-key") -> APIServerAdapter:
    cfg = PlatformConfig(enabled=True, extra={"key": key})
    return APIServerAdapter(cfg)


@pytest.mark.asyncio
async def test_kanban_board_requires_auth():
    adapter = _adapter()
    app = web.Application()
    register_kanban_routes(app, adapter)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get("/v1/kanban/board")
        assert resp.status == 401
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kanban_board_ok(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    adapter = _adapter()
    app = web.Application()
    register_kanban_routes(app, adapter)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        headers = {"Authorization": "Bearer test-key"}
        resp = await client.get("/v1/kanban/board", headers=headers)
        assert resp.status == 200
        data = await resp.json()
        assert "columns" in data
        assert "latest_event_id" in data
    finally:
        await client.close()


def test_kanban_http_create_and_update(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    created = kh.http_create_task(
        kh.CreateTaskRequest(title="Gateway unit test"),
        None,
    )
    task_id = created["task"]["id"]
    updated = kh.http_update_task(task_id, kh.UpdateTaskRequest(status="triage"), None)
    assert updated["task"]["status"] == "triage"
