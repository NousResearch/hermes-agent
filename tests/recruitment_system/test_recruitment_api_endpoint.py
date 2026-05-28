from __future__ import annotations

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import APIServerAdapter
from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_recruitment_ai_query_endpoint_returns_service_payload():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app.router.add_post("/api/recruitment-system/ai-query", adapter._handle_recruitment_ai_query)

    response = {
        "success": True,
        "intent": "recruiting_job_list",
        "answer": "当前正在招聘的岗位有 AI算法工程师。",
        "data": [{"job_name": "AI算法工程师"}],
        "safe": True,
        "trace_id": "trace-1",
    }

    with patch("tools.recruitment_system_tool.query_recruitment_system_api", return_value=response):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/recruitment-system/ai-query",
                json={"question": "当前正在招聘的岗位有哪些？", "user_id": "u001"},
            )
            body = await resp.json()

    assert resp.status == 200
    assert body["success"] is True
    assert body["intent"] == "recruiting_job_list"
    assert body["data"] == [{"job_name": "AI算法工程师"}]


@pytest.mark.asyncio
async def test_recruitment_ai_query_endpoint_requires_auth_when_configured():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    app = web.Application()
    app.router.add_post("/api/recruitment-system/ai-query", adapter._handle_recruitment_ai_query)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/recruitment-system/ai-query",
            json={"question": "当前正在招聘的岗位有哪些？", "user_id": "u001"},
        )

    assert resp.status == 401
