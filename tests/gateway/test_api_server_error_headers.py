"""Regression coverage for control characters in API error headers (#133848)."""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_openai_routes import _api_error_header_value


def test_api_error_header_value_replaces_forbidden_control_characters():
    out = _api_error_header_value("first\r\nsecond\tthird\x00\x7f", limit=200)

    assert out == "first second third"
    assert all(ord(char) >= 0x20 and ord(char) != 0x7F for char in out)


@pytest.mark.asyncio
async def test_partial_failure_sanitizes_error_response_header():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    error = "provider rejected request:\r\nModel blocked\x00by guardrail"
    result = {
        "final_response": "Partial answer",
        "completed": False,
        "partial": True,
        "failed": True,
        "error": error,
        "messages": [],
        "api_calls": 1,
    }

    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run_agent:
            run_agent.return_value = (
                result,
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status == 200
        header = response.headers["X-Hermes-Error"]
        assert header == "provider rejected request: Model blocked by guardrail"
        assert all(ord(char) >= 0x20 and ord(char) != 0x7F for char in header)
        data = await response.json()
        assert data["hermes"]["error"] == error
