"""OpenAI-standard top-level ``reasoning_effort`` fold (#138299).

Strict OpenAI-compatible clients (Vercel AI SDK ``openai-compatible`` — e.g.
Homarr's assistant) send the standard top-level ``reasoning_effort`` instead of
the Hermes ``model_options`` extension; ``_request_agent_overrides`` folds it in
as the weakest source, so an explicit extension spelling still wins.
"""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _request_agent_overrides,
    cors_middleware,
    security_headers_middleware,
)


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


class TestTopLevelReasoningEffortFold:
    def test_top_level_folds_into_model_options(self):
        overrides = _request_agent_overrides({"reasoning_effort": "None"})
        assert overrides["model_options"] == {"reasoning_effort": "none"}

    def test_top_level_creates_model_options_when_absent(self):
        overrides = _request_agent_overrides({"reasoning_effort": "low"})
        assert overrides["model_options"] == {"reasoning_effort": "low"}

    def test_explicit_model_options_spelling_wins(self):
        overrides = _request_agent_overrides({
            "model_options": {"reasoning_effort": "high"},
            "reasoning_effort": "none",
        })
        assert overrides["model_options"] == {"reasoning_effort": "high"}

    def test_structured_reasoning_block_wins(self):
        overrides = _request_agent_overrides({
            "model_options": {"reasoning": {"enabled": False}},
            "reasoning_effort": "high",
        })
        assert overrides["model_options"] == {"reasoning": {"enabled": False}}

    @pytest.mark.parametrize("value", [None, "", "   ", 3, {"effort": "low"}])
    def test_non_string_or_blank_values_are_ignored(self, value):
        overrides = _request_agent_overrides({"reasoning_effort": value})
        assert "model_options" not in overrides


class TestChatCompletionsTopLevelEffort:
    @pytest.mark.asyncio
    async def test_chat_completions_honors_top_level_effort(self, adapter):
        mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
        app = web.Application(middlewares=mws)
        app["api_server_adapter"] = adapter
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "reasoning_effort": "none",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
        assert mock_run.call_args.kwargs.get("model_options") == {"reasoning_effort": "none"}
