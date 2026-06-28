"""Request-scoped model/provider overrides for API-server agent runs."""

import asyncio
from unittest.mock import MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _chat_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _runs_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def test_chat_completions_passes_request_model_provider_to_agent(monkeypatch):
    async def _case():
        adapter = _make_adapter()
        captured = {}

        async def fake_run_agent(**kwargs):
            captured.update(kwargs)
            return {"final_response": "ok"}, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        monkeypatch.setattr(adapter, "_run_agent", fake_run_agent)

        async with TestClient(TestServer(_chat_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "pichot-moa-frontier",
                    "provider": "moa",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            body = await resp.json()

        assert resp.status == 200
        assert body["choices"][0]["message"]["content"] == "ok"
        assert captured["model_override"] == "pichot-moa-frontier"
        assert captured["provider_override"] == "moa"

    asyncio.run(_case())


def test_runs_api_passes_request_model_provider_to_created_agent(monkeypatch):
    async def _case():
        adapter = _make_adapter()
        captured = {}
        agent_created = asyncio.Event()

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            agent_created.set()
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent.session_prompt_tokens = 1
            mock_agent.session_completion_tokens = 1
            mock_agent.session_total_tokens = 2
            return mock_agent

        monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

        async with TestClient(TestServer(_runs_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"model": "pichot-moa-frontier", "provider": "moa", "input": "hello"},
            )
            body = await resp.json()
            await asyncio.wait_for(agent_created.wait(), timeout=2)

        assert resp.status == 202
        assert body["status"] == "started"
        assert captured["model_override"] == "pichot-moa-frontier"
        assert captured["provider_override"] == "moa"

    asyncio.run(_case())
