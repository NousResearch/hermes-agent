import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.realtime_loop import LiveContextStore, RealtimeLoop
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _fake_llm_response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
            )
        ]
    )


@pytest.mark.asyncio
async def test_realtime_loop_updates_context_and_creates_task():
    loop = RealtimeLoop(LiveContextStore())

    payload = {
        "say": "Let me check that. I will keep you updated.",
        "action": "start_task",
        "action_request": "Check recent email from Sam.",
        "context_patch": {"current_user_need": "email lookup"},
    }
    with patch(
        "agent.auxiliary_client.async_call_llm",
        return_value=_fake_llm_response(json.dumps(payload)),
    ) as call:
        result = await loop.handle_turn(
            session_key="voice:test",
            user_text="Can you check my email from Sam?",
            transcript=[{"role": "user", "text": "Can you check my email from Sam?"}],
        )

    assert call.await_count == 1
    assert result["say"] == "Let me check that. I will keep you updated."
    assert result["action"] == "start_task"
    assert result["context"]["current_user_need"] == "email lookup"
    assert result["task"]["request"] == "Check recent email from Sam."

    tasks = loop.store.list_tasks("voice:test")
    assert len(tasks) == 1
    assert tasks[0]["status"] == "queued"


@pytest.mark.asyncio
async def test_realtime_loop_falls_back_when_talker_unavailable():
    loop = RealtimeLoop(LiveContextStore())

    with patch(
        "agent.auxiliary_client.async_call_llm",
        side_effect=RuntimeError("No LLM provider configured"),
    ):
        result = await loop.handle_turn(
            session_key="voice:test",
            user_text="Can you check my latest email?",
            timeout=0.5,
        )

    assert result["degraded"] is True
    assert result["action"] == "start_task"
    assert "checking" in result["say"].lower()
    assert result["task"]["request"] == "Can you check my latest email?"


def _create_realtime_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/realtime/turn", adapter._handle_realtime_turn)
    app.router.add_get("/v1/realtime/context", adapter._handle_realtime_context)
    app.router.add_post("/v1/realtime/context", adapter._handle_realtime_context)
    app.router.add_get("/v1/realtime/tasks", adapter._handle_realtime_tasks)
    app.router.add_post("/v1/realtime/tasks", adapter._handle_realtime_tasks)
    return app


@pytest.mark.asyncio
async def test_realtime_api_turn_and_context():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    app = _create_realtime_app(adapter)
    payload = {
        "say": "I am checking that now.",
        "action": "none",
        "action_request": "",
        "context_patch": {"known_facts": {"_append": ["caller asked about invoices"]}},
    }

    with patch(
        "agent.auxiliary_client.async_call_llm",
        return_value=_fake_llm_response(json.dumps(payload)),
    ):
        async with TestClient(TestServer(app)) as cli:
            headers = {
                "Authorization": "Bearer sk-test",
                "X-Hermes-Session-Key": "voice:test",
            }
            resp = await cli.post(
                "/v1/realtime/turn",
                headers=headers,
                json={"input": "Check invoices", "timeout": 1.0},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "hermes.realtime.turn"
            assert data["say"] == "I am checking that now."

            context_resp = await cli.get("/v1/realtime/context", headers=headers)
            assert context_resp.status == 200
            context_data = await context_resp.json()
            assert context_data["context"]["known_facts"] == ["caller asked about invoices"]


@pytest.mark.asyncio
async def test_realtime_api_task_creation():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    app = _create_realtime_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        headers = {
            "Authorization": "Bearer sk-test",
            "X-Hermes-Session-Key": "voice:test",
        }
        resp = await cli.post(
            "/v1/realtime/tasks",
            headers=headers,
            json={"request": "Research the latest invoice email."},
        )
        assert resp.status == 202
        data = await resp.json()
        assert data["object"] == "hermes.realtime.task"
        assert data["request"] == "Research the latest invoice email."

        list_resp = await cli.get("/v1/realtime/tasks", headers=headers)
        assert list_resp.status == 200
        listed = await list_resp.json()
        assert len(listed["data"]) == 1
        assert listed["data"][0]["task_id"] == data["task_id"]
