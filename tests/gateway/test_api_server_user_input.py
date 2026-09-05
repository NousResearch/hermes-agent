"""API-server transport coverage for native asynchronous user input."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _STATIC_FEATURE_FLAGS,
)


class _SessionDB:
    def get_session(self, session_id):
        return {"id": session_id, "source": "api_server"}

    def get_pending_user_input(self, request_id, *, session_id, now=None):
        return {"request_id": request_id, "session_id": session_id, "turn_id": "turn-1"}


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get(
        "/api/sessions/{session_id}/user-input/pending",
        adapter._handle_session_user_input_pending,
    )
    app.router.add_post(
        "/api/sessions/{session_id}/user-input/{request_id}/answer",
        adapter._handle_session_user_input_answer,
    )
    return app


@pytest.mark.asyncio
async def test_run_agent_forwards_user_input_callback_and_emits_request():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = _SessionDB()
    callback_payload = {
        "request_id": "request-1",
        "session_id": "session-1",
        "questions": [{"id": "choice", "text": "Pick one", "options": ["a"]}],
    }
    captured = {}
    user_input_callback = MagicMock()

    agent = MagicMock()
    agent.session_id = "session-1"
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def run_conversation(**_kwargs):
        captured["callback"](callback_payload)
        return {"final_response": "continued"}

    agent.run_conversation.side_effect = run_conversation

    def create_agent(**kwargs):
        captured.update(kwargs)
        captured["callback"] = kwargs["user_input_callback"]
        return agent

    adapter._create_agent = create_agent
    result, _usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="session-1",
        user_input_callback=user_input_callback,
    )

    assert result["final_response"] == "continued"
    assert captured["user_input_callback"] is user_input_callback
    user_input_callback.assert_called_once_with(callback_payload)


@pytest.mark.asyncio
async def test_pending_endpoint_replays_session_scoped_requests(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = _SessionDB()
    pending = [{
        "request_id": "request-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "questions": [],
        "context": "",
        "status": "pending",
        "answer": None,
        "expires_at": 0,
    }]
    monkeypatch.setattr("tools.user_input_tool.list_pending_user_inputs", lambda session_id, **_: pending)

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.get("/api/sessions/session-1/user-input/pending")
        payload = await response.json()

    assert response.status == 200
    assert payload == {"object": "list", "session_id": "session-1", "data": pending}


@pytest.mark.asyncio
async def test_answer_endpoint_delivers_structured_answers_to_matching_live_turn(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = _SessionDB()
    live_agent = SimpleNamespace(session_id="session-1", _current_turn_id="turn-1")
    adapter._active_run_agents = {"run-1": live_agent}
    captured = {}

    def answer(request_id, answers, **kwargs):
        captured.update(request_id=request_id, answers=answers, **kwargs)
        return {
            "request_id": request_id,
            "status": "answered",
            "accepted": True,
            "answer": answers,
            "turn_id": kwargs.get("turn_id") or "turn-1",
            "delivery": "queued",
        }

    monkeypatch.setattr("tools.user_input_tool.answer_user_input", answer)

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/user-input/request-1/answer",
            json={"answers": {"choice": "a"}},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["accepted"] is True
    assert captured["request_id"] == "request-1"
    assert captured["answers"] == {"choice": "a"}
    assert captured["session_id"] == "session-1"
    assert captured["agent"] is live_agent
    assert captured["turn_id"] == "turn-1"


@pytest.mark.asyncio
async def test_answer_endpoint_rejects_non_object_answers(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = _SessionDB()
    called = False

    def answer(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr("tools.user_input_tool.answer_user_input", answer)

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/session-1/user-input/request-1/answer",
            json={"answers": ["a"]},
        )
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == "invalid_user_input_answer"
    assert called is False


def test_capabilities_and_route_table_advertise_user_input():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}
    assert ("GET", "/api/sessions/{session_id}/user-input/pending") in routes
    assert ("POST", "/api/sessions/{session_id}/user-input/{request_id}/answer") in routes
    assert _STATIC_FEATURE_FLAGS["session_user_input"] is True
