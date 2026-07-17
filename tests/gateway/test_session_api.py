"""Focused tests for API server session-control endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _session_chat_runtime_overrides
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


@pytest.mark.asyncio
async def test_capabilities_advertises_session_control_surface(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    features = data["features"]
    assert features["session_resources"] is True
    assert features["session_chat"] is True
    assert features["session_chat_streaming"] is True
    assert features["session_fork"] is True
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }


@pytest.mark.asyncio
async def test_run_agent_binds_api_session_context_for_tool_env(adapter, monkeypatch):
    """API-server request sessions should reach tools and terminal subprocess env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-session")
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.session_context import get_session_env
            from tools.environments.local import _make_run_env

            observed["task_id"] = task_id
            observed["context_session_id"] = get_session_env("HERMES_SESSION_ID")
            observed["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            observed["child_session_id"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
    )

    assert result["session_id"] == "request-session"
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
    }


@pytest.mark.asyncio
async def test_session_crud_and_message_history(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        create_resp = await cli.post("/api/sessions", json={"title": "Mobile chat", "model": "test-model"})
        assert create_resp.status == 201
        created = await create_resp.json()
        session_id = created["session"]["id"]
        assert created["object"] == "hermes.session"
        assert created["session"]["title"] == "Mobile chat"

        session_db.append_message(session_id, "user", "hello from phone")
        session_db.append_message(session_id, "assistant", "hello from hermes")

        list_resp = await cli.get("/api/sessions?limit=10&offset=0")
        assert list_resp.status == 200
        listed = await list_resp.json()
        assert listed["object"] == "list"
        assert [s["id"] for s in listed["data"]] == [session_id]
        assert listed["data"][0]["message_count"] == 2

        get_resp = await cli.get(f"/api/sessions/{session_id}")
        assert get_resp.status == 200
        got = await get_resp.json()
        assert got["session"]["id"] == session_id
        assert got["session"]["message_count"] == 2

        messages_resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()
        assert messages["object"] == "list"
        assert [m["role"] for m in messages["data"]] == ["user", "assistant"]
        assert messages["data"][0]["content"] == "hello from phone"

        patch_resp = await cli.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"})
        assert patch_resp.status == 200
        patched = await patch_resp.json()
        assert patched["session"]["title"] == "Renamed"

        delete_resp = await cli.delete(f"/api/sessions/{session_id}")
        assert delete_resp.status == 200
        deleted = await delete_resp.json()
        assert deleted == {"object": "hermes.session.deleted", "id": session_id, "deleted": True}
        assert session_db.get_session(session_id) is None


@pytest.mark.asyncio
async def test_session_patch_end_reason_returns_the_updated_session(adapter, session_db):
    session_id = session_db.create_session("end-session", "api_server")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.patch(
            f"/api/sessions/{session_id}",
            json={"end_reason": "client_close"},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["session"]["id"] == session_id
    assert payload["session"]["end_reason"] == "client_close"
    assert payload["session"]["ended_at"] is not None
    assert session_db.get_session(session_id)["end_reason"] == "client_close"


@pytest.mark.asyncio
async def test_session_patch_applies_title_and_end_reason_to_the_compression_tip(
    adapter,
    session_db,
):
    session_db.create_session("root-end", "api_server")
    session_db.end_session("root-end", "compression")
    session_db.create_session("tip-end", "api_server", parent_session_id="root-end")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.patch(
            "/api/sessions/root-end",
            json={"title": "Finished Project", "end_reason": "client_close"},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["session"]["id"] == "tip-end"
    assert payload["session"]["title"] == "Finished Project"
    assert payload["session"]["end_reason"] == "client_close"
    assert payload["session"]["ended_at"] is not None
    assert session_db.get_session("root-end")["end_reason"] == "compression"
    assert session_db.get_session("tip-end")["end_reason"] == "client_close"


@pytest.mark.asyncio
async def test_session_patch_does_not_end_an_ordinary_child_session(adapter, session_db):
    session_db.create_session("source-end", "api_server")
    session_db.end_session("source-end", "branched")
    session_db.create_session(
        "fork-end",
        "api_server",
        parent_session_id="source-end",
    )
    session_db.append_message("fork-end", "user", "fork content")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.patch(
            "/api/sessions/source-end",
            json={"title": "Old Source", "end_reason": "client_close"},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["session"]["id"] == "source-end"
    assert payload["session"]["title"] == "Old Source"
    assert payload["session"]["end_reason"] == "branched"
    assert session_db.get_session("source-end")["end_reason"] == "branched"
    assert session_db.get_session("fork-end")["end_reason"] is None


@pytest.mark.asyncio
async def test_session_auto_title_patch_only_names_an_untitled_session(adapter, session_db):
    session_id = session_db.create_session("title-once", "api_server")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        first = await cli.patch(
            f"/api/sessions/{session_id}",
            json={"title": "First Automatic Title", "if_untitled": True},
        )
        first_payload = await first.json()
        second = await cli.patch(
            f"/api/sessions/{session_id}",
            json={"title": "Latest User Message", "if_untitled": True},
        )
        second_payload = await second.json()

    assert first.status == 200
    assert first_payload["session"]["title"] == "First Automatic Title"
    assert first_payload["title_updated"] is True
    assert second.status == 200
    assert second_payload["title_updated"] is False
    assert second_payload["session"]["title"] == "First Automatic Title"
    assert session_db.get_session_title(session_id) == "First Automatic Title"


@pytest.mark.asyncio
async def test_session_patch_title_targets_the_visible_compression_tip(adapter, session_db):
    session_db.create_session("root", "api_server")
    session_db.end_session("root", "compression")
    session_db.create_session("tip", "api_server", parent_session_id="root")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.patch(
            "/api/sessions/root",
            json={"title": "Manual Project Name"},
        )
        assert response.status == 200
        payload = await response.json()

    assert payload["session"]["id"] == "tip"
    assert payload["session"]["title"] == "Manual Project Name"
    assert session_db.get_session_title("root") is None
    assert session_db.get_session_title("tip") == "Manual Project Name"


@pytest.mark.asyncio
async def test_session_messages_follow_compression_tip(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server")
    session_db.append_message(source_id, "user", "before compression")
    session_db.end_session(source_id, "compression")
    session_db.create_session("tip-session", "api_server", parent_session_id=source_id)
    session_db.replace_messages(source_id, [])
    session_db.append_message("tip-session", "user", "after compression")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        messages_resp = await cli.get(f"/api/sessions/{source_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()

    assert messages["object"] == "list"
    assert messages["session_id"] == "tip-session"
    assert [m["content"] for m in messages["data"]] == ["after compression"]


@pytest.mark.asyncio
async def test_session_fork_uses_current_sessiondb_branch_primitives(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server", model="test-model")
    session_db.set_session_title(source_id, "Original")
    session_db.append_message(source_id, "user", "first path")
    session_db.append_message(source_id, "assistant", "answer")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(f"/api/sessions/{source_id}/fork", json={"title": "Alternative"})
        assert resp.status == 201
        payload = await resp.json()

    fork = payload["session"]
    assert payload["object"] == "hermes.session"
    assert fork["id"] != source_id
    assert fork["parent_session_id"] == source_id
    assert fork["title"] == "Alternative"
    assert [m["content"] for m in session_db.get_messages(fork["id"])] == ["first path", "answer"]
    assert session_db.get_session(source_id)["end_reason"] == "branched"


@pytest.mark.asyncio
async def test_api_fork_cannot_be_selected_as_a_compression_continuation(adapter, session_db):
    session_db.create_session("root-fork", "api_server")
    session_db.append_message("root-fork", "user", "before compression")
    session_db.end_session("root-fork", "compression")
    session_db.create_session("tip-fork", "api_server", parent_session_id="root-fork")
    session_db.append_message("tip-fork", "user", "real continuation")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        fork_response = await cli.post(
            "/api/sessions/root-fork/fork",
            json={"id": "ordinary-fork", "title": "Alternative Path"},
        )
        assert fork_response.status == 201
        session_db.append_message("ordinary-fork", "user", "new fork turn")
        patch_response = await cli.patch(
            "/api/sessions/root-fork",
            json={"title": "Finished Project", "end_reason": "client_close"},
        )
        payload = await patch_response.json()

    assert patch_response.status == 200
    assert payload["session"]["id"] == "tip-fork"
    assert payload["session"]["title"] == "Finished Project"
    assert payload["session"]["end_reason"] == "client_close"
    assert session_db.get_session("ordinary-fork")["title"] == "Alternative Path"
    assert session_db.get_session("ordinary-fork")["end_reason"] is None


@pytest.mark.asyncio
async def test_session_chat_loads_history_and_preserves_session_headers(auth_adapter, session_db):
    session_id = session_db.create_session("chat-session", "api_server")
    session_db.set_session_title(session_id, "Chat")
    session_db.append_message(session_id, "user", "earlier")
    session_db.append_message(session_id, "assistant", "prior answer")

    mock_run = AsyncMock(return_value=({"final_response": "fresh answer", "session_id": session_id}, {"total_tokens": 3}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "next", "system_message": "stay focused"},
                headers={"Authorization": "Bearer sk-test", "X-Hermes-Session-Key": "client-42"},
            )
            assert resp.status == 200
            payload = await resp.json()

    assert resp.headers["X-Hermes-Session-Id"] == session_id
    assert resp.headers["X-Hermes-Session-Key"] == "client-42"
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["session_id"] == session_id
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "fresh answer"
    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == session_id
    assert kwargs["gateway_session_key"] == "client-42"
    assert kwargs["ephemeral_system_prompt"] == "stay focused"
    history = kwargs["conversation_history"]
    assert len(history) == 2
    assert isinstance(history[0].pop("timestamp"), (int, float))
    assert isinstance(history[1].pop("timestamp"), (int, float))
    assert history == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "prior answer"},
    ]


@pytest.mark.asyncio
async def test_session_chat_forwards_explicit_model_and_provider(auth_adapter, session_db):
    """The authenticated request body is authoritative for this agent turn."""
    session_id = session_db.create_session("routed-chat-session", "api_server")
    mock_run = AsyncMock(
        return_value=(
            {"final_response": "routed answer", "session_id": session_id},
            {"total_tokens": 3},
        )
    )
    app = _create_session_app(auth_adapter)

    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "use the selected model",
                    "model": "gpt-5.6-sol",
                    "provider": "openai-codex",
                    "reasoning_effort": "xhigh",
                    "service_tier": "priority",
                },
                headers={"Authorization": "Bearer sk-test"},
            )

    assert resp.status == 200
    kwargs = mock_run.call_args.kwargs
    assert kwargs["request_route"] == {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
    }
    assert kwargs["reasoning_effort_override"] == "xhigh"
    assert kwargs["service_tier_override"] == "priority"


@pytest.mark.asyncio
async def test_session_chat_resolves_configured_model_route_alias(auth_adapter, session_db):
    session_id = session_db.create_session("aliased-chat-session", "api_server")
    auth_adapter._model_routes = {
        "webui-gpt": {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://example.invalid/v1",
        }
    }
    mock_run = AsyncMock(
        return_value=(
            {"final_response": "routed answer", "session_id": session_id},
            {"total_tokens": 3},
        )
    )
    app = _create_session_app(auth_adapter)

    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "use the alias", "model": "webui-gpt"},
                headers={"Authorization": "Bearer sk-test"},
            )

    assert resp.status == 200
    assert mock_run.call_args.kwargs["request_route"] == {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "https://example.invalid/v1",
    }


@pytest.mark.asyncio
async def test_session_chat_accepts_multimodal_message(auth_adapter, session_db):
    session_id = session_db.create_session("image-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    mock_run = AsyncMock(return_value=({"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": image_payload},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert resp.status == 200, await resp.text()

    _, kwargs = mock_run.call_args
    assert kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_accepts_multimodal_message(adapter, session_db):
    session_id = session_db.create_session("image-stream-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["stream_delta_callback"]("A cat.")
        return {"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": image_payload},
            )
            assert resp.status == 200, await resp.text()
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: assistant.completed" in body
    assert captured_kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_forwards_explicit_model_and_provider(adapter, session_db):
    session_id = session_db.create_session("routed-stream-session", "api_server")
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {"final_response": "routed", "session_id": session_id}, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={
                    "message": "use the selected model",
                    "model": "gpt-5.6-sol",
                    "provider": "openai-codex",
                    "reasoning_effort": "low",
                    "service_tier": "normal",
                },
            )
            assert resp.status == 200
            await resp.text()

    assert captured["request_route"] == {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
    }
    assert captured["reasoning_effort_override"] == "low"
    assert captured["service_tier_override"] == "normal"


@pytest.mark.asyncio
async def test_session_chat_stream_emits_lifecycle_events_and_keepalive_safe_shape(adapter, session_db):
    session_id = session_db.create_session("stream-session", "api_server")
    session_db.set_session_title(session_id, "Stream")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"]("Hello")
        kwargs["stream_delta_callback"](" world")
        kwargs["tool_progress_callback"]("reasoning.available", tool_name="_thinking", preview="thinking")
        return {"final_response": "Hello world", "session_id": session_id}, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "stream please"})
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: run.started" in body
    assert "event: message.started" in body
    assert "event: assistant.delta" in body
    assert "Hello world" in body
    assert "event: tool.progress" in body
    assert "event: assistant.completed" in body
    assert "event: run.completed" in body
    assert "event: done" in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "field", "value", "error_code"),
    [
        ("chat", "model", {"bad": "shape"}, "invalid_model"),
        ("chat/stream", "provider", ["bad"], "invalid_provider"),
        ("chat", "reasoning_effort", "extreme", "invalid_reasoning_effort"),
        ("chat/stream", "reasoning_effort", {"bad": "shape"}, "invalid_reasoning_effort"),
        ("chat", "service_tier", "turbo", "invalid_service_tier"),
        ("chat/stream", "service_tier", True, "invalid_service_tier"),
    ],
)
async def test_session_chat_rejects_invalid_request_controls(
    adapter,
    session_db,
    endpoint,
    field,
    value,
    error_code,
):
    session_id = session_db.create_session(
        f"bad-request-control-{field}-{endpoint.replace('/', '-')}",
        "api_server",
    )
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/{endpoint}",
            json={"message": "hello", field: value},
        )
        assert resp.status == 400
        data = await resp.json()

    assert data["error"]["code"] == error_code


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_session_chat_accepts_all_supported_reasoning_efforts(effort):
    parsed_effort, service_tier, err = _session_chat_runtime_overrides(
        {"reasoning_effort": effort}
    )

    assert err is None
    assert parsed_effort == effort
    assert service_tier is None


@pytest.mark.asyncio
async def test_session_chat_stream_run_completed_carries_turn_transcript(adapter, session_db):
    """run.completed must include the full interleaved turn transcript so a
    client that lost intermediate (pre-tool-call) assistant text from the live
    delta stream can reconcile without a separate /messages fetch. Refs #34703.
    """
    import json as _json

    session_id = session_db.create_session("transcript-session", "api_server")

    async def fake_run(**kwargs):
        # Stream the intermediate planning text the way a real turn would.
        kwargs["stream_delta_callback"]("Let me search for that:")
        kwargs["stream_delta_callback"]("Here is the summary.")
        result = {
            "final_response": "Here is the summary.",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "search then summarize"},
                {
                    "role": "assistant",
                    "content": "Let me search for that:",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        }
        return result, {"total_tokens": 6}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "search then summarize"},
            )
            assert resp.status == 200
            body = await resp.text()

    # Pull the run.completed event payload out of the SSE body.
    run_completed_payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_completed_payload = _json.loads(line[len("data: "):])
            break
    assert run_completed_payload is not None, body
    messages = run_completed_payload.get("messages")
    assert isinstance(messages, list) and messages, run_completed_payload

    # The colon-ended intermediate text that preceded the tool call must be present.
    contents = [m.get("content") for m in messages]
    assert "Let me search for that:" in contents
    assert "Here is the summary." in contents
    # No prior-turn user message should leak into the per-turn slice.
    assert all(m.get("role") in ("assistant", "tool") for m in messages)
    # The tool call is preserved alongside the intermediate text.
    assert any(m.get("tool_calls") for m in messages)



@pytest.mark.asyncio
async def test_session_endpoints_require_auth_when_key_configured(auth_adapter):
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/sessions")
        assert resp.status == 401
        body = await resp.json()
        assert body["error"]["code"] == "invalid_api_key"

        ok = await cli.get("/api/sessions", headers={"Authorization": "Bearer sk-test"})
        assert ok.status == 200
        data = await ok.json()
        assert data["object"] == "list"
        assert data["data"] == []


@pytest.mark.asyncio
async def test_session_header_rejected_without_api_key(adapter, session_db):
    session_id = session_db.create_session("unsafe-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hello"},
            headers={"X-Hermes-Session-Key": "client-42"},
        )
        assert resp.status == 403
        data = await resp.json()
        assert "X-Hermes-Session-Key requires API key" in data["error"]["message"]
