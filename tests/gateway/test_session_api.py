"""Focused tests for API server session-control endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
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
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert "runtime" not in usage
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
        assert body["error"]["code"] == "gateway_auth_failed"

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


# ---------------------------------------------------------------------------
# Session-persisted model threading + provider-auth failure surfacing
# (salvaged from PR #57947 by @FvanW and PR #59941 by @kaishi00)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_chat_threads_session_model_to_run_agent(auth_adapter, session_db):
    """POST /api/sessions persists a per-session model, but the chat handler
    previously fetched the session record and threw it away — the session's
    chosen model silently had no effect on any chat turn."""
    session_id = session_db.create_session("model-pinned-session", "api_server", model="claude-sonnet-4-6")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": session_id}, {"total_tokens": 1}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hi"},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert resp.status == 200

    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_session_chat_stream_threads_session_model_to_run_agent(adapter, session_db):
    """Streaming twin of the session-model threading test above."""
    session_id = session_db.create_session("model-pinned-stream-session", "api_server", model="gpt-5.5")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": session_id}, {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "hi"},
            )
            assert resp.status == 200
            await resp.read()

    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_model"] == "gpt-5.5"


@pytest.mark.asyncio
async def test_session_chat_resolves_stored_model_route_alias(session_db, monkeypatch):
    """A session-persisted model that matches a model_routes alias must go
    through the route path (so route provider/credentials apply) and NOT be
    passed as a raw session_model (idea from PR #59941 by @kaishi00)."""
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"model_routes": {"alias": {"model": "route/model", "provider": "openrouter"}}},
        )
    )
    adapter._session_db = session_db
    session_id = session_db.create_session("route-pinned-session", "api_server", model="alias")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": session_id}, {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hi"},
            )
            assert resp.status == 200

    _, kwargs = mock_run.call_args
    assert kwargs["route"] == {"model": "route/model", "provider": "openrouter"}
    assert kwargs["session_model"] is None


@pytest.mark.asyncio
async def test_run_agent_returns_controlled_response_on_provider_auth_failure(adapter, monkeypatch):
    """_resolve_runtime_agent_kwargs() (inside _create_agent()) raises
    RuntimeError on provider auth/credential failure. Previously this
    propagated unhandled out of _run_agent(): /v1/chat/completions caught it
    as a generic 500, and /api/sessions/{id}/chat didn't catch it at all
    (raw aiohttp 500, no JSON body). Must now return run.py's controlled
    response shape instead of raising. Exercises the REAL boundary
    (gateway.run._resolve_runtime_agent_kwargs, the sole raiser)."""
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: (_ for _ in ()).throw(
            RuntimeError("No credentials found for provider 'nous' — run `hermes auth add nous`")
        ),
    )

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
    )

    assert result == {
        "final_response": "⚠️ Provider authentication failed: No credentials found for provider 'nous' — run `hermes auth add nous`",
        "messages": [],
        "api_calls": 0,
        "tools": [],
    }
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_run_agent_does_not_swallow_unrelated_exceptions(adapter, monkeypatch):
    """The _ProviderAuthResolutionError catch must stay narrow — a TypeError
    elsewhere in _create_agent()/run_conversation() must still propagate."""
    def fake_create_agent(**kwargs):
        raise TypeError("unrelated bug: unexpected keyword argument")

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    with pytest.raises(TypeError, match="unrelated bug"):
        await adapter._run_agent(
            user_message="hello",
            conversation_history=[],
            session_id="request-session",
        )


@pytest.mark.asyncio
async def test_run_agent_does_not_swallow_unrelated_runtime_error_from_run_conversation(adapter, monkeypatch):
    """agent.run_conversation() can legitimately raise a RuntimeError
    unrelated to provider auth (e.g. run_agent.py's "Failed to recreate
    closed OpenAI client"). A bare `except RuntimeError` around the whole
    _create_agent()+run_conversation() span would mislabel it as
    "Provider authentication failed". Only _ProviderAuthResolutionError —
    raised exclusively inside _create_agent() at the
    _resolve_runtime_agent_kwargs() call site — may trigger the controlled
    response; this unrelated RuntimeError must propagate unhandled."""
    class _FakeAgent:
        def run_conversation(self, **kwargs):
            raise RuntimeError("Failed to recreate closed OpenAI client")

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: _FakeAgent())

    with pytest.raises(RuntimeError, match="Failed to recreate closed OpenAI client"):
        await adapter._run_agent(
            user_message="hello",
            conversation_history=[],
            session_id="request-session",
        )


@pytest.mark.asyncio
async def test_session_chat_surfaces_controlled_response_on_provider_auth_failure(auth_adapter, session_db, monkeypatch):
    """End-to-end: POST /api/sessions/{id}/chat previously had zero wrapping
    around _run_agent() — an unhandled RuntimeError produced a raw aiohttp
    500 with no JSON body. Must now return 200 with the controlled error
    message as the assistant content. Exercises the real
    gateway.run._resolve_runtime_agent_kwargs() boundary, not a mocked
    _create_agent()."""
    session_id = session_db.create_session("auth-fail-session", "api_server")

    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: (_ for _ in ()).throw(RuntimeError("Auth failed: token expired")),
    )

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer sk-test"},
        )
        assert resp.status == 200
        payload = await resp.json()

    assert payload["message"]["content"] == "⚠️ Provider authentication failed: Auth failed: token expired"
def _register_session_model_route(app, adapter):
    app.router.add_post("/api/sessions/{session_id}/model", adapter._handle_session_model_lock)


def _patch_api_server_runtime(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "sk-global",
            "base_url": "https://openrouter.example/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda model="": {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "api_key": f"sk-{provider}",
            "base_url": f"https://{provider}.example/v1",
            "api_mode": "chat_completions",
        },
    )


@pytest.mark.asyncio
async def test_session_chat_builds_raw_provider_model_route_when_alias_missing(adapter, session_db):
    session_id = session_db.create_session("route-session", "api_server")
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": "ok",
                "session_id": session_id,
                "runtime": {"provider": "nous", "model": "x-ai/grok-4.5", "route_source": "raw_request"},
            },
            {"total_tokens": 2, "runtime": {"provider": "nous", "model": "x-ai/grok-4.5"}},
        )
    )
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "hello",
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "require_model_lock": True,
                },
            )
            assert resp.status == 200, await resp.text()
            payload = await resp.json()

    kwargs = mock_run.call_args.kwargs
    assert kwargs["route"] == {"provider": "nous", "model": "x-ai/grok-4.5"}
    assert payload["runtime"]["provider"] == "nous"
    assert payload["runtime"]["model"] == "x-ai/grok-4.5"
    assert payload["runtime"]["requested"]["model"] == "x-ai/grok-4.5"


@pytest.mark.asyncio
async def test_session_chat_passes_runtime_options_to_run_agent(adapter, session_db):
    session_id = session_db.create_session("options-session", "api_server")
    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": session_id}, {}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "hello",
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "model_options": {
                        "reasoning": {"enabled": True, "effort": "xhigh"},
                        "service_tier": "priority",
                        "fast": True,
                    },
                },
            )
            assert resp.status == 200, await resp.text()

    kwargs = mock_run.call_args.kwargs
    # In the merged design model_options travel raw to _create_agent, which
    # parses reasoning/service-tier itself (see _request_reasoning_config /
    # _request_service_tier) — there is no separate runtime_options kwarg.
    assert kwargs["model_options"] == {
        "reasoning": {"enabled": True, "effort": "xhigh"},
        "service_tier": "priority",
        "fast": True,
    }


@pytest.mark.asyncio
async def test_session_chat_stream_uses_same_runtime_lock(adapter, session_db):
    session_id = session_db.create_session("stream-lock-session", "api_server")
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        kwargs["stream_delta_callback"]("hi")
        return (
            {
                "final_response": "hi",
                "session_id": session_id,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                    "route_source": "raw_request",
                },
            },
            {
                "total_tokens": 1,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                },
            },
        )

    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={
                    "message": "stream",
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "model_options": {"reasoning": {"enabled": False}},
                    "require_model_lock": True,
                },
            )
            assert resp.status == 200, await resp.text()
            body = await resp.text()

    assert captured["route"] == {"provider": "nous", "model": "x-ai/grok-4.5"}
    assert captured["model_options"] == {"reasoning": {"enabled": False}}
    assert captured["confirmed_runtime_lock"] is True
    assert "x-ai/grok-4.5" in body
    assert "run.started" in body or "event: run.started" in body


@pytest.mark.asyncio
async def test_create_session_respects_browser_source_and_model_lock(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions",
            json={
                "id": "browser-lock-session",
                "source": "hermes_browser",
                "provider": "nous",
                "model": "x-ai/grok-4.5",
                "require_model_lock": True,
                "title": "Browser lock",
                "system_prompt": "browser prompt",
            },
        )
        assert resp.status == 201, await resp.text()
        payload = await resp.json()

    assert payload["session"]["source"] == "hermes_browser"
    assert payload["session"]["model"] == "x-ai/grok-4.5"
    row = session_db.get_session("browser-lock-session")
    assert row["source"] == "hermes_browser"
    assert row["model"] == "x-ai/grok-4.5"
    import json as _json
    model_config = row.get("model_config")
    if isinstance(model_config, str):
        model_config = _json.loads(model_config)
    assert model_config["browser_model_lock"]["provider"] == "nous"
    assert model_config["browser_model_lock"]["model"] == "x-ai/grok-4.5"
    assert model_config["browser_model_lock"]["confirmed"] is True


@pytest.mark.asyncio
async def test_session_model_lock_endpoint_persists_and_invalidates_prompt(adapter, session_db):
    session_id = session_db.create_session(
        "lock-endpoint-session",
        "api_server",
        model="gpt-5.5",
        model_config={"_branched_from": "parent-session"},
        system_prompt="Conversation started:\nModel: gpt-5.5\nProvider: openai-codex\n",
    )
    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, "_resolve_route", return_value=None):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/model",
                json={
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "model_options": {"reasoning": {"enabled": True, "effort": "high"}},
                    "require_model_lock": True,
                },
            )
            assert resp.status == 200, await resp.text()
            payload = await resp.json()

    assert payload["object"] == "hermes.session.model_lock"
    assert payload["runtime"]["requested"]["provider"] == "nous"
    assert payload["runtime"]["model"] == "x-ai/grok-4.5"
    assert payload["runtime"]["model_lock"] in {"accepted", "confirmed"}
    row = session_db.get_session(session_id)
    assert row["model"] == "x-ai/grok-4.5"
    assert row["system_prompt"] is None
    import json as _json
    model_config = row.get("model_config")
    if isinstance(model_config, str):
        model_config = _json.loads(model_config)
    assert model_config["_branched_from"] == "parent-session"
    assert model_config["browser_model_lock"]["provider"] == "nous"


@pytest.mark.asyncio
async def test_session_model_lock_endpoint_then_chat_reuses_persisted_lock_and_provider_credentials(
    adapter,
    session_db,
    monkeypatch,
):
    session_id = session_db.create_session(
        "endpoint-lock-chat",
        "api_server",
        model="gpt-5.5",
        system_prompt="Conversation started:\nModel: gpt-5.5\nProvider: openai-codex\n",
    )
    captured = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = kwargs["session_id"]
            self.provider = kwargs.get("provider") or ""
            self.model = kwargs.get("model") or ""

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "locked", "session_id": self.session_id}

    _patch_api_server_runtime(monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        adapter,
        "_session_model_override_for",
        lambda *_: {
            "model": "session/override-model",
            "provider": "openai-codex",
            "api_key": "sk-session-override",
            "base_url": "https://override.example/v1",
            "api_mode": "codex_responses",
        },
    )

    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, "_resolve_route", return_value=None):
        async with TestClient(TestServer(app)) as cli:
            lock_resp = await cli.post(
                f"/api/sessions/{session_id}/model",
                json={
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "require_model_lock": True,
                },
            )
            assert lock_resp.status == 200, await lock_resp.text()

            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "use the stored lock"},
            )
            assert resp.status == 200, await resp.text()
            payload = await resp.json()

    assert captured["provider"] == "nous"
    assert captured["model"] == "x-ai/grok-4.5"
    assert captured["api_key"] == "sk-nous"
    assert captured["base_url"] == "https://nous.example/v1"
    assert payload["runtime"]["provider"] == "nous"
    assert payload["runtime"]["model"] == "x-ai/grok-4.5"
    assert payload["runtime"]["requested"] == {
        "provider": "nous",
        "model": "x-ai/grok-4.5",
    }
    assert payload["runtime"]["route_source"] == "session_model_lock"


@pytest.mark.asyncio
async def test_session_model_lock_endpoint_then_chat_stream_reuses_persisted_lock(
    adapter,
    session_db,
):
    session_id = session_db.create_session("endpoint-lock-stream", "api_server")
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        kwargs["stream_delta_callback"]("hi")
        return (
            {
                "final_response": "hi",
                "session_id": session_id,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                    "route_source": "session_model_lock",
                },
            },
            {
                "total_tokens": 1,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                    "route_source": "session_model_lock",
                },
            },
        )

    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(
        adapter,
        "_run_agent",
        side_effect=fake_run,
    ):
        async with TestClient(TestServer(app)) as cli:
            lock_resp = await cli.post(
                f"/api/sessions/{session_id}/model",
                json={
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "require_model_lock": True,
                },
            )
            assert lock_resp.status == 200, await lock_resp.text()

            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "stream with stored lock"},
            )
            assert resp.status == 200, await resp.text()
            body = await resp.text()

    assert captured["route"] == {"provider": "nous", "model": "x-ai/grok-4.5"}
    assert captured["requested_runtime"]["provider"] == "nous"
    assert captured["requested_runtime"]["model"] == "x-ai/grok-4.5"
    assert captured["route_source"] == "session_model_lock"
    assert "x-ai/grok-4.5" in body


@pytest.mark.asyncio
async def test_run_agent_reports_actual_agent_runtime_not_requested_metadata(adapter, monkeypatch):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self):
            self.session_id = "runtime-session"
            self.provider = "actual-provider"
            self.model = "actual-model"
            self._hermes_api_runtime = {
                "provider": "requested-provider",
                "model": "requested-model",
                "route_source": "raw_request",
            }

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "ok", "session_id": self.session_id}

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: FakeAgent())

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="runtime-session",
        route={"provider": "requested-provider", "model": "requested-model"},
        requested_runtime={
            "provider": "requested-provider",
            "model": "requested-model",
        },
        route_source="session_model_lock",
    )

    assert result["runtime"]["provider"] == "actual-provider"
    assert result["runtime"]["model"] == "actual-model"
    assert result["runtime"]["requested"] == {
        "provider": "requested-provider",
        "model": "requested-model",
    }
    assert usage["runtime"]["provider"] == "actual-provider"
    assert usage["runtime"]["model"] == "actual-model"


@pytest.mark.asyncio
async def test_confirmed_runtime_lock_rejects_actual_runtime_mismatch(adapter, monkeypatch):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "mismatch-session"
        provider = "fallback-provider"
        model = "fallback-model"

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "wrong runtime", "session_id": self.session_id}

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: FakeAgent())

    with pytest.raises(RuntimeError, match="confirmed model lock runtime mismatch"):
        await adapter._run_agent(
            user_message="hello",
            conversation_history=[],
            session_id="mismatch-session",
            route={"provider": "nous", "model": "x-ai/grok-4.5"},
            requested_runtime={"provider": "nous", "model": "x-ai/grok-4.5"},
            route_source="session_model_lock",
            confirmed_runtime_lock=True,
        )


def test_confirmed_runtime_lock_fails_closed_on_provider_resolution_error(adapter, monkeypatch):
    _patch_api_server_runtime(monkeypatch)
    # Break BOTH resolution paths (primary picker-based resolver + the
    # gateway fallback) — a confirmed lock must propagate the failure
    # instead of constructing an agent on the previous global credentials.
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda provider: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
    )
    agent_ctor = patch("run_agent.AIAgent")
    with agent_ctor as mocked_agent:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            adapter._create_agent(
                session_id="locked-session",
                route={"provider": "nous", "model": "x-ai/grok-4.5"},
                confirmed_runtime_lock=True,
            )
    mocked_agent.assert_not_called()


def test_confirmed_runtime_lock_disables_global_fallback_model(adapter, monkeypatch):
    _patch_api_server_runtime(monkeypatch)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: "openrouter/fallback-model"),
    )
    captured = {}

    class FakeAgent:
        provider = "nous"
        model = "x-ai/grok-4.5"

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    adapter._create_agent(
        session_id="locked-session",
        route={"provider": "nous", "model": "x-ai/grok-4.5"},
        confirmed_runtime_lock=True,
    )

    assert captured["fallback_model"] is None


@pytest.mark.asyncio
async def test_unconfirmed_request_does_not_replace_confirmed_session_lock(adapter, session_db):
    session_id = session_db.create_session("one-off-override", "api_server")
    session_db.update_session_runtime_lock(
        session_id,
        provider="nous",
        model="x-ai/grok-4.5",
        route_source="raw_request",
        confirmed=True,
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": "ok",
                "session_id": session_id,
                "runtime": {"provider": "openrouter", "model": "anthropic/claude-sonnet"},
            },
            {"total_tokens": 1},
        )
    )
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(
        adapter,
        "_run_agent",
        mock_run,
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "one turn only",
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet",
                },
            )
            assert resp.status == 200, await resp.text()

    import json as _json

    row = session_db.get_session(session_id)
    config = row["model_config"]
    if isinstance(config, str):
        config = _json.loads(config)
    assert config["browser_model_lock"]["provider"] == "nous"
    assert config["browser_model_lock"]["model"] == "x-ai/grok-4.5"
    assert config["browser_model_lock"]["confirmed"] is True


@pytest.mark.asyncio
async def test_require_model_lock_hard_fails_when_global_default_would_be_used(adapter, session_db, monkeypatch):
    session_id = session_db.create_session("lock-fail-session", "api_server")
    monkeypatch.setattr(adapter, "_model_name", "gpt-5.5")
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
        async with TestClient(TestServer(app)) as cli:
            # empty model + require_model_lock must not silently fall through
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "hello",
                    "provider": "nous",
                    "model": "",
                    "require_model_lock": True,
                },
            )
            assert resp.status in (400, 409), await resp.text()
            body = await resp.json()
            assert body["error"]["code"] in {"model_lock_unavailable", "invalid_model_lock", "missing_model"}
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_capabilities_advertises_session_model_lock(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()
    assert data["features"]["session_model_lock"] is True
    assert data["endpoints"]["session_model_lock"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/model",
    }
