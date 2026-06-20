"""Focused tests for API server session-control endpoints."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, ResponseStore
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
def adapter(session_db, tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._response_store.close()
    adapter._response_store = ResponseStore(db_path=str(tmp_path / "response_store.db"))
    adapter._session_db = session_db
    try:
        yield adapter
    finally:
        adapter._response_store.close()


@pytest.fixture
def auth_adapter(session_db, tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._response_store.close()
    adapter._response_store = ResponseStore(db_path=str(tmp_path / "response_store.db"))
    adapter._session_db = session_db
    try:
        yield adapter
    finally:
        adapter._response_store.close()


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/sessions/active", adapter._handle_list_active_sessions)
    app.router.add_get("/v1/sessions/{session_id}/state", adapter._handle_session_state)
    app.router.add_get("/v1/sessions/{session_id}/items", adapter._handle_session_items)
    app.router.add_get("/v1/sessions/{session_id}/events", adapter._handle_session_events)
    app.router.add_get("/v1/session_events/firehose", adapter._handle_session_firehose)
    app.router.add_post("/v1/session_events/ack", adapter._handle_session_events_ack)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_get("/api/sessions/{session_id}/active-run", adapter._handle_session_active_run)
    app.router.add_get("/api/session-runs/{run_id}/events", adapter._handle_session_run_events)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/sessions/{session_id}/steer", adapter._handle_steer_session)
    app.router.add_post("/v1/sessions/{session_id}/halt", adapter._handle_halt_session)
    app.router.add_post("/v1/sessions/{session_id}/queue", adapter._handle_queue_session)
    return app


class _CaptureAgent:
    def __init__(self):
        self.call = None

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        self.call = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
        }
        return {"final_response": "ok", "messages": []}


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
    assert features["session_run_replay"] is True
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
    assert data["endpoints"]["session_active_run"] == {
        "method": "GET",
        "path": "/api/sessions/{session_id}/active-run",
    }
    assert data["endpoints"]["session_run_events"] == {
        "method": "GET",
        "path": "/api/session-runs/{run_id}/events",
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
async def test_query_access_token_auth_supports_websocket_clients(auth_adapter):
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities?access_token=sk-test")
        assert resp.status == 200


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
        assert messages["data"][0]["items"][0]["id"].startswith("msg:")

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
async def test_runs_hydrate_existing_session_history(adapter, session_db):
    session_id = session_db.create_session("continuity-session", "api_server")
    session_db.append_message(session_id, "user", "terminal twice")
    session_db.append_message(session_id, "assistant", "Ran terminal twice.")
    agent = _CaptureAgent()

    app = _create_session_app(adapter)
    with (
        patch.object(adapter, "_create_agent", return_value=agent),
        patch.object(adapter, "_agent_usage_snapshot", return_value={}),
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"session_id": session_id, "input": "before that", "model": "test-model"},
            )
            assert resp.status == 202, await resp.text()
            for _ in range(50):
                if agent.call is not None:
                    break
                await asyncio.sleep(0.02)

    assert agent.call is not None
    assert agent.call["user_message"] == "before that"
    assert agent.call["task_id"] == session_id
    assert agent.call["conversation_history"] == [
        {"role": "user", "content": "terminal twice"},
        {"role": "assistant", "content": "Ran terminal twice."},
    ]


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
    assert kwargs["conversation_history"] == [
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
async def test_session_replay_api_exposes_stable_items_events_and_state(adapter, session_db):
    session_id = session_db.create_session("sync-session", "api_server")
    session_db.append_message(session_id, "user", "please inspect")
    session_db.append_message(
        session_id,
        "assistant",
        "",
        tool_calls=[{"id": "call_1", "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}}],
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        items_resp = await cli.get(f"/v1/sessions/{session_id}/items")
        assert items_resp.status == 200
        items_payload = await items_resp.json()
        item_ids = [item["id"] for item in items_payload["data"]]
        assert item_ids[0].startswith("msg:")
        assert any(item_id.endswith(":tool_call:call_1") for item_id in item_ids)
        assert items_payload["latest_event_cursor"] is not None

        events_resp = await cli.get(f"/v1/sessions/{session_id}/events?after=0")
        assert events_resp.status == 200
        events_payload = await events_resp.json()
        assert events_payload["object"] == "list"
        assert all(event["event_id"].startswith("evt:") for event in events_payload["data"])
        assert [event["event_type"] for event in events_payload["data"]].count("session.item.upserted") >= 2
        cursor = events_payload["data"][0]["cursor"]

        replay_resp = await cli.get(f"/v1/sessions/{session_id}/events?after={cursor}")
        assert replay_resp.status == 200
        replay_payload = await replay_resp.json()
        assert all(int(event["cursor"]) > int(cursor) for event in replay_payload["data"])

        state_resp = await cli.get(f"/v1/sessions/{session_id}/state")
        assert state_resp.status == 200
        state_payload = await state_resp.json()
        assert state_payload["state"] == "interrupted"
        assert state_payload["pending_tool_call_ids"] == ["call_1"]

        active_resp = await cli.get("/v1/sessions/active")
        assert active_resp.status == 200
        active_payload = await active_resp.json()
        listed = {session["id"]: session for session in active_payload["data"]}
        assert listed[session_id]["sync_state"]["state"] == "interrupted"


@pytest.mark.asyncio
async def test_session_state_and_firehose_include_run_terminal_events(adapter, session_db):
    session_id = session_db.create_session("run-terminal-session", "api_server")
    adapter._set_run_status("run_test", "running", session_id=session_id)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        state_resp = await cli.get(f"/v1/sessions/{session_id}/state")
        assert state_resp.status == 200
        state_payload = await state_resp.json()
        assert state_payload["state"] == "processing"
        assert state_payload["active"] is True

        cursor = session_db.latest_session_event_cursor(session_id) or "0"
        adapter._record_run_session_event(
            session_id=session_id,
            event_type="run.completed",
            run_id="run_test",
            payload={"status": "completed", "output": "done"},
        )

        events_resp = await cli.get(f"/v1/sessions/{session_id}/events?after={cursor}")
        assert events_resp.status == 200
        events_payload = await events_resp.json()

    terminal_event = events_payload["data"][0]
    assert terminal_event["event_id"].startswith("evt:")
    assert terminal_event["event_type"] == "run.completed"
    assert terminal_event["item_id"] == "run:run_test"
    assert terminal_event["payload"]["run_id"] == "run_test"
    assert terminal_event["payload"]["status"] == "completed"


@pytest.mark.asyncio
async def test_session_firehose_streams_missed_events(adapter, session_db):
    session_id = session_db.create_session("firehose-session", "api_server")
    cursor = session_db.latest_session_event_cursor() or "0"

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        ws = await cli.ws_connect(f"/v1/session_events/firehose?after={cursor}")
        session_db.append_message(session_id, "user", "from firehose")
        try:
            message = await asyncio.wait_for(ws.receive(), timeout=3)
            assert message.type == web.WSMsgType.TEXT
            payload = message.json()
            assert payload["session_id"] == session_id
            assert payload["event_type"] == "session.item.upserted"
            assert payload["payload"]["item"]["id"].startswith("msg:")
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_session_events_ack_records_monotonic_consumer_offset(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/session_events/ack", json={"consumer": "nako", "cursor": "7"})
        assert resp.status == 200
        body = await resp.json()
        assert body["object"] == "hermes.session_events.ack"
        assert body["consumer"] == "nako"
        assert body["cursor"] == "7"

        # Idempotent + monotonic: a lower/equal ack must never rewind the offset.
        resp = await cli.post("/v1/session_events/ack", json={"consumer": "nako", "cursor": "3"})
        assert (await resp.json())["cursor"] == "7"

        # A newer ack advances it.
        resp = await cli.post("/v1/session_events/ack", json={"consumer": "nako", "cursor": "9"})
        assert (await resp.json())["cursor"] == "9"

    assert session_db.get_consumer_offset("nako") == "9"
    assert session_db.get_consumer_offset("other") is None


@pytest.mark.asyncio
async def test_session_events_ack_validates_body(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/session_events/ack", json={"cursor": "1"})
        assert resp.status == 400
        resp = await cli.post("/v1/session_events/ack", json={"consumer": "nako"})
        assert resp.status == 400
        resp = await cli.post("/v1/session_events/ack", data="not-json")
        assert resp.status == 400


@pytest.mark.asyncio
async def test_session_firehose_resumes_from_consumer_offset(adapter, session_db):
    session_id = session_db.create_session("firehose-consumer-session", "api_server")
    # One event already exists and nako has durably acknowledged it.
    session_db.append_message(session_id, "user", "first")
    first_cursor = session_db.latest_session_event_cursor()
    session_db.acknowledge_consumer_offset("nako", first_cursor)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        # No explicit `after`: the firehose must resume from nako's acked offset,
        # skip the already-stored event, and replay only the newer one.
        ws = await cli.ws_connect("/v1/session_events/firehose?consumer=nako")
        session_db.append_message(session_id, "assistant", "second")
        try:
            message = await asyncio.wait_for(ws.receive(), timeout=3)
            assert message.type == web.WSMsgType.TEXT
            payload = message.json()
            assert int(payload["cursor"]) > int(first_cursor)
        finally:
            await ws.close()


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
async def test_session_chat_stream_forwards_model_override_and_records_live_tool_items(adapter, session_db):
    import json as _json

    session_id = session_db.create_session("stream-tool-session", "api_server")
    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        assert adapter._active_response_agents_by_session.get(session_id) is kwargs["agent_ref"]
        kwargs["tool_start_callback"]("call_1", "terminal", {"cmd": "pwd"})
        kwargs["tool_complete_callback"]("call_1", "terminal", {"cmd": "pwd"}, "/Users/quill")
        return {
            "final_response": "done",
            "session_id": session_id,
            "messages": [{"role": "assistant", "content": "done"}],
        }, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with (
        patch.object(
            adapter,
            "_resolve_request_model_override",
            return_value=({"model": "gpt-5.5", "provider": "openai-codex"}, None),
        ),
        patch.object(adapter, "_run_agent", side_effect=fake_run),
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "message": "use a tool",
                },
            )
            assert resp.status == 200
            body = await resp.text()

    assert captured_kwargs["request_model_override"]["model"] == "gpt-5.5"
    run_id = None
    for block in body.split("\n\n"):
        if "event: run.started" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_id = _json.loads(line[len("data: "):])["run_id"]
            break
    assert run_id is not None, body
    events = session_db.list_session_events(session_id, after=0)
    live_tool_events = [
        event
        for event in events
        if event["event_type"] == "session.item.upserted"
        and event.get("response_id") == run_id
    ]
    assert len(live_tool_events) == 2
    item_types = [event["payload"]["item"]["type"] for event in live_tool_events]
    assert item_types == ["tool_call", "tool_result"]
    assert session_id not in adapter._active_response_agents_by_session


@pytest.mark.asyncio
async def test_session_halt_interrupts_active_chat_stream_agent(adapter, session_db):
    session_id = session_db.create_session("halt-stream-session", "api_server")

    class Agent:
        def __init__(self):
            self.reason = None

        def interrupt(self, reason):
            self.reason = reason

    agent = Agent()
    adapter._active_response_agents_by_session[session_id] = [agent]
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/v1/sessions/{session_id}/halt",
            json={"reason": "user stopped it"},
        )
        assert resp.status == 200
        body = await resp.json()

    assert body["accepted"] is True
    assert body["session_id"] == session_id
    assert agent.reason == "user stopped it"


@pytest.mark.asyncio
async def test_session_queue_drains_after_active_chat_stream_turn(adapter, session_db):
    import json as _json

    session_id = session_db.create_session("queue-stream-session", "api_server")
    first_turn_started = asyncio.Event()
    release_first_turn = asyncio.Event()
    seen_messages = []

    async def fake_run(**kwargs):
        seen_messages.append(kwargs["user_message"])
        if len(seen_messages) == 1:
            first_turn_started.set()
            await release_first_turn.wait()
        return {
            "final_response": f"done {len(seen_messages)}",
            "session_id": session_id,
            "messages": [{"role": "assistant", "content": f"done {len(seen_messages)}"}],
        }, {"total_tokens": len(seen_messages)}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            stream_task = asyncio.create_task(
                cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "first"})
            )
            await asyncio.wait_for(first_turn_started.wait(), timeout=2)
            queue_resp = await cli.post(
                f"/v1/sessions/{session_id}/queue",
                json={"input": [{"role": "user", "content": "second"}]},
            )
            assert queue_resp.status == 202, await queue_resp.text()
            queued = await queue_resp.json()
            release_first_turn.set()
            stream_resp = await stream_task
            assert stream_resp.status == 200
            stream_body = await stream_resp.text()

    assert seen_messages == ["first", "second"]
    run_started_payloads = []
    for block in stream_body.split("\n\n"):
        if "event: run.started" not in block:
            continue
        for line in block.splitlines():
            if line.startswith("data: "):
                run_started_payloads.append(_json.loads(line[len("data: "):]))
    assert len(run_started_payloads) == 2
    assert run_started_payloads[1]["run_id"] == queued["run_id"]
    assert run_started_payloads[1]["user_message"]["content"] == "second"


@pytest.mark.asyncio
async def test_session_chat_stream_persists_replayable_run_state(adapter, session_db):
    import json as _json

    session_id = session_db.create_session("durable-stream-session", "api_server")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"]("Hello")
        kwargs["tool_progress_callback"]("reasoning.available", tool_name="_thinking", preview="thinking")
        kwargs["stream_delta_callback"](" world")
        return {"final_response": "Hello world", "session_id": session_id}, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "stream please"})
            assert resp.status == 200
            body = await resp.text()

            active_resp = await cli.get(f"/api/sessions/{session_id}/active-run")
            assert active_resp.status == 200
            active = await active_resp.json()

            run = active["run"]
            run_id = run["run_id"]
            replay_resp = await cli.get(f"/api/session-runs/{run_id}/events?after=0")
            assert replay_resp.status == 200
            replay_body = await replay_resp.text()

    assert run["session_id"] == session_id
    assert run["status"] == "completed"
    assert run["user_message"] == "stream please"
    assert run["draft"] == "Hello world"
    assert run["reasoning"] == "thinking"
    assert run["last_seq"] >= 7

    assert "event: run.started" in replay_body
    assert "event: assistant.delta" in replay_body
    assert "event: run.completed" in replay_body
    assert "event: done" in replay_body

    replay_payloads = []
    for block in replay_body.split("\n\n"):
        data_line = next((line for line in block.splitlines() if line.startswith("data: ")), None)
        if data_line:
            replay_payloads.append(_json.loads(data_line[len("data: "):]))
    assert [payload["seq"] for payload in replay_payloads] == sorted(payload["seq"] for payload in replay_payloads)
    assert run_id in body


def test_response_store_marks_incomplete_session_runs_interrupted_after_restart(tmp_path):
    db_path = tmp_path / "response_store.db"
    store = ResponseStore(db_path=str(db_path))
    store.create_session_run("run_pending", "session-1", "hello")
    store.put_session_run_event("run_pending", "run.started", {})
    assert store.get_session_run("run_pending")["status"] == "running"
    store.close()

    restarted_store = ResponseStore(db_path=str(db_path))
    try:
        run = restarted_store.get_session_run("run_pending")
        assert run["status"] == "interrupted"
        assert "restarted" in run["error"]
        events = restarted_store.list_session_run_events("run_pending")
        assert [event["event"] for event in events] == ["run.started"]
    finally:
        restarted_store.close()


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
