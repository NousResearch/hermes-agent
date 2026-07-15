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
    app.router.add_get("/api/sessions/{session_id}/telegram-binding", adapter._handle_get_telegram_binding)
    app.router.add_post("/api/sessions/{session_id}/telegram-binding", adapter._handle_post_telegram_binding)
    app.router.add_delete("/api/sessions/{session_id}/telegram-binding", adapter._handle_delete_telegram_binding)
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
async def test_session_chat_telegram_delivery_uses_persistent_binding_state(adapter, session_db):
    session_id = session_db.create_session("telegram-delivery-state", "api_server")
    session_db.bind_telegram_topic(
        chat_id="-100123", thread_id="77", user_id="-100123", session_key="telegram:-100123:77",
        session_id=session_id, managed_mode="api", delivery_enabled=False,
    )
    app = _create_session_app(adapter)
    run = AsyncMock(return_value=({"final_response": "answer", "session_id": session_id}, {}))
    sync = AsyncMock(return_value={"mode": "incremental", "status": "completed", "sent": 0, "failed": 0, "skipped": 0})
    with patch.object(adapter, "_run_agent", run), patch.object(adapter, "_sync_telegram_history", sync):
        async with TestClient(TestServer(app)) as cli:
            disabled = await cli.post(f"/api/sessions/{session_id}/chat", json={"message": "private"})
            assert disabled.status == 200
            assert sync.await_count == 0
            # The legacy field remains a per-request override for callers that
            # need one-off delivery while the durable preference is disabled.
            override = await cli.post(f"/api/sessions/{session_id}/chat", json={"message": "share", "duplicate_to_telegram": True})
            assert override.status == 200
            session_db.set_telegram_topic_delivery_enabled(session_id=session_id, delivery_enabled=True)
            enabled = await cli.post(f"/api/sessions/{session_id}/chat", json={"message": "automatic"})
            assert enabled.status == 200
    # Each delivered turn drains older pending rows before execution and then
    # synchronizes the newly persisted user/assistant rows after completion.
    assert [call.args for call in sync.await_args_list] == [
        (session_id,), (session_id,), (session_id,), (session_id,),
    ]


@pytest.mark.asyncio
async def test_telegram_binding_uses_canonical_db_mapping_and_reports_backfill(adapter, session_db):
    session_id = session_db.create_session("bind-session", "api_server")
    session_db.append_message(session_id, "user", "one")
    session_db.append_message(session_id, "assistant", "two")

    class Telegram:
        async def create_handoff_thread(self, chat_id, name):
            assert chat_id == "-100123"
            return "77"

    adapter._telegram_runner_and_adapter = lambda: (None, Telegram())
    adapter._forward_to_telegram = AsyncMock(return_value={"status": "sent"})
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        created = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"chat_id": "-100123", "backfill": "summary"})
        assert created.status == 201, await created.text()
        payload = await created.json()
        assert payload["bound"] is True
        assert payload["thread_id"] == "77"
        assert payload["delivery_enabled"] is True
        assert payload["future_duplication"] == {
            "persistent": True, "enabled": True, "default": "binding_delivery_enabled",
            "request_field": "duplicate_to_telegram", "request_override": True,
        }
        assert payload["link"] == "https://t.me/c/123/77"
        assert payload["backfill"] == {"mode": "incremental", "status": "completed", "sent": 2, "failed": 0, "skipped": 0}
        fetched = await cli.get(f"/api/sessions/{session_id}/telegram-binding")
        assert (await fetched.json())["bound"] is True
        deleted = await cli.delete(f"/api/sessions/{session_id}/telegram-binding")
        assert deleted.status == 200
        assert (await deleted.json())["deleted"] is True
    assert session_db.get_messages(session_id)[0]["content"] == "one"


@pytest.mark.asyncio
async def test_existing_telegram_binding_toggles_delivery_without_topic_or_backfill(adapter, session_db):
    session_id = session_db.create_session("toggle-bind-session", "api_server")
    created_topics = []

    class Telegram:
        async def create_handoff_thread(self, chat_id, name):
            created_topics.append((chat_id, name))
            return "77"

    adapter._telegram_runner_and_adapter = lambda: (None, Telegram())
    forward = AsyncMock(return_value={"status": "sent"})
    adapter._forward_to_telegram = forward
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        created = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"chat_id": "-100123", "backfill": "none"})
        assert created.status == 201
        disabled = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"delivery_enabled": False})
        assert disabled.status == 200, await disabled.text()
        disabled_payload = await disabled.json()
        assert disabled_payload["delivery_enabled"] is False
        assert disabled_payload["backfill"]["status"] == "not_requested"
        assert forward.await_count == 0
        assert len(created_topics) == 1
        # State is durable and preserves the canonical inbound topic mapping.
        fetched = await cli.get(f"/api/sessions/{session_id}/telegram-binding")
        assert (await fetched.json())["delivery_enabled"] is False
        binding = session_db.get_telegram_topic_binding(chat_id="-100123", thread_id="77")
        assert binding["session_id"] == session_id
        enabled = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"delivery_enabled": True})
        assert enabled.status == 200
        assert (await enabled.json())["delivery_enabled"] is True
    assert len(created_topics) == 1


@pytest.mark.asyncio
async def test_existing_v2_binding_migrates_before_delivery_toggle(adapter, session_db):
    session_id = session_db.create_session("legacy-v2-toggle", "api_server")
    session_db.apply_telegram_topic_migration()
    session_db.bind_telegram_topic(
        chat_id="-100123", thread_id="77", user_id="-100123",
        session_key="telegram:-100123:77", session_id=session_id, managed_mode="api",
    )
    legacy_tail_id = session_db.append_message(session_id, "assistant", "already delivered before v4")
    # Recreate the production v2 shape: existing binding, no delivery column.
    with session_db._lock:
        session_db._conn.executescript(
            """
            ALTER TABLE telegram_dm_topic_bindings RENAME TO telegram_dm_topic_bindings_v3;
            CREATE TABLE telegram_dm_topic_bindings (
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                managed_mode TEXT NOT NULL DEFAULT 'auto',
                linked_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            );
            INSERT INTO telegram_dm_topic_bindings
                SELECT chat_id, thread_id, user_id, session_key, session_id,
                       managed_mode, linked_at, updated_at
                FROM telegram_dm_topic_bindings_v3;
            DROP TABLE telegram_dm_topic_bindings_v3;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_session
                ON telegram_dm_topic_bindings(session_id);
            UPDATE state_meta SET value = '2'
                WHERE key = 'telegram_dm_topic_schema_version';
            """
        )
        session_db._conn.commit()

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        disabled = await cli.post(
            f"/api/sessions/{session_id}/telegram-binding",
            json={"delivery_enabled": False, "backfill": "none"},
        )
        assert disabled.status == 200, await disabled.text()
        payload = await disabled.json()
        assert payload["thread_id"] == "77"
        assert payload["delivery_enabled"] is False
        assert payload["backfill"]["status"] == "not_requested"

    binding = session_db.get_telegram_topic_binding_by_session(session_id=session_id)
    assert binding["thread_id"] == "77"
    assert binding["delivery_enabled"] == 0
    assert binding["last_synced_message_id"] == legacy_tail_id
    assert session_db.get_meta("telegram_dm_topic_schema_version") == "4"


@pytest.mark.asyncio
async def test_existing_telegram_binding_resyncs_clean_user_facing_transcript(adapter, session_db):
    session_id = session_db.create_session("clean-bind-session", "api_server")
    session_db.append_message(session_id, "system", "hidden instruction")
    session_db.append_message(session_id, "user", "visible question")
    session_db.append_message(session_id, "assistant", "")
    session_db.append_message(session_id, "tool", '{"raw":"tool result"}')
    session_db.append_message(session_id, "assistant", '{"result":"structured execution"}')
    session_db.append_message(session_id, "assistant", "visible answer")

    class Telegram:
        async def create_handoff_thread(self, chat_id, name):
            return "77"

    adapter._telegram_runner_and_adapter = lambda: (None, Telegram())
    forward = AsyncMock(return_value={"status": "sent"})
    adapter._forward_to_telegram = forward
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"chat_id": "-100123", "delivery_enabled": False})
        assert first.status == 201
        second = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"delivery_enabled": True, "backfill": "full"})
        assert second.status == 200, await second.text()
        payload = await second.json()
        assert payload["backfill"] == {"mode": "incremental", "status": "completed", "sent": 2, "failed": 0, "skipped": 4}
        messages = await cli.get(f"/api/sessions/{session_id}/messages")
        assert [m["content"] for m in (await messages.json())["data"]] == ["visible question", "visible answer"]

    assert forward.await_args_list[-2].args[:2] == (session_id, "user")
    assert "visible question" in forward.await_args_list[-2].args[2]
    assert forward.await_args_list[-1].args[:2] == (session_id, "assistant")
    assert "visible answer" in forward.await_args_list[-1].args[2]


@pytest.mark.asyncio
async def test_telegram_binding_incrementally_resumes_after_disable_and_failure(adapter, session_db):
    session_id = session_db.create_session("incremental-binding", "api_server")
    first_id = session_db.append_message(session_id, "user", "first")
    hidden_id = session_db.append_message(session_id, "tool", "internal")
    second_id = session_db.append_message(session_id, "assistant", "second")

    class Telegram:
        async def create_handoff_thread(self, chat_id, name):
            return "77"

    adapter._telegram_runner_and_adapter = lambda: (None, Telegram())
    forward = AsyncMock(side_effect=[{"status": "sent"}, {"status": "failed", "error": "down"}])
    adapter._forward_to_telegram = forward
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        # Disabled creation neither sends nor moves the checkpoint.
        created = await cli.post(
            f"/api/sessions/{session_id}/telegram-binding",
            json={"chat_id": "-100123", "delivery_enabled": False},
        )
        assert created.status == 201
        assert forward.await_count == 0

        # Re-enable walks visible + safe skipped rows in order and stops on a
        # send failure, leaving that failed message retryable.
        resumed = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"delivery_enabled": True})
        assert resumed.status == 200
        assert (await resumed.json())["backfill"]["status"] == "partial"
        binding = session_db.get_telegram_topic_binding_by_session(session_id=session_id)
        assert binding["last_synced_message_id"] == hidden_id

        forward.side_effect = [{"status": "sent"}]
        retried = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"delivery_enabled": True})
        assert retried.status == 200
        assert (await retried.json())["backfill"]["sent"] == 1
        binding = session_db.get_telegram_topic_binding_by_session(session_id=session_id)
        assert binding["last_synced_message_id"] == second_id

        # With no new rows a repeated enable must send nothing.
        noop = await cli.post(f"/api/sessions/{session_id}/telegram-binding", json={"delivery_enabled": True})
        assert noop.status == 200
        assert (await noop.json())["backfill"]["sent"] == 0

    assert forward.await_args_list[0].args[1] == "user"
    assert "first" in forward.await_args_list[0].args[2]
    assert "second" in forward.await_args_list[-1].args[2]


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
async def test_session_chat_stream_uses_persistent_telegram_delivery_state(adapter, session_db):
    session_id = session_db.create_session("stream-delivery-state", "api_server")
    session_db.bind_telegram_topic(
        chat_id="-100123", thread_id="77", user_id="-100123", session_key="telegram:-100123:77",
        session_id=session_id, managed_mode="api", delivery_enabled=True,
    )

    async def fake_run(**kwargs):
        return {"final_response": "stream answer", "session_id": session_id}, {}

    sync = AsyncMock(return_value={"mode": "incremental", "status": "completed", "sent": 0, "failed": 0, "skipped": 0})
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run), patch.object(adapter, "_sync_telegram_history", sync):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "stream me"})
            assert response.status == 200
            await response.text()
    assert [call.args for call in sync.await_args_list] == [(session_id,), (session_id,)]


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
