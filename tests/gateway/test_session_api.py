"""Focused tests for API server session-control endpoints."""

import asyncio
import json
import re
import sqlite3
import threading
from unittest.mock import AsyncMock, MagicMock, patch

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
    app.router.add_post("/api/sessions/search", adapter._handle_search_sessions)
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
    assert features["transcript_derivation_v1"] is True
    assert features["session_search"] is True
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_search"] == {
        "method": "POST",
        "path": "/api/sessions/search",
    }
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }
    assert data["endpoints"]["transcript_derivation_v1"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/fork",
        "required_fields": [
            "derivation_contract", "id", "operation_id", "kind",
            "target_message_id", "boundary",
        ],
        "optional_fields": ["title"],
        "request_discriminator": {
            "field": "derivation_contract",
            "value": "transcript_derivation_v1",
        },
        "target_message_id_format": "msg:v1:<sqlite-int>",
        "kind_boundaries": {
            "branch": "after_turn",
            "edit": "before_turn",
            "retry": "before_turn",
        },
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
        observed["reasoning_callback"] = kwargs["reasoning_callback"]
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)
    reasoning_callback = lambda delta: None

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
        reasoning_callback=reasoning_callback,
    )

    assert result["session_id"] == "request-session"
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
        "reasoning_callback": reasoning_callback,
    }


@pytest.mark.asyncio
async def test_session_search_is_scoped_and_returns_only_safe_summaries(adapter, session_db):
    session_db.create_session("allowed", "api_server")
    session_db.set_session_title("allowed", "Allowed conversation")
    session_db.append_message("allowed", "user", "private transcript needle")
    session_db.create_session("foreign", "api_server")
    session_db.append_message("foreign", "assistant", "private transcript needle")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["allowed"], "limit": 10},
        )
        payload = await response.json()

    assert response.status == 200
    assert [session["id"] for session in payload["data"]] == ["allowed"]
    assert payload["data"][0]["title"] == "Allowed conversation"
    assert set(payload["data"][0]) == {
        "id",
        "title",
        "started_at",
    }
    assert "content" not in json.dumps(payload)
    assert "snippet" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_session_search_marks_result_limit_as_truncated(adapter, session_db):
    for session_id in ("first", "second"):
        session_db.create_session(session_id, "api_server")
        session_db.append_message(session_id, "user", "bounded needle")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["first", "second"], "limit": 1},
        )
        payload = await response.json()

    assert response.status == 200
    assert len(payload["data"]) == 1
    assert payload["truncated"] is True


@pytest.mark.asyncio
async def test_session_search_distinct_results_cannot_be_starved(adapter, session_db):
    session_db.create_session("noisy", "api_server")
    session_db.create_session("quiet", "api_server")
    for index in range(5_001):
        session_db.append_message("noisy", "user", f"needle repeated {index}")
    session_db.append_message("quiet", "assistant", "one needle match")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["noisy", "quiet"], "limit": 2},
        )
        payload = await response.json()

    assert response.status == 200
    assert {session["id"] for session in payload["data"]} == {"noisy", "quiet"}
    assert payload["truncated"] is False


@pytest.mark.asyncio
async def test_session_search_never_expands_beyond_explicit_compression_chain_ids(adapter, session_db):
    session_db.create_session("root", "api_server")
    session_db.end_session("root", "compression")
    session_db.create_session("child", "api_server", parent_session_id="root")
    session_db.append_message("child", "assistant", "forward lineage needle")
    session_db.create_session(
        "branch",
        "api_server",
        parent_session_id="root",
        model_config={"_branched_from": "root"},
    )
    session_db.append_message("branch", "assistant", "branch-only secret")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        forward = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["root", "child"], "limit": 10},
        )
        forward_payload = await forward.json()
        branch = await cli.post(
            "/api/sessions/search",
            json={"query": "secret", "session_ids": ["root"], "limit": 10},
        )
        branch_payload = await branch.json()

    assert forward.status == 200
    assert [item["id"] for item in forward_payload["data"]] == ["child"]
    assert branch.status == 200
    assert branch_payload["data"] == []


@pytest.mark.asyncio
async def test_session_search_owned_tip_does_not_authorize_ancestors(adapter, session_db):
    session_db.create_session("root", "api_server")
    session_db.append_message("root", "user", "ancestor-only needle")
    session_db.end_session("root", "compression")
    session_db.create_session("child", "api_server", parent_session_id="root")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["child"], "limit": 10},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["data"] == []


@pytest.mark.asyncio
async def test_session_search_uses_one_bounded_database_query(adapter, session_db):
    session_ids = [f"owned-{index}" for index in range(3)]
    for session_id in session_ids:
        session_db.create_session(session_id, "api_server")
        session_db.append_message(session_id, "user", "needle")
    statements = []
    session_db._conn.set_trace_callback(statements.append)
    adapter._session_db = session_db
    app = _create_session_app(adapter)
    try:
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/api/sessions/search",
                json={"query": "needle", "session_ids": session_ids, "limit": 10},
            )
    finally:
        session_db._conn.set_trace_callback(None)

    search_statements = [
        statement for statement in statements
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    assert response.status == 200
    assert len(search_statements) == 1


@pytest.mark.asyncio
async def test_session_search_reports_runtime_fts_failure(adapter, session_db):
    session_db.create_session("owned", "api_server")
    session_db.append_message("owned", "user", "needle")
    with session_db._lock:
        session_db._conn.execute("DROP TABLE messages_fts")
        session_db._conn.commit()
    adapter._session_db = session_db

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["owned"], "limit": 10},
        )
        payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "session_search_unavailable"


@pytest.mark.asyncio
async def test_session_search_reports_trigram_fts_failure(adapter, session_db):
    session_db.create_session("owned", "api_server")
    session_db.append_message("owned", "user", "大别山项目")
    with session_db._lock:
        session_db._conn.execute("DROP TABLE messages_fts_trigram")
        session_db._conn.commit()
    adapter._session_db = session_db

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "大别山项目", "session_ids": ["owned"], "limit": 10},
        )
        payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "session_search_unavailable"


@pytest.mark.asyncio
async def test_session_search_reports_general_database_failure(adapter, session_db, monkeypatch):
    session_db.create_session("owned", "api_server")
    adapter._session_db = session_db

    def broken_search(*args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(session_db, "search_messages", broken_search)
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["owned"], "limit": 10},
        )
        payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "session_search_unavailable"


@pytest.mark.asyncio
async def test_session_search_reports_query_budget_exhaustion(adapter, session_db, monkeypatch):
    session_db.create_session("owned", "api_server")
    adapter._session_db = session_db

    def exhausted_search(*args, **kwargs):
        raise TimeoutError("Session search budget exceeded")

    monkeypatch.setattr(session_db, "search_messages", exhausted_search)
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["owned"], "limit": 10},
        )
        payload = await response.json()

    assert response.status == 422
    assert payload["error"]["code"] == "session_search_too_broad"


@pytest.mark.asyncio
async def test_session_search_rejects_non_json_media_type(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            data=json.dumps({"query": "needle", "session_ids": []}),
            headers={"Content-Type": "text/plain"},
        )
        payload = await response.json()

    assert response.status == 415
    assert payload["error"]["code"] == "unsupported_media_type"


@pytest.mark.asyncio
async def test_session_search_groups_explicit_aliases_before_result_limit(adapter, session_db):
    session_ids = []
    session_aliases = {}
    for index in range(251):
        root = f"root-{index}"
        tip = f"tip-{index}"
        for session_id in (root, tip):
            session_db.create_session(session_id, "api_server")
            session_db.append_message(session_id, "user", "needle")
            session_ids.append(session_id)
        session_aliases[root] = tip

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={
                "query": "needle",
                "session_ids": session_ids,
                "session_aliases": session_aliases,
                "limit": 500,
            },
        )
        payload = await response.json()

    assert response.status == 200
    assert len(payload["data"]) == 251
    assert len({item["id"] for item in payload["data"]}) == 251
    assert payload["truncated"] is False


@pytest.mark.asyncio
async def test_session_search_uses_public_alias_metadata(adapter, session_db):
    session_db.create_session("metadata-root", "api_server")
    session_db.set_session_title("metadata-root", "Stale root title")
    session_db.append_message("metadata-root", "user", "metadata needle")
    session_db.end_session("metadata-root", "compression")
    session_db.create_session(
        "metadata-tip", "api_server", parent_session_id="metadata-root"
    )
    session_db.set_session_title("metadata-tip", "Current conversation title")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={
                "query": "needle",
                "session_ids": ["metadata-root", "metadata-tip"],
                "session_aliases": {"metadata-root": "metadata-tip"},
                "limit": 10,
            },
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["data"] == [
        {
            "id": "metadata-tip",
            "title": "Current conversation title",
            "started_at": session_db.get_session("metadata-tip")["started_at"],
        }
    ]


@pytest.mark.asyncio
async def test_cancelled_session_search_holds_permit_until_worker_stops(
    adapter, session_db, monkeypatch
):
    session_db.create_session("cancel-search", "api_server")
    started = threading.Event()
    release = threading.Event()

    def blocked_search(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return []

    monkeypatch.setattr(session_db, "search_messages", blocked_search)
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        request_task = asyncio.create_task(
            cli.post(
                "/api/sessions/search",
                json={
                    "query": "needle",
                    "session_ids": ["cancel-search"],
                    "limit": 10,
                },
            )
        )
        assert await asyncio.to_thread(started.wait, 2)
        request_task.cancel()
        await asyncio.sleep(0.05)
        assert adapter._session_search_semaphore._value == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request_task


@pytest.mark.asyncio
async def test_session_search_accepts_webui_maximum_scope(adapter, session_db):
    session_ids = [f"owned-{index}" for index in range(20_000)]
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": session_ids, "limit": 1},
        )

    assert response.status == 200


@pytest.mark.asyncio
async def test_session_search_rejects_scope_above_webui_maximum(adapter, session_db):
    session_ids = [f"owned-{index}" for index in range(20_001)]
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": session_ids, "limit": 1},
        )
        payload = await response.json()

    assert response.status == 400
    assert "at most 20000" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_session_search_requires_bearer_auth(auth_adapter, session_db):
    session_db.create_session("owned", "api_server")
    session_db.append_message("owned", "user", "needle")
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        unauthorized = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": ["owned"]},
        )
        authorized = await cli.post(
            "/api/sessions/search",
            headers={"Authorization": "Bearer sk-test"},
            json={"query": "needle", "session_ids": ["owned"]},
        )

    assert unauthorized.status == 401
    assert authorized.status == 200


@pytest.mark.asyncio
async def test_session_search_accepts_bounded_500_result_limit(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/sessions/search",
            json={"query": "needle", "session_ids": [], "limit": 500},
        )

    assert response.status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"query": "", "session_ids": []}, "invalid_search_query"),
        ({"query": "x" * 201, "session_ids": []}, "invalid_search_query"),
        ({"query": "x", "session_ids": ["../foreign"]}, "invalid_session_ids"),
        ({"query": "x", "session_ids": [], "limit": True}, "invalid_search_limit"),
        ({"query": "x", "session_ids": [], "limit": 501}, "invalid_search_limit"),
        (
            {
                "query": "x",
                "session_ids": ["owned"],
                "session_aliases": {"owned": []},
            },
            "invalid_session_aliases",
        ),
        ({"query": "x", "session_ids": [], "extra": 1}, "unsupported_search_field"),
    ],
)
async def test_session_search_rejects_invalid_request_shapes(adapter, body, code):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post("/api/sessions/search", json=body)
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == code


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
        assert messages["data"][0]["derivation_id"] == f"msg:v1:{messages['data'][0]['id']}"
        assert messages["data"][1]["derivation_id"] == f"msg:v1:{messages['data'][1]['id']}"

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
    session_db.append_message("tip-session", "user", "after compression")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        messages_resp = await cli.get(f"/api/sessions/{source_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()

    assert messages["object"] == "list"
    assert messages["session_id"] == "tip-session"
    assert [m["content"] for m in messages["data"]] == ["before compression", "after compression"]


@pytest.mark.asyncio
async def test_session_messages_paginate_newest_first_across_compression_lineage(
    adapter, session_db
):
    from agent.context_compressor import SUMMARY_PREFIX

    root_id = session_db.create_session("paged-root", "api_server")
    session_db.append_message(root_id, "user", "oldest")
    session_db.append_message(root_id, "assistant", "older reply")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session(
        "paged-tip", "api_server", parent_session_id=root_id
    )
    session_db.append_message(tip_id, "user", f"{SUMMARY_PREFIX}\ninternal")
    session_db.append_message(tip_id, "assistant", "older reply")
    session_db.append_message(tip_id, "user", "newer question")
    session_db.append_message(tip_id, "assistant", "newest reply")
    session_db.append_message(
        tip_id,
        "tool",
        "large tool output hidden by the chat renderer",
        tool_call_id="call-hidden",
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        latest_response = await cli.get(
            f"/api/sessions/{root_id}/messages?limit=3"
        )
        assert latest_response.status == 200
        latest = await latest_response.json()
        # Cursor boundaries are absolute in the append-only display projection,
        # so a new turn arriving after page one must not shift page two.
        session_db.append_message(tip_id, "user", "arrived after page one")
        session_db.append_message(tip_id, "assistant", "new arrival reply")
        older_response = await cli.get(
            f"/api/sessions/{root_id}/messages"
            f"?limit=3&before={latest['next_cursor']}"
        )
        assert older_response.status == 200
        older = await older_response.json()

    assert latest["session_id"] == tip_id
    assert [m["content"] for m in latest["data"]] == [
        "newer question",
        "newest reply",
        "large tool output hidden by the chat renderer",
    ]
    assert [m["role"] for m in latest["data"]] == ["user", "assistant", "tool"]
    assert latest["has_more"] is True
    assert isinstance(latest["next_cursor"], str)
    assert [m["content"] for m in older["data"]] == [
        "oldest",
        "older reply",
    ]
    assert older["has_more"] is False
    assert older["next_cursor"] is None


@pytest.mark.asyncio
async def test_paginated_session_messages_preserve_tool_rows(adapter, session_db):
    session_id = session_db.create_session("paged-tools", "api_server")
    session_db.append_message(session_id, "user", "run the check")
    session_db.append_message(
        session_id,
        "assistant",
        None,
        tool_calls=[{"name": "terminal", "arguments": "{}"}],
    )
    session_db.append_message(
        session_id,
        "tool",
        "check passed",
        tool_name="terminal",
        tool_call_id="call-1",
    )
    session_db.append_message(session_id, "assistant", "done")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=4"
        )
        payload = await response.json()

    assert response.status == 200
    assert [message["role"] for message in payload["data"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert payload["data"][2]["content"] == "check passed"


@pytest.mark.asyncio
async def test_session_messages_reject_invalid_pagination(adapter, session_db):
    session_id = session_db.create_session("paged-invalid", "api_server")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        too_large = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=501"
        )
        bad_cursor = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=50&before=not-a-cursor"
        )
        oversized_cursor = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=50&before=v1:"
            + ("9" * 5000)
        )
        sqlite_overflow_cursor = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=50&before=v2:9223372036854775808"
        )

    assert too_large.status == 400
    assert bad_cursor.status == 400
    assert oversized_cursor.status == 400
    assert sqlite_overflow_cursor.status == 400

@pytest.mark.asyncio
async def test_session_messages_do_not_prepend_explicit_branch_parent(
    adapter, session_db
):
    """A branch already carries copied history, so display must not concatenate
    the parent again merely because it has parent_session_id set."""
    source_id = session_db.create_session("branch-source", "api_server")
    session_db.append_message(source_id, "user", "one copy only")
    branch_id = session_db.create_session(
        "explicit-branch",
        "api_server",
        parent_session_id=source_id,
        model_config={"_branched_from": source_id},
    )
    source_history = session_db.get_messages_as_conversation(source_id)
    session_db.replace_messages(branch_id, source_history)
    session_db.append_message(branch_id, "assistant", "branch-only reply")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{branch_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert payload["session_id"] == branch_id
    assert [m["content"] for m in payload["data"]] == [
        "one copy only",
        "branch-only reply",
    ]


@pytest.mark.asyncio
async def test_real_api_fork_renders_inherited_history_once(adapter, session_db):
    source_id = session_db.create_session("api-fork-source", "api_server")
    session_db.append_message(source_id, "user", "copied request")
    session_db.append_message(source_id, "assistant", "copied answer")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        fork_response = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={"id": "api-fork-child", "title": "Fork"},
        )
        assert fork_response.status == 201
        session_db.append_message("api-fork-child", "user", "fork-only request")
        response = await cli.get("/api/sessions/api-fork-child/messages")
        payload = await response.json()

    assert response.status == 200
    assert [m["content"] for m in payload["data"]] == [
        "copied request",
        "copied answer",
        "fork-only request",
    ]


@pytest.mark.asyncio
async def test_delegate_display_does_not_prepend_parent_transcript(adapter, session_db):
    parent_id = session_db.create_session("display-parent", "api_server")
    session_db.append_message(parent_id, "user", "PARENT ONLY SECRET")
    child_id = session_db.create_session(
        "display-delegate",
        "tool",
        parent_session_id=parent_id,
        model_config={"_delegate_from": parent_id},
    )
    session_db.append_message(child_id, "assistant", "delegate-only result")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{child_id}/messages")
        payload = await response.json()

    assert response.status == 200
    assert [m["content"] for m in payload["data"]] == ["delegate-only result"]


@pytest.mark.asyncio
async def test_session_messages_do_not_silently_truncate_deep_compression_lineage(
    adapter, session_db
):
    first_id = "deep-000"
    session_db.create_session(first_id, "api_server")
    session_db.append_message(first_id, "user", "message 000")
    previous_id = first_id
    for index in range(1, 102):
        session_db.end_session(previous_id, "compression")
        current_id = f"deep-{index:03d}"
        session_db.create_session(
            current_id, "api_server", parent_session_id=previous_id
        )
        session_db.append_message(current_id, "user", f"message {index:03d}")
        previous_id = current_id

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{first_id}/messages?limit=200")
        payload = await response.json()

    assert response.status == 200
    assert len(payload["data"]) == 102
    assert payload["data"][0]["content"] == "message 000"


def test_session_turn_lock_ignores_delegate_siblings(adapter, session_db):
    root_id = session_db.create_session("lock-root", "api_server")
    session_db.end_session(root_id, "compression")
    session_db.create_session(
        "lock-delegate",
        "tool",
        parent_session_id=root_id,
        model_config={"_delegate_from": root_id},
    )
    tip_id = session_db.create_session(
        "lock-tip",
        "api_server",
        parent_session_id=root_id,
    )

    assert adapter._session_turn_lock(root_id) is adapter._session_turn_lock(tip_id)


@pytest.mark.asyncio
async def test_paginated_history_projection_runs_off_event_loop(
    adapter, session_db, monkeypatch
):
    session_id = session_db.create_session("off-loop-history", "api_server")
    session_db.append_message(session_id, "user", "hello")
    event_loop_thread = threading.get_ident()
    observed = {}
    original = session_db.get_messages_for_display_page

    def checked_page(*args, **kwargs):
        observed["thread"] = threading.get_ident()
        return original(*args, **kwargs)

    monkeypatch.setattr(session_db, "get_messages_for_display_page", checked_page)
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages?limit=1")

    assert response.status == 200
    assert observed["thread"] != event_loop_thread


@pytest.mark.asyncio
async def test_paginated_history_pushes_limit_into_sql(adapter, session_db):
    session_id = session_db.create_session("keyset-history", "api_server")
    for index in range(250):
        session_db.append_message(session_id, "user", f"message {index}")

    statements = []
    session_db._conn.set_trace_callback(statements.append)
    try:
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(
                f"/api/sessions/{session_id}/messages?limit=1"
            )
            payload = await response.json()
    finally:
        session_db._conn.set_trace_callback(None)

    limits = [
        int(match.group(1))
        for statement in statements
        for match in re.finditer(r"\bLIMIT\s+(\d+)", statement, re.IGNORECASE)
    ]
    assert response.status == 200
    assert payload["next_cursor"].startswith("v2:")
    assert len(payload["data"]) == 1
    assert limits and max(limits) <= 64


@pytest.mark.asyncio
async def test_session_history_reports_bounded_projection_overflow(
    adapter, session_db, monkeypatch
):
    from hermes_state import DisplayProjectionTooLargeError

    session_id = session_db.create_session("bounded-history", "api_server")

    def too_large(*args, **kwargs):
        raise DisplayProjectionTooLargeError("too many rows")

    monkeypatch.setattr(session_db, "get_messages_for_display_page", too_large)
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages?limit=1")
        payload = await response.json()

    assert response.status == 422
    assert payload["error"]["code"] == "session_history_too_large"


@pytest.mark.asyncio
async def test_session_messages_hide_compaction_handoff_and_deduplicate_snapshot(
    adapter, session_db
):
    """The user-facing transcript stays continuous across rotating compaction."""
    from agent.context_compressor import SUMMARY_PREFIX

    root_id = session_db.create_session("display-root", "api_server")
    session_db.append_message(root_id, "user", "Build the feature")
    session_db.append_message(root_id, "assistant", "Working on it")
    session_db.end_session(root_id, "compression")

    tip_id = session_db.create_session(
        "display-tip", "api_server", parent_session_id=root_id
    )
    session_db.append_message(tip_id, "user", f"{SUMMARY_PREFIX}\ninternal handoff")
    # A preserved tail row copied into the compacted child must not appear twice.
    session_db.append_message(tip_id, "assistant", "Working on it")
    session_db.append_message(tip_id, "assistant", "Finished and verified")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{root_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert [m["content"] for m in payload["data"]] == [
        "Build the feature",
        "Working on it",
        "Finished and verified",
    ]


@pytest.mark.asyncio
async def test_session_messages_include_archived_turns_after_in_place_compaction(
    adapter, session_db
):
    """Model-context compaction must not erase the visible chat transcript."""
    from agent.context_compressor import SUMMARY_PREFIX

    session_id = session_db.create_session("display-in-place", "api_server")
    session_db.append_message(session_id, "user", "Original request")
    session_db.append_message(session_id, "assistant", "Original visible answer")
    session_db.archive_and_compact(
        session_id,
        [
            {"role": "user", "content": f"{SUMMARY_PREFIX}\ninternal handoff"},
            {"role": "assistant", "content": "Original visible answer"},
        ],
    )
    session_db.append_message(session_id, "assistant", "Continuation after compaction")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert [m["content"] for m in payload["data"]] == [
        "Original request",
        "Original visible answer",
        "Continuation after compaction",
    ]


@pytest.mark.asyncio
async def test_session_messages_preserve_legitimate_repeated_turns(adapter, session_db):
    """Snapshot removal must not become global content deduplication."""
    from agent.context_compressor import SUMMARY_PREFIX

    session_id = session_db.create_session("display-repeats", "api_server")
    for role, content in [
        ("user", "repeat"),
        ("assistant", "ack"),
        ("user", "repeat"),
        ("assistant", "ack"),
    ]:
        session_db.append_message(session_id, role, content)
    session_db.archive_and_compact(
        session_id,
        [{"role": "user", "content": f"{SUMMARY_PREFIX}\ninternal handoff"}],
    )
    session_db.append_message(session_id, "user", "repeat")
    session_db.append_message(session_id, "assistant", "ack")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages")
        assert response.status == 200
        payload = await response.json()

    assert [(m["role"], m["content"]) for m in payload["data"]] == [
        ("user", "repeat"),
        ("assistant", "ack"),
        ("user", "repeat"),
        ("assistant", "ack"),
        ("user", "repeat"),
        ("assistant", "ack"),
    ]


@pytest.mark.asyncio
async def test_session_messages_sanitize_internal_context_fences(adapter, session_db):
    session_id = session_db.create_session("display-sanitized", "api_server")
    session_db.append_message(
        session_id,
        "assistant",
        "<memory-context>PRIVATE INTERNAL MEMORY</memory-context>Visible answer",
    )
    session_db.append_message(
        session_id,
        "tool",
        "<memory-context>PRIVATE TOOL MEMORY</memory-context>Visible tool output",
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(f"/api/sessions/{session_id}/messages")
        assert response.status == 200
        payload = await response.json()

    rendered = "\n".join(str(message.get("content") or "") for message in payload["data"])
    assert "PRIVATE INTERNAL MEMORY" not in rendered
    assert "PRIVATE TOOL MEMORY" not in rendered
    assert "Visible answer" in rendered
    assert "Visible tool output" in rendered


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
async def test_legacy_session_fork_ignores_derivation_like_extension_fields(
    adapter,
    session_db,
):
    source_id = session_db.create_session("legacy-extension-source", "api_server")
    session_db.append_message(source_id, "user", "legacy request")
    session_db.append_message(source_id, "assistant", "legacy answer")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "legacy-extension-child",
                "title": "Legacy extension",
                "operation_id": "client-owned-operation",
                "kind": "experiment",
                "target_message_id": "client-owned-message",
                "boundary": "client-owned-boundary",
            },
        )
        payload = await response.json()

    assert response.status == 201
    assert payload["object"] == "hermes.session"
    assert payload["session"]["id"] == "legacy-extension-child"
    assert [
        message["content"]
        for message in session_db.get_messages("legacy-extension-child")
    ] == ["legacy request", "legacy answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "boundary", "target_name", "expects_retry_request"),
    [
        ("branch", "after_turn", "first_assistant", False),
        ("edit", "before_turn", "second_user", False),
        ("retry", "before_turn", "second_assistant", True),
    ],
)
async def test_session_derivation_v1_returns_action_specific_contract(
    adapter,
    session_db,
    kind,
    boundary,
    target_name,
    expects_retry_request,
):
    source_id = session_db.create_session("derive-source", "api_server", model="test-model")
    session_db.set_session_title(source_id, "Original")
    message_ids = {
        "first_user": session_db.append_message(source_id, "user", "first request"),
        "first_assistant": session_db.append_message(source_id, "assistant", "first answer"),
        "second_user": session_db.append_message(source_id, "user", "second request"),
        "second_assistant": session_db.append_message(source_id, "assistant", "second answer"),
    }
    child_id = f"derive-{kind}"
    operation_id = f"operation-{kind}"
    target_id = message_ids[target_name]
    body = {
        "derivation_contract": "transcript_derivation_v1",
        "id": child_id,
        "operation_id": operation_id,
        "kind": kind,
        "target_message_id": f"msg:v1:{target_id}",
        "boundary": boundary,
        "title": f"Derived {kind}",
    }

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        payload = await response.json()

    assert response.status == 201
    assert payload["object"] == "hermes.session.derivation"
    assert payload["session"]["id"] == child_id
    assert payload["session"]["parent_session_id"] == source_id
    assert payload["session"]["title"] == f"Derived {kind}"
    derivation = payload["derivation"]
    assert derivation == {
        "operation_id": operation_id,
        "kind": kind,
        "boundary": boundary,
        "target_message_id": f"msg:v1:{target_id}",
        "source_requested_session_id": source_id,
        "source_resolved_session_id": source_id,
        "child_session_id": child_id,
        "child_parent_session_id": source_id,
        **({
            "retry_user_message_id": f"msg:v1:{message_ids['second_user']}",
            "retry_user_content": "second request",
            "retry_user_attachments": [],
        } if expects_retry_request else {}),
    }
    assert [message["content"] for message in session_db.get_messages(child_id)] == [
        "first request",
        "first answer",
    ]
    assert session_db.get_session(child_id)["parent_session_id"] == source_id
    assert session_db.get_session(child_id)["title"] == f"Derived {kind}"
    assert session_db.get_session(source_id)["end_reason"] is None
    assert [message["content"] for message in session_db.get_messages(source_id)] == [
        "first request",
        "first answer",
        "second request",
        "second answer",
    ]


@pytest.mark.asyncio
async def test_session_derivation_v1_replays_and_maps_operation_conflict(adapter, session_db):
    source_id = session_db.create_session("derive-replay-source", "api_server")
    session_db.append_message(source_id, "user", "request")
    target_id = session_db.append_message(source_id, "assistant", "answer")
    body = {
        "derivation_contract": "transcript_derivation_v1",
        "id": "derive-replay-child",
        "operation_id": "derive-replay-operation",
        "kind": "branch",
        "target_message_id": f"msg:v1:{target_id}",
        "boundary": "after_turn",
    }

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        first_payload = await first.json()
        replay = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        replay_payload = await replay.json()
        conflict = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={**body, "title": "Conflicting replay"},
        )
        conflict_payload = await conflict.json()

    assert first.status == 201
    assert replay.status == 201
    assert replay_payload == first_payload
    assert [message["content"] for message in session_db.get_messages(body["id"])] == [
        "request",
        "answer",
    ]
    assert conflict.status == 409
    assert conflict_payload["error"]["code"] == "transcript_derivation_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "remove", "expected_code"),
    [
        ({"extra": True}, None, "invalid_transcript_derivation_request"),
        ({}, "boundary", "invalid_transcript_derivation_request"),
        ({"derivation_contract": "transcript_derivation_v2"}, None, "invalid_transcript_derivation_contract"),
        ({"id": ""}, None, "invalid_derivation_session_id"),
        ({"id": "../derived"}, None, "invalid_derivation_session_id"),
        ({"id": "x" * 257}, None, "invalid_derivation_session_id"),
        ({"operation_id": ""}, None, "invalid_derivation_operation_id"),
        ({"operation_id": "x" * 257}, None, "invalid_derivation_operation_id"),
        ({"kind": "copy"}, None, "invalid_derivation_boundary"),
        ({"boundary": "before_turn"}, None, "invalid_derivation_boundary"),
        ({"target_message_id": 1}, None, "invalid_derivation_message_id"),
        ({"target_message_id": "msg:v1:01"}, None, "invalid_derivation_message_id"),
        ({"target_message_id": "msg:v1:9223372036854775808"}, None, "invalid_derivation_message_id"),
        ({"target_message_id": f"msg:v1:{'9' * 100}"}, None, "invalid_derivation_message_id"),
        ({"title": None}, None, "invalid_derivation_title"),
        ({"title": "x" * (SessionDB.MAX_TITLE_LENGTH + 1)}, None, "invalid_derivation_title"),
    ],
)
async def test_session_derivation_v1_strictly_validates_request_shape(
    adapter,
    session_db,
    changes,
    remove,
    expected_code,
):
    source_id = session_db.create_session("derive-validation-source", "api_server")
    session_db.append_message(source_id, "user", "request")
    target_id = session_db.append_message(source_id, "assistant", "answer")
    body = {
        "derivation_contract": "transcript_derivation_v1",
        "id": "derive-validation-child",
        "operation_id": "derive-validation-operation",
        "kind": "branch",
        "target_message_id": f"msg:v1:{target_id}",
        "boundary": "after_turn",
    }
    body.update(changes)
    if remove:
        body.pop(remove)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == expected_code
    assert session_db.get_session("derive-validation-child") is None


@pytest.mark.asyncio
async def test_session_derivation_v1_maps_boundary_validation_to_422(adapter, session_db):
    source_id = session_db.create_session("derive-invalid-target-source", "api_server")
    user_id = session_db.append_message(source_id, "user", "request")
    session_db.append_message(source_id, "assistant", "answer")
    body = {
        "derivation_contract": "transcript_derivation_v1",
        "id": "derive-invalid-target-child",
        "operation_id": "derive-invalid-target-operation",
        "kind": "branch",
        "target_message_id": f"msg:v1:{user_id}",
        "boundary": "after_turn",
    }

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        payload = await response.json()

    assert response.status == 422
    assert payload["error"]["code"] == "transcript_derivation_invalid"
    assert session_db.get_session(body["id"]) is None


@pytest.mark.asyncio
async def test_session_derivation_v1_offloads_and_maps_projection_limit(
    adapter,
    session_db,
    monkeypatch,
):
    from hermes_state import DisplayProjectionTooLargeError

    source_id = session_db.create_session("derive-large-source", "api_server")
    session_db.append_message(source_id, "user", "request")
    target_id = session_db.append_message(source_id, "assistant", "answer")
    body = {
        "derivation_contract": "transcript_derivation_v1",
        "id": "derive-large-child",
        "operation_id": "derive-large-operation",
        "kind": "branch",
        "target_message_id": f"msg:v1:{target_id}",
        "boundary": "after_turn",
    }
    offload_calls = 0

    async def run_off_loop(operation):
        nonlocal offload_calls
        offload_calls += 1
        return operation()

    def exceed_projection(*args):
        raise DisplayProjectionTooLargeError("too large")

    monkeypatch.setattr(
        adapter,
        "_run_blocking_operation_cancellation_safe",
        run_off_loop,
    )
    monkeypatch.setattr(SessionDB, "derive_session_at_boundary", exceed_projection)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        payload = await response.json()

    assert offload_calls == 1
    assert response.status == 422
    assert payload["error"]["code"] == "transcript_derivation_too_large"
    assert session_db.get_session(body["id"]) is None


@pytest.mark.asyncio
async def test_session_derivation_v1_fails_if_child_metadata_is_missing(
    adapter,
    session_db,
    monkeypatch,
):
    source_id = session_db.create_session("derive-missing-child-source", "api_server")
    session_db.append_message(source_id, "user", "request")
    target_id = session_db.append_message(source_id, "assistant", "answer")
    body = {
        "derivation_contract": "transcript_derivation_v1",
        "id": "derive-missing-child",
        "operation_id": "derive-missing-child-operation",
        "kind": "branch",
        "target_message_id": f"msg:v1:{target_id}",
        "boundary": "after_turn",
    }
    monkeypatch.setattr(
        SessionDB,
        "derive_session_at_boundary",
        lambda self, *args: {
            "operation_id": body["operation_id"],
            "kind": body["kind"],
            "boundary": body["boundary"],
            "target_message_id": body["target_message_id"],
            "source_requested_session_id": source_id,
            "source_resolved_session_id": source_id,
            "child_session_id": body["id"],
            "child_parent_session_id": source_id,
        },
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        payload = await response.json()

    assert response.status == 502
    assert payload["error"]["code"] == "transcript_derivation_incomplete"


@pytest.mark.asyncio
async def test_session_derivation_v1_rejects_changed_child_snapshot(
    adapter,
    session_db,
    monkeypatch,
):
    source_id = session_db.create_session("derive-swap-source", "api_server")
    session_db.append_message(source_id, "user", "request")
    target_id = session_db.append_message(source_id, "assistant", "answer")
    body = {
        "derivation_contract": "transcript_derivation_v1",
        "id": "derive-swap-child",
        "operation_id": "derive-swap-operation",
        "kind": "branch",
        "target_message_id": f"msg:v1:{target_id}",
        "boundary": "after_turn",
    }
    original_get_session = SessionDB.get_session

    def swapped_child_snapshot(self, session_id):
        snapshot = original_get_session(self, session_id)
        if session_id == body["id"] and snapshot is not None:
            return {**snapshot, "model_config": "{}"}
        return snapshot

    monkeypatch.setattr(SessionDB, "get_session", swapped_child_snapshot)

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(f"/api/sessions/{source_id}/fork", json=body)
        payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == "transcript_derivation_conflict"


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
        kwargs["reasoning_callback"]("thinking")
        # Legacy prose-derived progress must not be projected as reasoning.
        kwargs["tool_progress_callback"](
            "reasoning.available",
            tool_name="_thinking",
            preview="Hello world",
        )
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
    assert '"delta": "thinking"' in body
    assert '"delta": "Hello world"' not in body
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
async def test_session_chat_stream_rotation_emits_only_current_turn(adapter, session_db):
    """A rotating compaction must not copy the compacted transcript into SSE.

    The pre-turn history and the post-compaction transcript intentionally have
    different prefixes.  Returning every assistant/tool row in run.completed
    can exceed the WebUI event budget and makes a completed turn look broken.
    """
    import json as _json

    parent_id = session_db.create_session("rotation-parent", "api_server")
    session_db.append_message(parent_id, "user", "older request")
    session_db.append_message(parent_id, "assistant", "older answer " + ("x" * 10_000))
    child_id = "rotation-child"

    async def fake_run(**kwargs):
        session_db.end_session(parent_id, "compression")
        session_db.create_session(child_id, "api_server", parent_session_id=parent_id)
        kwargs["stream_delta_callback"]("Current plan")
        kwargs["stream_delta_callback"]("Current answer")
        return {
            "final_response": "Current answer",
            "session_id": child_id,
            "messages": [
                {"role": "user", "content": "[CONTEXT COMPACTION] internal handoff"},
                {"role": "assistant", "content": "older answer " + ("x" * 10_000)},
                {"role": "user", "content": "continue after compaction"},
                {
                    "role": "assistant",
                    "content": "Current plan",
                    "tool_calls": [{
                        "id": "call_current",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }],
                },
                {
                    "role": "tool",
                    "content": "current result",
                    "tool_call_id": "call_current",
                    "tool_name": "terminal",
                },
                {"role": "assistant", "content": "Current answer"},
            ],
        }, {"total_tokens": 7}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{parent_id}/chat/stream",
                json={"message": "continue after compaction"},
            )
            assert resp.status == 200
            body = await resp.text()

    payloads = {}
    for block in body.split("\n\n"):
        lines = block.splitlines()
        event_name = next((line[7:] for line in lines if line.startswith("event: ")), None)
        data_line = next((line[6:] for line in lines if line.startswith("data: ")), None)
        if event_name and data_line:
            payloads[event_name] = _json.loads(data_line)

    completed = payloads["run.completed"]
    assert completed["session_id"] == child_id
    assert [message.get("content") for message in completed["messages"]] == [
        "Current plan",
        "current result",
        "Current answer",
    ]
    assert payloads["done"]["session_id"] == child_id
    assert "older answer" not in _json.dumps(completed)


@pytest.mark.asyncio
async def test_session_chat_stream_resolves_stale_compression_root(adapter, session_db):
    root_id = session_db.create_session("stale-stream-root", "api_server")
    session_db.append_message(root_id, "user", "root turn")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session(
        "stale-stream-tip", "api_server", parent_session_id=root_id
    )
    session_db.append_message(tip_id, "assistant", "tip answer")
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        kwargs["stream_delta_callback"]("continued")
        return {"final_response": "continued", "session_id": tip_id}, {"total_tokens": 1}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{root_id}/chat/stream",
                json={"message": "continue"},
            )
            assert resp.status == 200
            await resp.text()

    assert captured["session_id"] == tip_id
    assert [message["content"] for message in captured["conversation_history"]] == [
        "tip answer"
    ]


@pytest.mark.asyncio
async def test_session_chat_resolves_stale_compression_root(adapter, session_db):
    root_id = session_db.create_session("stale-chat-root", "api_server")
    session_db.append_message(root_id, "user", "root turn")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session(
        "stale-chat-tip", "api_server", parent_session_id=root_id
    )
    session_db.append_message(tip_id, "assistant", "tip answer")
    mock_run = AsyncMock(
        return_value=({"final_response": "continued", "session_id": tip_id}, {"total_tokens": 1})
    )

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{root_id}/chat",
                json={"message": "continue"},
            )
            assert resp.status == 200

    assert mock_run.call_args.kwargs["session_id"] == tip_id
    assert [
        message["content"]
        for message in mock_run.call_args.kwargs["conversation_history"]
    ] == ["tip answer"]


@pytest.mark.asyncio
async def test_session_chat_turns_serialize_on_compression_lineage(adapter, session_db):
    """Queued stale-root turns must re-resolve after a predecessor rotates."""
    import asyncio

    root_id = session_db.create_session("locked-root", "api_server")
    session_db.end_session(root_id, "compression")
    first_tip = session_db.create_session(
        "locked-tip-one", "api_server", parent_session_id=root_id
    )
    session_db.append_message(first_tip, "assistant", "first tip history")
    second_tip = "locked-tip-two"
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs["session_id"])
        if len(calls) == 1:
            first_started.set()
            await release_first.wait()
            session_db.end_session(first_tip, "compression")
            session_db.create_session(
                second_tip, "api_server", parent_session_id=first_tip
            )
            session_db.append_message(second_tip, "assistant", "second tip history")
            return {
                "final_response": "first completed",
                "session_id": second_tip,
            }, {"total_tokens": 1}
        return {
            "final_response": "second completed",
            "session_id": second_tip,
        }, {"total_tokens": 1}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            first_response = await cli.post(
                f"/api/sessions/{root_id}/chat/stream",
                json={"message": "first request"},
            )
            await first_started.wait()
            second_request = asyncio.create_task(cli.post(
                f"/api/sessions/{root_id}/chat",
                json={"message": "second request"},
            ))
            await asyncio.sleep(0.05)
            try:
                assert calls == [first_tip], (
                    "second turn entered before the lineage lock released"
                )
            finally:
                release_first.set()
            await first_response.text()
            second_response = await second_request
            assert second_response.status == 200

    assert calls == [first_tip, second_tip]


@pytest.mark.asyncio
async def test_session_chat_stream_cancellation_holds_lineage_lock_until_agent_stops(
    adapter, session_db
):
    """Cancelling SSE delivery must not release the lock before executor work ends."""
    import asyncio

    root_id = session_db.create_session("cancel-root", "api_server")
    session_db.end_session(root_id, "compression")
    first_tip = session_db.create_session(
        "cancel-tip-one", "api_server", parent_session_id=root_id
    )
    session_db.append_message(first_tip, "assistant", "first tip history")
    second_tip = "cancel-tip-two"
    worker_started = asyncio.Event()
    release_worker = asyncio.Event()
    calls = []
    worker_tasks = []

    async def fake_run(**kwargs):
        calls.append(kwargs["session_id"])
        if len(calls) == 1:
            async def executor_like_worker():
                worker_started.set()
                await release_worker.wait()
                session_db.end_session(first_tip, "compression")
                session_db.create_session(
                    second_tip, "api_server", parent_session_id=first_tip
                )
                session_db.append_message(second_tip, "assistant", "second tip history")
                return {
                    "final_response": "first completed",
                    "session_id": second_tip,
                }, {"total_tokens": 1}

            worker = asyncio.create_task(executor_like_worker())
            worker_tasks.append(worker)
            return await asyncio.shield(worker)
        return {
            "final_response": "second completed",
            "session_id": second_tip,
        }, {"total_tokens": 1}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            first_response = await cli.post(
                f"/api/sessions/{root_id}/chat/stream",
                json={"message": "first request"},
            )
            await worker_started.wait()
            stream_task = next(task for task in adapter._background_tasks if not task.done())
            stream_task.cancel()
            second_request = asyncio.create_task(cli.post(
                f"/api/sessions/{root_id}/chat",
                json={"message": "second request"},
            ))
            await asyncio.sleep(0.05)
            try:
                assert calls == [first_tip], (
                    "cancelled SSE task released its lineage lock while worker continued"
                )
            finally:
                release_worker.set()
            await first_response.text()
            second_response = await second_request
            assert second_response.status == 200

    await asyncio.gather(*worker_tasks, return_exceptions=True)
    assert calls == [first_tip, second_tip]


def test_session_turn_lock_fails_closed_when_lineage_lookup_fails(
    adapter, session_db
):
    root_id = session_db.create_session("fail-closed-root", "api_server")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session(
        "fail-closed-tip", "api_server", parent_session_id=root_id
    )

    with patch.object(
        session_db,
        "get_compression_lineage_root",
        side_effect=RuntimeError("state DB unavailable"),
    ):
        with pytest.raises(RuntimeError, match="state DB unavailable"):
            adapter._session_turn_lock(tip_id)


@pytest.mark.asyncio
async def test_session_chat_stream_resolves_stale_compression_root(adapter, session_db):
    """If the client sends to a compression-ended root, the API must resolve
    to the current tip before running, preventing sibling continuations."""
    root_id = session_db.create_session("root-session", "api_server")
    session_db.append_message(root_id, "user", "hello root")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session("tip-session", "api_server", parent_session_id=root_id)
    session_db.append_message(tip_id, "user", "hello tip")

    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["stream_delta_callback"]("response")
        return {"final_response": "response", "session_id": tip_id}, {"total_tokens": 1}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{root_id}/chat/stream",
                json={"message": "next message"},
            )
            assert resp.status == 200

    assert captured_kwargs["session_id"] == tip_id, (
        "Chat stream must resolve a stale compression root to the current tip"
    )


@pytest.mark.asyncio
async def test_session_chat_resolves_stale_compression_root(adapter, session_db):
    """Non-streaming chat must also resolve a stale root to the tip."""
    root_id = session_db.create_session("root-chat", "api_server")
    session_db.append_message(root_id, "user", "hello root")
    session_db.end_session(root_id, "compression")
    tip_id = session_db.create_session("tip-chat", "api_server", parent_session_id=root_id)
    session_db.append_message(tip_id, "user", "hello tip")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": tip_id}, {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{root_id}/chat",
                json={"message": "next"},
            )
            assert resp.status == 200

    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == tip_id


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
