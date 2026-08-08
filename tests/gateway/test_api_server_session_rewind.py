"""Tests for POST /api/sessions/{id}/rewind on api_server.

Exposes SessionDB.rewind_to_message for /v1/runs clients (regenerate H3).
Complements web_server PR #60443 — same primitive, api_server surface.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB


def _make_adapter(api_key: str = "sk-secret") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post(
        "/api/sessions/{session_id}/rewind",
        adapter._handle_rewind_session,
    )
    app.router.add_get(
        "/api/sessions/{session_id}/messages",
        adapter._handle_session_messages,
    )
    return app


@pytest.fixture()
def seeded_db_path(tmp_path, monkeypatch):
    """Seed a SessionDB on disk; return path (not a live connection).

    Handler and assertions each open their own SessionDB so close() in the
    handler cannot wipe the connection used for post-condition checks
    (review note on #60443).
    """
    db_path = tmp_path / "state.db"
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", db_path)

    writer = SessionDB(db_path=db_path)
    writer.create_session("s1", source="api_server")
    ids = []
    for i in range(1, 4):
        ids.append(writer.append_message("s1", "user", f"q{i}"))
        ids.append(writer.append_message("s1", "assistant", f"a{i}"))
    writer.close()
    return db_path, ids


@pytest.fixture()
def adapter_with_db(seeded_db_path):
    db_path, ids = seeded_db_path
    adapter = _make_adapter()
    # Keep the adapter's SessionDB open for the request lifetime (handler
    # does not close the shared adapter db — unlike web_server's per-request open).
    adapter._session_db = SessionDB(db_path=db_path)
    yield adapter, ids, db_path
    adapter._session_db.close()


@pytest.mark.asyncio
async def test_rewind_to_user_message_clears_later_context(adapter_with_db):
    adapter, ids, db_path = adapter_with_db
    q2_id = ids[2]  # third seeded row is the 2nd user turn (q2)

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/s1/rewind",
            headers={"Authorization": "Bearer sk-secret"},
            json={"target_message_id": q2_id},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["object"] == "hermes.session.rewind"
        assert body["session_id"] == "s1"
        assert body["rewound_count"] == 4  # q2, a2, q3, a3
        assert body["target_message"]["content"] == "q2"
        assert body["target_message"]["role"] == "user"

    # Separate connection for assertions after the request.
    reader = SessionDB(db_path=db_path)
    try:
        active = reader.get_messages("s1")
        assert [m["content"] for m in active] == ["q1", "a1"]
        assert len(reader.get_messages("s1", include_inactive=True)) == 6
    finally:
        reader.close()


@pytest.mark.asyncio
async def test_rewind_rejects_non_user_target(adapter_with_db):
    adapter, ids, _db_path = adapter_with_db
    a1_id = ids[1]

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/s1/rewind",
            headers={"Authorization": "Bearer sk-secret"},
            json={"target_message_id": a1_id},
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "invalid_rewind_target"


@pytest.mark.asyncio
async def test_rewind_unknown_session_returns_404(adapter_with_db):
    adapter, _ids, _db_path = adapter_with_db
    assert adapter._session_db.get_session("does-not-exist") is None
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/does-not-exist/rewind",
            headers={"Authorization": "Bearer sk-secret"},
            json={"target_message_id": 1},
        )
        body = await resp.json()
        assert resp.status == 404, body
        assert body["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_rewind_requires_target_message_id(adapter_with_db):
    adapter, _ids, _db_path = adapter_with_db
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/s1/rewind",
            headers={"Authorization": "Bearer sk-secret"},
            json={},
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["code"] == "invalid_target_message_id"


@pytest.mark.asyncio
async def test_rewind_rejects_active_run(adapter_with_db):
    adapter, ids, _db_path = adapter_with_db
    adapter._run_statuses["run_busy"] = {
        "status": "running",
        "session_id": "s1",
    }
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/s1/rewind",
            headers={"Authorization": "Bearer sk-secret"},
            json={"target_message_id": ids[2]},
        )
        assert resp.status == 409
        body = await resp.json()
        assert body["error"]["code"] == "session_busy"


@pytest.mark.asyncio
async def test_capabilities_advertise_session_rewind(adapter_with_db):
    adapter, _ids, _db_path = adapter_with_db
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            "/v1/capabilities",
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["features"]["session_rewind"] is True
        assert body["endpoints"]["session_rewind"]["path"] == (
            "/api/sessions/{session_id}/rewind"
        )


@pytest.mark.asyncio
async def test_rewind_guard_blocks_run_admission_while_rewind_in_flight(adapter_with_db):
    """A /v1/runs queued registration cannot slip in during rewind_to_message."""
    adapter, ids, _db_path = adapter_with_db
    q2_id = ids[2]
    rewind_gate = threading.Event()
    rewind_entered = threading.Event()
    original_rewind = adapter._session_db.rewind_to_message

    def hooked_rewind(session_id: str, target_message_id: int):
        rewind_entered.set()
        if not rewind_gate.wait(timeout=2):
            raise AssertionError("rewind gate timed out")
        return original_rewind(session_id, target_message_id)

    adapter._session_db.rewind_to_message = hooked_rewind

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        rewind_task = asyncio.create_task(
            cli.post(
                "/api/sessions/s1/rewind",
                headers={"Authorization": "Bearer sk-secret"},
                json={"target_message_id": q2_id},
            )
        )
        await asyncio.to_thread(rewind_entered.wait, 2)

        admit_err = await adapter._admit_run_for_session(
            "run_race",
            "s1",
            created_at=0.0,
            model="test-model",
        )
        assert admit_err is not None
        body = json.loads(admit_err.text)
        assert body["error"]["code"] == "session_rewind_in_progress"
        assert "s1" not in adapter._run_statuses

        rewind_gate.set()
        resp = await rewind_task
        assert resp.status == 200


@pytest.mark.asyncio
async def test_rewind_rejects_queued_run_registered_under_guard(adapter_with_db):
    adapter, ids, _db_path = adapter_with_db
    admit_err = await adapter._admit_run_for_session(
        "run_busy",
        "s1",
        created_at=0.0,
        model="test-model",
    )
    assert admit_err is None

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/s1/rewind",
            headers={"Authorization": "Bearer sk-secret"},
            json={"target_message_id": ids[2]},
        )
        assert resp.status == 409
        body = await resp.json()
        assert body["error"]["code"] == "session_busy"


@pytest.mark.asyncio
async def test_session_chat_rejects_rewind_in_progress(adapter_with_db):
    adapter, ids, _db_path = adapter_with_db
    adapter._sessions_rewinding.add("s1")

    rewind_err = await adapter._session_rewind_blocked_response("s1")
    assert rewind_err is not None
    body = json.loads(rewind_err.text)
    assert body["error"]["code"] == "session_rewind_in_progress"
