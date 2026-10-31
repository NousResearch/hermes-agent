"""Session binding fields on the API server's session endpoints.

A gateway session row persists its routing binding (``session_key``, ``chat_id``,
``chat_type``, ``thread_id``). An external API client needs those fields to map a
live session back to the chat, group, or topic thread it belongs to: ``user_id`` is
the sender's personal id and is identical across DMs and group messages. Every
session endpoint serializes through ``APIServerAdapter._session_response``, so the
binding must survive it on the list route and on single-session reads, while the
``system_prompt`` / ``model_config`` snapshots stay reduced to presence flags.
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

BINDING_FIELDS = ("session_key", "chat_id", "chat_type", "thread_id")
BINDING = {
    "session_key": "agent:main:telegram:group:-1001234567890:1",
    "chat_id": "-1001234567890",
    "chat_type": "group",
    "thread_id": "1",
}


@pytest.fixture
def adapter(tmp_path):
    """An adapter whose lazy SessionDB is pinned to a scratch state.db holding one
    gateway-bound Telegram session and one CLI session with no gateway origin."""
    a = APIServerAdapter(PlatformConfig(enabled=True))
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="bound", source="telegram", system_prompt="secret prompt")
    db.record_gateway_session_peer("bound", source="telegram", user_id="456", **BINDING)
    db.create_session(session_id="unbound", source="cli")
    a._session_db = db
    try:
        yield a
    finally:
        db.close()


def _app(adapter):
    app = web.Application()
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    return app


@pytest.mark.asyncio
async def test_list_sessions_carries_the_gateway_binding(adapter):
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get("/api/sessions?limit=20&offset=0")
        assert resp.status == 200
        body = await resp.json()

    assert body["object"] == "list"
    rows = {s["id"]: s for s in body["data"]}
    assert {k: rows["bound"][k] for k in BINDING_FIELDS} == BINDING
    assert "system_prompt" not in rows["bound"]
    assert rows["bound"]["has_system_prompt"] is True
    # A session without a gateway origin reports the fields as null, not missing.
    assert {k: rows["unbound"][k] for k in BINDING_FIELDS} == dict.fromkeys(BINDING_FIELDS)


@pytest.mark.asyncio
async def test_single_session_read_carries_the_same_binding(adapter):
    async with TestClient(TestServer(_app(adapter))) as cli:
        listed = (await (await cli.get("/api/sessions")).json())["data"]
        resp = await cli.get("/api/sessions/bound")
        assert resp.status == 200
        session = (await resp.json())["session"]

    assert {k: session[k] for k in BINDING_FIELDS} == BINDING
    listed_bound = next(s for s in listed if s["id"] == "bound")
    assert {k: session[k] for k in BINDING_FIELDS} == {k: listed_bound[k] for k in BINDING_FIELDS}
