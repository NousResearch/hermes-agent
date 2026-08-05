"""Regression tests for compression-lineage display in the session messages
endpoint (#79565).

Desktop session switching prefetches ``GET /api/sessions/{id}/messages``.
The endpoint used to resolve the id to the latest compression child and load
only that child's rows, so a compressed conversation showed only the latest
continuation segment. The fix serves the full lineage display history
(via ``get_resume_conversations``) — the same projection the CLI/TUI resume
path uses — while the model continues to see the compacted tip.
"""

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


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get(
        "/api/sessions/{session_id}/messages", adapter._handle_session_messages
    )
    return app


@pytest.mark.asyncio
async def test_session_messages_returns_full_compression_lineage(session_db):
    """A compressed lineage serves early + tip dialogue, deduped overlap."""
    root_id = "root-session"
    child_id = "child-session"
    session_db.create_session(root_id, "api_server")

    # Early dialogue lives only in the root (before compression ends it).
    session_db.append_message(root_id, "user", "early question")
    session_db.append_message(root_id, "assistant", "early answer")

    session_db.create_session(
        child_id, "api_server", parent_session_id=root_id
    )
    # Parent ended by compression → the tip walk follows the child.
    session_db.end_session(root_id, "compression")

    # The continuation child carries the latest dialogue.
    session_db.append_message(child_id, "user", "latest question")
    session_db.append_message(child_id, "assistant", "latest answer")

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(f"/api/sessions/{root_id}/messages")
        assert resp.status == 200
        data = await resp.json()

    contents = [m.get("content") for m in data["data"]]
    # Both early and latest dialogue visible (the bug: only the child rows).
    assert "early question" in contents
    assert "early answer" in contents
    assert "latest question" in contents
    assert "latest answer" in contents


@pytest.mark.asyncio
async def test_session_messages_uncompressed_unchanged(session_db):
    """A short uncompressed session still returns exactly its own messages."""
    session_id = "plain-session"
    session_db.create_session(session_id, "api_server")
    session_db.append_message(session_id, "user", "hello")
    session_db.append_message(session_id, "assistant", "hi there")

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert resp.status == 200
        data = await resp.json()

    contents = [m.get("content") for m in data["data"]]
    assert contents == ["hello", "hi there"]
