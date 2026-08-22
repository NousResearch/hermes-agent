"""``GET /api/sessions`` must be able to include hidden sessions.

``list_sessions_rich`` has always taken ``include_hidden``, but the HTTP
handler never read it from the query string, so every API client saw only the
visible rows. That is not merely a missing filter: session creation enforces
title uniqueness across hidden rows too (raw ``SELECT id FROM sessions WHERE
title = ?``), so anything that looks a session up by title over HTTP and
creates it when absent hits a permanent, unrecoverable collision once that
title belongs to a hidden row.

``hermes peer dm`` is exactly that pattern — it resolves the peer agent's
canonical "Bot Chat" by title — and Bot Mode hides a canonical chat once it
owns it. Observed Aug 2026 on 4 of 4 agents: every peer DM failed with
``Title already in use by session <id>`` naming a session the listing refused
to return.
"""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

AUTH = {"Authorization": "Bearer sk-secret"}


class _RecordingDB:
    """Captures the kwargs the handler forwards to ``list_sessions_rich``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        return []


@pytest.fixture()
def adapter_and_db():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    db = _RecordingDB()
    adapter._session_db = db

    return adapter, db


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)

    return app


@pytest.mark.asyncio
async def test_list_sessions_hides_hidden_rows_by_default(adapter_and_db):
    """The default stays unchanged — hidden rows are still out of the listing."""
    adapter, db = adapter_and_db

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get("/api/sessions", headers=AUTH)

        assert resp.status == 200

    assert db.calls[0]["include_hidden"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["1", "true", "yes"])
async def test_list_sessions_includes_hidden_when_asked(adapter_and_db, raw):
    """``include_hidden`` must reach the DB, in the same truthy spellings the
    sibling ``include_children`` flag already accepts."""
    adapter, db = adapter_and_db

    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get(f"/api/sessions?include_hidden={raw}", headers=AUTH)

        assert resp.status == 200

    assert db.calls[0]["include_hidden"] is True
