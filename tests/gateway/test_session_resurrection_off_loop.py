"""#113772: Bot Chat resurrection must not run sqlite on the event loop."""

import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_bot_chat_resurrection_runs_off_event_loop(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    try:
        sid = db.create_session("botchat_resurrect_1", "gateway_botmode")
        assert db.set_session_title(sid, "Bot Chat")
        db.end_session(sid, "ws_orphan_reap")
        assert db.set_session_archived(sid, True)

        loop_thread = threading.current_thread()
        off_loop = []
        orig_lookup = db.get_session_by_title
        orig_unarchive = db.unarchive_recoverable_session

        def probed_lookup(title):
            off_loop.append(threading.current_thread() is not loop_thread)
            return orig_lookup(title)

        def probed_unarchive(session_id):
            off_loop.append(threading.current_thread() is not loop_thread)
            return orig_unarchive(session_id)

        monkeypatch.setattr(db, "get_session_by_title", probed_lookup)
        monkeypatch.setattr(db, "unarchive_recoverable_session", probed_unarchive)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._session_db = db
        app = web.Application()
        app.router.add_get("/api/sessions", adapter._handle_list_sessions)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions", params={"title": "Bot Chat"})
            assert resp.status == 200
            data = await resp.json()

        assert off_loop == [True, True]
        assert [s["title"] for s in data["data"]] == ["Bot Chat"]
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()
