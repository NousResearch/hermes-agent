"""A peer DM into a Bot Chat that a Desktop holds open is answered BY that open chat.

``hermes peer dm`` posts to ``/api/sessions/{id}/chat`` on the peer. When the peer's canonical Bot
Chat is open in its Desktop, the Desktop session holds the chat's single-writer lease; running the
turn in the API server beside it made a second writer the open chat never saw. The message now goes
through the owner's mailbox, like local and relayed DMs, and the owner's receipt carries the reply.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.subcommands import peer as peer_mod
from hermes_state import SessionDB
from tools import bot_live_delivery as mailbox

AUTHOR = {"id": "bot:cto", "name": "cto", "is_bot": True}


def _app(adapter):
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


def _owner_answers(home, reply):
    """What the Desktop's live session does: claim the delivery, run it as its next turn, settle it."""
    def _run():
        owner = mailbox.find_canonical_live_owner(home)
        for _ in range(200):
            claimed = mailbox.claim_pending_delivery(home, owner)
            if claimed is not None:
                mailbox.complete_delivery(home, claimed["delivery_id"], status="settled", reply=reply)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "open_in_desktop", "owner_replies", "status", "content", "turn_ran_here"),
    [
        ("bot-chat", True, True, 200, "pong", False),
        ("bot-chat", True, False, 202, None, False),
        ("scratch", True, True, 200, "ran here", True),
        ("bot-chat", False, False, 200, "ran here", True),
    ],
    ids=["open-bot-chat-answers", "open-bot-chat-still-running", "other-session", "bot-chat-not-open"],
)
async def test_a_peer_turn_into_an_open_bot_chat_is_answered_by_its_live_owner(
    tmp_path, monkeypatch, target, open_in_desktop, owner_replies, status, content, turn_ran_here
):
    """Only the canonical Bot Chat's live owner takes the turn; every other session, and a Bot Chat
    nobody holds, still runs here. A turn the owner has not finished inside the wait is reported as
    queued in that chat, never as a failure the sender would resend."""
    home = tmp_path.resolve()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("tools.bot_mode_dm._LIVE_WAIT_SECONDS", 1.0)
    db = SessionDB(home / "state.db")
    db.create_session("bot-chat", "desktop")
    db.set_session_title("bot-chat", "Bot Chat")
    db.create_session("scratch", "api_server")
    lease = None
    if open_in_desktop:
        from hermes_cli.active_sessions import try_acquire_active_session
        lease, refusal = try_acquire_active_session(
            session_id="bot-chat", surface="desktop", config={}, registry_home=home, track_liveness=True,
            metadata={"live_session_id": "live-1", "bot_live_delivery_consumer": True})
        assert lease is not None and refusal is None
    owner = _owner_answers(home, "pong") if owner_replies and target == "bot-chat" else None
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    try:
        with patch.object(adapter, "_run_agent", AsyncMock(return_value=({"final_response": "ran here"}, {}))) as run:
            async with TestClient(TestServer(_app(adapter))) as cli:
                resp = await cli.post(f"/api/sessions/{target}/chat", json={"message": "ping", "author": AUTHOR})
                body = await resp.json()
        if owner is not None:
            owner.join(5)
        assert resp.status == status, body
        assert run.called is turn_ran_here
        admitted = sorted((home / "runtime" / "bot_live_delivery").glob("*.json"))
        if turn_ran_here:
            assert body["message"]["content"] == content
            assert not admitted
            return
        [record] = [json.loads(path.read_text()) for path in admitted]
        assert (record["message"], record["author"], record["owner"]["live_session_id"]) == ("ping", AUTHOR, "live-1")
        assert body["delivery_id"] == record["delivery_id"]
        if status == 200:
            assert body["message"]["content"] == content
        else:
            assert (body["object"], body["status"]) == ("hermes.session.chat.queued", "queued")
    finally:
        if lease is not None:
            lease.release()
        db.close()


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_peer_dm_reports_a_turn_queued_in_the_open_bot_chat_as_delivered(monkeypatch, capsys, as_json):
    """The queued answer means the message IS in the peer's open Bot Chat: say so, succeed, and tell
    the sender not to resend, instead of printing ``(no reply)`` as if the turn were empty."""
    monkeypatch.setattr(peer_mod, "_ensure_bot_chat", lambda base, key: "bot-chat")
    monkeypatch.setattr(peer_mod, "_request", lambda url, key, **kw: {
        "object": "hermes.session.chat.queued", "session_id": "bot-chat", "status": "claimed", "delivery_id": "d" * 32})

    code = peer_mod._peer_dm(SimpleNamespace(json=as_json), "hello", "mini", None, "http://peer:8642", "key")
    out = capsys.readouterr().out

    assert code == 0
    assert "(no reply)" not in out
    if as_json:
        assert json.loads(out) == {"peer": "mini", "profile": None, "session_id": "bot-chat",
                                   "status": "claimed", "delivery_id": "d" * 32}
    else:
        assert "went into that chat (session bot-chat)" in out and "Do NOT resend" in out
