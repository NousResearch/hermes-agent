"""E2E: ``hermes peer dm`` against a REAL api_server gateway whose canonical
Bot Chat session is HIDDEN (issue #91583).

Two real HERMES homes in spirit: the "peer" side is a real
:class:`APIServerAdapter` bound to a real loopback TCP socket over a real
SQLite ``state.db`` (its own tmp HERMES_HOME) containing a hidden
``Bot Chat`` row — exactly what Bot Mode leaves behind. The "local" side is
the stock ``hermes peer dm`` client code (``hermes_cli.subcommands.peer``),
untouched, talking real HTTP with the real API key auth.

Only the model turn itself is stubbed (``_run_agent``); every HTTP handler,
auth check, and SQLite query in the resolution chain is the real thing.
"""

import asyncio
import json
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest
from aiohttp import web

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.subcommands import peer as peer_cmd
from hermes_state import SessionDB

API_KEY = "sk-peer-e2e-key-123456"


def _build_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


@pytest.fixture()
def peer_gateway(tmp_path, monkeypatch):
    """A real api_server gateway (own HERMES_HOME + state.db) on a real socket."""
    peer_home = tmp_path / "peer_home"
    peer_home.mkdir()
    db = SessionDB(peer_home / "state.db")

    # Bot Mode's footprint: the canonical Bot Chat exists and is HIDDEN.
    hidden_id = db.create_session("botchat_hidden_1", "gateway_botmode")
    assert db.set_session_title(hidden_id, "Bot Chat")
    assert db.set_session_hidden(hidden_id, True)
    # And a decoy visible session, so the listing isn't trivially empty.
    other_id = db.create_session("ordinary_1", "api_server")
    db.set_session_title(other_id, "Ordinary Chat")

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    adapter._session_db = db

    async def fake_run_agent(user_message, **kwargs):
        return (
            {
                "final_response": f"e2e reply to: {user_message}",
                "session_id": kwargs.get("session_id"),
            },
            {},
        )

    adapter._run_agent = fake_run_agent

    loop = asyncio.new_event_loop()
    started = threading.Event()
    state = {}

    def _serve():
        asyncio.set_event_loop(loop)

        async def _start():
            runner = web.AppRunner(_build_app(adapter))
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            state["runner"] = runner
            state["port"] = runner.addresses[0][1]
            started.set()

        loop.run_until_complete(_start())
        loop.run_forever()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    assert started.wait(timeout=10), "peer gateway failed to start"
    try:
        yield SimpleNamespace(
            url=f"http://127.0.0.1:{state['port']}",
            db=db,
            hidden_id=hidden_id,
        )
    finally:
        async def _stop():
            await state["runner"].cleanup()

        asyncio.run_coroutine_threadsafe(_stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        close = getattr(db, "close", None)
        if callable(close):
            close()


def test_peer_dm_reaches_hidden_canonical_bot_chat_e2e(peer_gateway, monkeypatch, capsys):
    """The stock peer dm client resolves the HIDDEN canonical Bot Chat on a
    real gateway, runs the chat turn on it, and creates no duplicate row."""
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": peer_gateway.url}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: API_KEY)

    rc = peer_cmd.cmd_peer(
        SimpleNamespace(peer_action="dm", target="spark", message="disk status?", json=True)
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reply"] == "e2e reply to: disk status?"
    assert payload["session_id"] == peer_gateway.hidden_id

    # The real DB still holds exactly ONE Bot Chat row and it is still hidden.
    rows = peer_gateway.db.list_sessions_rich(limit=200, include_hidden=True)
    bot_chats = [r for r in rows if (r.get("title") or "").strip() == "Bot Chat"]
    assert [r["id"] for r in bot_chats] == [peer_gateway.hidden_id]
    assert bool(bot_chats[0].get("hidden"))


def test_hidden_lookup_requires_title_filter_e2e(peer_gateway):
    """Server-side bound: include_hidden without a title filter stays a
    visible-only listing (no blanket hidden exposure on this surface)."""
    import urllib.request

    req = urllib.request.Request(
        f"{peer_gateway.url}/api/sessions?limit=200&include_hidden=1",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        listing = json.loads(resp.read().decode())
    ids = [s["id"] for s in listing["data"]]
    assert peer_gateway.hidden_id not in ids
    assert "ordinary_1" in ids


@pytest.fixture()
def peer_gateway_compressed_hidden(peer_gateway):
    """Aged Bot Mode footprint: the hidden canonical Bot Chat has a live
    compression tip (issue #106165). The tip is untitled and hidden via the
    lineage; the title lives only on the ended root."""
    db = peer_gateway.db
    tip_id = db.create_session(
        "botchat_tip_1", "gateway_botmode", parent_session_id=peer_gateway.hidden_id)
    db.end_session(peer_gateway.hidden_id, "compression")
    db.set_session_hidden(peer_gateway.hidden_id, True)  # lineage-hide root+tip
    peer_gateway.tip_id = tip_id
    return peer_gateway


def test_peer_dm_reaches_compressed_hidden_bot_chat_e2e(
        peer_gateway_compressed_hidden, monkeypatch, capsys):
    """Hidden canonical Bot Chat under compression: the peer lookup must still
    resolve (to the live tip) instead of missing it and colliding with
    UNIQUE(title) on create (HTTP 400, issue #106165)."""
    gw = peer_gateway_compressed_hidden
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": gw.url}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: API_KEY)

    rc = peer_cmd.cmd_peer(
        SimpleNamespace(peer_action="dm", target="spark", message="disk status?", json=True)
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reply"] == "e2e reply to: disk status?"
    assert payload["session_id"] == gw.tip_id

    # No duplicate "Bot Chat" row was minted; the title still lives on the root.
    assert gw.db.get_session_by_title("Bot Chat")["id"] == gw.hidden_id


# ── the resurrection must not run on the gateway's event loop ────────────────

_BLOCK_S = 1.5


def _get(url: str, query: str) -> float:
    """Issue one authenticated GET and return how long it took."""
    req = urllib.request.Request(f"{url}/api/sessions?{query}",
                                 headers={"Authorization": f"Bearer {API_KEY}"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    return time.monotonic() - started


def test_a_slow_bot_chat_resurrection_does_not_stall_the_gateway(peer_gateway, monkeypatch):
    """The canonical Bot Chat lookup is the first step of every `hermes peer dm`, and when the row
    is archived this branch UN-ARCHIVES it — a write. A write that meets a busy database retries in
    a `time.sleep` loop against the 20 s write patience, so running it on the loop thread stalls the
    whole gateway: every other request, every socket, every relay RPC waits it out. This drives a
    real gateway over a real socket and asks the only question that matters: is it still serving?"""
    entered = threading.Event()

    def _slow_unarchive(session_id):
        entered.set()
        time.sleep(_BLOCK_S)
        return False  # nothing to re-list; the point is the wait, not the outcome

    # Hidden rows are excluded without include_hidden, so the empty listing takes the branch.
    monkeypatch.setattr(peer_gateway.db, "get_session_by_title",
                        lambda title: {"id": peer_gateway.hidden_id, "archived": True})
    monkeypatch.setattr(peer_gateway.db, "unarchive_recoverable_session", _slow_unarchive)

    blocked = {}
    caller = threading.Thread(
        target=lambda: blocked.update(elapsed=_get(peer_gateway.url, "title=Bot%20Chat")))
    caller.start()
    try:
        assert entered.wait(timeout=10), "the resurrection branch never ran"
        # The gateway is now inside that write. A trivial listing must still be served.
        elapsed = _get(peer_gateway.url, "title=Ordinary%20Chat")
    finally:
        caller.join(timeout=30)

    assert elapsed < _BLOCK_S / 2, (
        f"a concurrent listing waited {elapsed:.2f}s behind the resurrection write — "
        "the gateway loop is blocked")
    assert blocked["elapsed"] >= _BLOCK_S  # the slow request really did carry the whole wait
