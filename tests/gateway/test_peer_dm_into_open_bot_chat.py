"""A peer DM into a Bot Chat is answered by that chat, never by a second writer.

``hermes peer dm`` posts to ``/api/sessions/{id}/chat`` on the peer. Under session authority the
Bot Chat's single writer is the authority FIFO and the turn enters through the ``bot_relay.deliver``
door (``tests/gateway/test_peer_dm_bot_chat_authority.py``). This file keeps the no-authority lane:
a Desktop lease is a viewer, not an executor (the lease-advertised mailbox consumer was retired), so
without an authority to hand the chat to, this process runs the turn itself and no mailbox record
is admitted.
"""

from __future__ import annotations

import asyncio
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
        ("scratch", True, True, 200, "ran here", True),
        ("bot-chat", True, False, 200, "ran here", True),
        ("bot-chat", False, False, 200, "ran here", True),
    ],
    ids=["other-session", "bot-chat-open-in-a-viewer", "bot-chat-not-open"],
)
async def test_a_peer_turn_into_an_open_bot_chat_is_answered_by_its_live_owner(
    tmp_path, monkeypatch, target, open_in_desktop, owner_replies, status, content, turn_ran_here
):
    """Without a profile authority nothing else may execute the Bot Chat's turn: a Desktop viewer
    lease is not an owner, so the turn runs here and no mailbox delivery is admitted."""
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


def _sse_events(raw: str) -> list[tuple[str, dict]]:
    events = []
    for frame in raw.split("\n\n"):
        lines = [line for line in frame.splitlines() if line and not line.startswith(":")]
        if lines and lines[0].startswith("event: "):
            events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def _runs_app(adapter):
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _owner_settles(home, *, status, reply="", error="", reason="", after=0.05):
    """The Desktop's live session: claim the delivery, run it, write the receipt."""
    def _run():
        owner = mailbox.find_canonical_live_owner(home)
        for _ in range(200):
            claimed = mailbox.claim_pending_delivery(home, owner)
            if claimed is not None:
                time.sleep(after)
                mailbox.complete_delivery(home, claimed["delivery_id"], status=status, reply=reply,
                                          error=error, reason=reason)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


async def _poll_terminal(cli, run_id, *, until=("completed", "failed", "cancelled"), tries=100):
    status = {}
    for _ in range(tries):
        status = await (await cli.get(f"/v1/runs/{run_id}")).json()
        if status.get("status") in until:
            break
        await asyncio.sleep(0.05)
    return status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "receipt", "stop", "expected", "turn_ran_here"),
    [
        ("scratch", None, False, ("completed", "ran here"), True),
        ("bot-chat", None, False, ("completed", "ran here"), True),
    ],
    ids=["other-session-runs-here", "bot-chat-runs-here-without-an-authority"],
)
async def test_a_peer_run_into_an_open_bot_chat_is_driven_by_its_owners_receipt(
    tmp_path, monkeypatch, target, receipt, stop, expected, turn_ran_here
):
    """`peer run` into the Bot Chat with no authority to hand it to runs here like any other
    session; the run keeps its run_id and `peer status` keeps working."""
    home = tmp_path.resolve()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    db.create_session("bot-chat", "desktop")
    db.set_session_title("bot-chat", "Bot Chat")
    db.create_session("scratch", "api_server")
    from hermes_cli.active_sessions import try_acquire_active_session
    lease, refusal = try_acquire_active_session(
        session_id="bot-chat", surface="desktop", config={}, registry_home=home, track_liveness=True,
        metadata={"live_session_id": "live-1", "bot_live_delivery_consumer": True})
    assert lease is not None and refusal is None
    owner = _owner_settles(home, status=receipt[0], reply=receipt[1], error=receipt[2], reason=receipt[3]) if receipt else None
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    ran_here = []

    def _create_agent(**kwargs):
        agent = type("Agent", (), {})()
        agent.run_conversation = lambda *a, **k: (ran_here.append(k.get("task_id")), {"final_response": "ran here"})[1]
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        return agent

    try:
        with patch.object(adapter, "_create_agent", side_effect=_create_agent):
            async with TestClient(TestServer(_runs_app(adapter))) as cli:
                resp = await cli.post("/v1/runs", json={"input": "ping", "session_id": target, "author": AUTHOR})
                body = await resp.json()
                assert resp.status == 202, body
                run_id = body["run_id"]
                if stop:
                    status = await _poll_terminal(cli, run_id, until=("running",))
                    assert status["status"] == "running"
                    assert (await cli.post(f"/v1/runs/{run_id}/stop")).status == 200
                status = await _poll_terminal(cli, run_id)
        if owner is not None:
            owner.join(5)
        assert status["status"] == expected[0], status
        assert bool(ran_here) is turn_ran_here
        admitted = sorted((home / "runtime" / "bot_live_delivery").glob("*.json"))
        if turn_ran_here:
            assert status["output"] == expected[1] and not admitted
            return
        [record] = [json.loads(path.read_text()) for path in admitted]
        assert (record["message"], record["author"], record["owner"]["live_session_id"]) == ("ping", AUTHOR, "live-1")
        assert status["delivery_id"] == record["delivery_id"]
        if expected[0] == "completed":
            assert status["output"] == expected[1] and status["completed"] is True
        elif expected[0] == "failed":
            assert status["reason"] == expected[1] and "429" in status["error"]
        else:
            assert status["interrupted"] is True and status["completed"] is False
        assert run_id not in adapter._active_run_tasks, "the run retired like an executor-backed one"
    finally:
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


@pytest.mark.asyncio
async def test_gateway_shutdown_reports_a_live_owned_run_as_interrupted(tmp_path):
    """A `peer run` whose turn is a live Bot Chat's is still a run of THIS gateway: shutdown publishes
    ``interrupted`` and the task cancel that follows must not rewrite it as a bare ``cancelled``."""
    from gateway.platforms import api_server_runs as runs_mod

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    queue: asyncio.Queue = asyncio.Queue()
    adapter._run_streams["run-1"] = queue
    launch = runs_mod._RunLaunch(
        adapter, "run-1", queue, "bot-chat", None, True, "ping", [], False, agent_kwargs={},
        request_profile=None, browser_control_principal=None, browser_control_transport_family=None)
    waiting = asyncio.Event()

    async def _receipt_never_arrives(*_a, **_k):
        waiting.set()
        await asyncio.sleep(3600)

    with patch("tools.bot_live_delivery.await_delivery_async", _receipt_never_arrives):
        task = asyncio.create_task(runs_mod._execute_run_via_live_owner(
            adapter, launch, tmp_path, {"delivery_id": "d" * 32, "status": "queued"}, _api_server=None))
        adapter._active_run_tasks["run-1"] = task
        await asyncio.wait_for(waiting.wait(), 5)
        runs_mod._mark_shutdown_interrupted_runs(adapter, ["run-1"])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    status = adapter._run_statuses["run-1"]
    assert status["status"] == "interrupted"
    assert status["error"] == "Gateway shutdown interrupted the run."
    events = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [e["event"] for e in events if e] == ["run.interrupted"]
