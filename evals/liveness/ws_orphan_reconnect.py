"""Real WS/session.create/disconnect/timers/RPC integration; offline model boundary.
Usage: python evals/liveness/ws_orphan_reconnect.py REPO [redetach]
Seeds a post-interrupt queued envelope to reproduce the bypass-writer schedule,
not the exact naturally arriving queue race. No lifecycle predicate is patched.
"""
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

repo = Path(sys.argv[1]).resolve()
home = tempfile.mkdtemp(prefix="hermes-orphan-wire-")
os.environ.clear()
os.environ.update(HOME=home, HERMES_HOME=home + "/.hermes", PATH="/usr/bin:/bin")
sys.path.insert(0, str(repo))
from tui_gateway import server as s
from tui_gateway import server_requests as sr
from tui_gateway.ws import handle_ws
from fastapi import FastAPI, WebSocket
import uvicorn
from websockets.sync.client import connect


def wait_for(predicate, label):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(label)


class Agent:
    def __init__(self):
        self.interrupted = threading.Event()

    def interrupt(self, *args, **kwargs):
        self.interrupted.set()

    def get_activity_summary(self):
        return {"seconds_since_activity": 100000}


# Only construction and model execution are offline doubles.
def build(sid):
    session = s._sessions[sid]
    session["agent"] = Agent()
    session["agent_ready"].set()


s._schedule_agent_build = build
s._run_prompt_submit = lambda *args, **kwargs: None
s._WS_ORPHAN_REAP_GRACE_S = 0.1
s._WS_ORPHAN_INTERRUPT_REAP_POLL_S = 0.5
app = FastAPI()


@app.websocket("/api/ws")
async def ws_endpoint(ws: WebSocket):
    await handle_ws(ws)


sock = socket.socket()
sock.bind(("127.0.0.1", 0))
url = f"ws://127.0.0.1:{sock.getsockname()[1]}/api/ws"
uv = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
thread = threading.Thread(target=uv.run, kwargs={"sockets": [sock]}, daemon=True)
thread.start()
wait_for(lambda: uv.started, "uvicorn not ready")
sequence = 0


def rpc(ws, method, params):
    global sequence
    sequence += 1
    ws.send(json.dumps({"jsonrpc": "2.0", "id": sequence, "method": method, "params": params}))
    while True:
        for line in ws.recv(timeout=15).splitlines():
            obj = json.loads(line)
            if obj.get("id") == sequence:
                return obj


def detached_running():
    ws = connect(url)
    reply = rpc(ws, "session.create", {})
    sid = reply["result"]["session_id"]
    session = s._sessions[sid]
    transport = session["transport"]
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, daemon=True)
    worker.start()
    with session["history_lock"]:
        session.update(running=True, _run_thread=worker)
    ws.close()
    assert session["agent"].interrupted.wait(15), "orphan did not interrupt"
    wait_for(lambda: sid in s._pending_ws_reaps, "settlement not registered")
    assert session["_client_gone_interrupt_requested"]
    return sid, session, transport, stop, worker


def _bind_offline_agent(sid):
    """The eval's offline boundary: the real session record, a double for the agent's construction."""
    session = s._sessions[sid]
    session["agent"] = Agent()
    session["agent_ready"].set()
    return session


def _capture_warnings():
    """Every root-logger record from here on (the `not in memory` warning is the subject)."""
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.WARNING)
    return records, handler


def dropped_mid_approval(grace, *, reaped):
    """Drop the socket while an approval request is unanswered, the way a reconnect finds it.

    The approval frame goes to the live transport (production `server_requests.send`), the renderer
    never answers it, the socket dies, and the turn is still running — nothing has interrupted it
    yet. Only the agent's construction and model execution are doubles; the client's advertisement
    is the real one (`client.capabilities {server_requests: true}`, which the shared channel sends
    on `gateway.ready`).

    `reaped=False` leaves `ws_orphan_reap_grace_s` unexpired: the runtime is PARKED, exactly what a
    reconnect inside the grace finds. `reaped=True` lets the grace expire and the orphan reap run to
    completion, so the runtime is gone by the time the client comes back.
    """
    s._WS_ORPHAN_REAP_GRACE_S = grace
    ws = connect(url)
    rpc(ws, "client.capabilities", {"server_requests": True})
    sid = rpc(ws, "session.create", {})["result"]["session_id"]
    session = _bind_offline_agent(sid)
    # `session.create` mints the stored key without a row until the first prompt; a persisted chat is
    # what a reconnect meets, and it is what makes the resume take the reuse-live path (whose payload
    # carries `open_requests`). Same call the server's own `_resume_materialize_minted` makes.
    s._get_db().create_session(session["session_key"], source="desktop", model="eval", profile_name="default")
    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, daemon=True)
    worker.start()
    with session["history_lock"]:
        session.update(running=True, _run_thread=worker)
    transport = session["transport"]
    assert sr.answers_requests(transport), "the eval's client did not advertise server_requests"
    answered = {}

    def on_result(result):
        answered["result"] = result

    # Production shape (`server._emit_approval_request` → `send_async`): the request is UNPINNED, so
    # `write_json` routes the frame to the owning session's transport, and whichever socket the user
    # answers on settles it. A pinned request would only be answerable from the socket it was sent to
    # (`server_requests.resolve_response`), which is what a reconnect replaces.
    sr.send_async("approval", sid,
                  {"request_id": "eval-approval", "command": "rm -rf /tmp/eval-approval",
                   "description": "eval approval", "choices": ["once", "deny"]}, on_result)
    frame = json.loads(ws.recv(timeout=15))  # the card a renderer would mount, unanswered
    assert frame["method"] == "approval", frame
    ws.close()
    wait_for(lambda: sid in s._pending_ws_reaps, "reap not armed for the detached session")
    if not reaped:
        return sid, session["session_key"], frame["id"], answered
    # Let the grace expire: the orphan interrupt fires, the detached turn settles, the reap pops.
    assert session["agent"].interrupted.wait(15), "orphan did not interrupt"
    stop.set()
    worker.join(2)
    with session["history_lock"]:
        session["running"] = False
    wait_for(lambda: sid not in s._sessions, "reaped runtime still in memory")
    return sid, session["session_key"], frame["id"], answered


def reconnect_resume_round_trip():
    """Drop mid-approval, reconnect, RESUME the stored session, then answer the card there.

    Two arms, because `ws_orphan_reap_grace_s` decides which runtime the reconnect meets:
    @60 s the runtime is PARKED and the resume reuses it (the card is re-delivered by
    `open_requests` and answerable on the new transport); @0.1 s it is REAPED, so the
    session-scoped follow-up must wait for the resume's FRESH id — the stale id is exactly the
    `session_id … not in memory` the card counts.
    """
    out = {}
    records, handler = _capture_warnings()
    try:
        for grace, arm in ((60.0, "parked"), (0.1, "reaped")):
            stale_sid, session_key, srq, answered = dropped_mid_approval(grace, reaped=arm == "reaped")
            before = len([m for m in records if "not in memory" in m])
            with connect(url) as ws:
                if arm == "reaped":
                    # What the client did before this card: a follow-up keyed on the id it held.
                    stale = rpc(ws, "approval.pending", {"session_id": stale_sid})
                    out["reaped_stale_follow_up"] = {"error": stale.get("error", {}).get("code"),
                                                     "warned": len([m for m in records if "not in memory" in m]) > before}
                    # Re-baseline: that probe is DELIBERATE, and the next number must count only what
                    # the fixed sequence (resume first, then the fresh id) produces.
                    before = len([m for m in records if "not in memory" in m])
                resumed = rpc(ws, "session.resume", {"session_id": session_key, "cols": 96, "source": "desktop",
                                                     "defer_history": True, "omit_messages": True})
                assert "result" in resumed, resumed
                live_sid = resumed["result"]["session_id"]
                reopened = [entry for entry in (resumed["result"].get("open_requests") or [])
                            if entry.get("id") == srq]
                out[arm] = {
                    "reused_runtime": live_sid == stale_sid,
                    "open_request_redelivered": bool(reopened),
                    "warnings_for_resumed_session": len([m for m in records if "not in memory" in m]) - before,
                }
                if arm == "parked":
                    # A session-scoped follow-up after the reconnect must name the runtime the gateway
                    # holds NOW: the fresh id the resume answered with, never the one the client held.
                    out[arm]["follow_up_at_live_id"] = "result" in rpc(ws, "approval.pending", {"session_id": live_sid})
                    # The answer travels as the JSON-RPC response to the card's own id, over the socket
                    # that owns it — the same frame a renderer sends when the user clicks.
                    assert reopened, resumed
                    ws.send(json.dumps({"jsonrpc": "2.0", "id": srq, "result": {"choice": "once"}}))
                    wait_for(lambda: answered.get("result") is not None, "approval answer did not settle")
                    out[arm]["answered_choice"] = answered["result"].get("choice")
                else:
                    assert not reopened, "a reaped runtime's request cannot ride the new runtime's replay"
        # The card's acceptance, as assertions: the parked runtime's card is re-delivered and ANSWERED
        # on the new transport with no stale-id warning; the reaped runtime's follow-up only works at
        # the FRESH id the resume minted — the stale id is exactly the logged warning above.
        assert out["parked"]["reused_runtime"], out
        assert out["parked"]["open_request_redelivered"], out
        assert out["parked"]["answered_choice"] == "once", out
        assert out["parked"]["warnings_for_resumed_session"] == 0, out
        assert out["reaped_stale_follow_up"]["warned"], out
        assert out["reaped"]["reused_runtime"] is False, out
        assert out["reaped"]["warnings_for_resumed_session"] == 0, out
        return out
    finally:
        logging.getLogger().removeHandler(handler)


trace = {"source": s.__file__, "home": home, "fidelity": "actual WebSocket, offline model boundary"}
try:
    # Negative: an orphan which never reconnects must still be reaped.
    sid, session, dead, stop, worker = detached_running()
    stop.set()
    worker.join(2)
    with session["history_lock"]:
        session["running"] = False
    wait_for(lambda: sid not in s._sessions, "unreconnected orphan was not reclaimed")
    assert sid not in s._pending_ws_reaps
    trace["negative_unreconnected_reaped"] = True

    sid, session, dead, stop, worker = detached_running()
    if len(sys.argv) > 2 and sys.argv[2] == "redetach":
        with session["history_lock"]:
            session["running"] = False
            session["queued_prompt"] = {"text": "queued writer", "transport": dead}
        assert s._drain_queued_prompt("drain", sid, session)
        session["running"] = True
        session["agent"].interrupted.clear()
        assert s._close_sessions_for_transport(dead) == (0, 1)
        assert session["agent"].interrupted.wait(15)
        trace["fresh_detachment_polls"] = session["_client_gone_interrupt_polls"]
        stop.set()
        worker.join(2)
        assert trace["fresh_detachment_polls"] == 1, "new detachment inherited old poll budget"
        trace["result"] = "PASS fresh detachment"
        raise SystemExit(0)
    with connect(url) as ws:
        guarded = rpc(ws, "session.resume", {"session_id": session["session_key"]})
        assert guarded.get("error", {}).get("code") == 4009, guarded
        trace["negative_settling_guard"] = 4009
        # Queue-arrival schedule fixture; the transport write is production.
        with session["history_lock"]:
            session["running"] = False
            session["queued_prompt"] = {"text": "queued writer", "transport": dead}
        assert s._drain_queued_prompt("drain", sid, session)
        # `session_auto_continue._drain_queued_prompt` attaches the queuer's transport only while it is
        # ALIVE: "a peer that disconnected while its prompt sat in the queue is skipped — the prompt
        # still runs, only the dead pin is dropped". The pre-#116464 form of this line
        # (`assert session["transport"] is dead`) is RED at the branch tip with none of this card's
        # changes applied (measured on 9ceec7483c); the drain admits the writer and leaves the session
        # detached, which the settlement wait below then retires.
        assert session["transport"] is not dead
        stop.set()
        worker.join(2)
        session["running"] = False
        wait_for(lambda: sid not in s._pending_ws_reaps, "owning settlement not retired")
        # Drop → resume → answer, ahead of the probe tail below (#C6). Placed here because the tail's
        # own assertions are stale at the branch tip (see the notes on its two lines).
        trace["reconnect_resume"] = reconnect_resume_round_trip()
        trace["flag_after_poll"] = session.get("_client_gone_interrupt_requested")
        trace["polls_after_poll"] = session.get("_client_gone_interrupt_polls")
        trace["responses"] = {}
        for method in ("session.resume", "session.activate", "prompt.submit"):
            target = session["session_key"] if method == "session.resume" else sid
            # Only `prompt.submit` takes a body: the attach methods reject an unknown key
            # (4000 "invalid params for session.resume: text: Extra inputs are not permitted").
            # Sending `text` to all three is the pre-existing eval's other stale line, RED at the
            # branch tip before this card's changes (measured on 9ceec7483c).
            params = {"session_id": target, **({"text": "probe"} if method == "prompt.submit" else {})}
            reply = rpc(ws, method, params)
            trace["responses"][method] = {"success": "result" in reply, "error": reply.get("error")}
        assert trace["flag_after_poll"] is None, "abandoned interrupt claim remains"
        assert trace["polls_after_poll"] is None
        assert all(r["success"] for r in trace["responses"].values()), trace
        trace["result"] = "PASS"
finally:
    for sid in list(s._sessions):
        s._cancel_ws_orphan_reap(sid)
    uv.should_exit = True
    thread.join(5)
    print(json.dumps(trace, indent=2), file=sys.stderr, flush=True)
