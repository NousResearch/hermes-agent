"""Real WS/session.create/disconnect/timers/RPC integration; offline model boundary.
Usage: python evals/liveness/ws_orphan_reconnect.py REPO [redetach]
Seeds a post-interrupt queued envelope to reproduce the bypass-writer schedule,
not the exact naturally arriving queue race. No lifecycle predicate is patched.
"""
import json
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
        assert session["transport"] is dead
        stop.set()
        worker.join(2)
        session["running"] = False
        wait_for(lambda: sid not in s._pending_ws_reaps, "owning settlement not retired")
        trace["flag_after_poll"] = session.get("_client_gone_interrupt_requested")
        trace["polls_after_poll"] = session.get("_client_gone_interrupt_polls")
        trace["responses"] = {}
        for method in ("session.resume", "session.activate", "prompt.submit"):
            target = session["session_key"] if method == "session.resume" else sid
            reply = rpc(ws, method, {"session_id": target, "text": "probe"})
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
