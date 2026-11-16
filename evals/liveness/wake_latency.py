#!/usr/bin/env python3
"""wake_latency.py — C1 wake/reconnect phase harness (t_c1f387b2).

Mounts the REAL ``tui_gateway.ws.handle_ws`` in-process (same fidelity line as
evals/liveness/ws_orphan_reconnect.py: actual WebSocket transport + real RPC + offline
model boundary), and drives one scripted client through N sleep/wake cycles, timing the
C1 phase contract per cycle:

    detect_ms pre_ipc_ms dial_ms ready_ms replay_barrier_ms usable_ms

Cycle model (all measured, nothing simulated inside a phase):
  * live leg  — dial, gateway.ready (the NAMED frame, not "first frame"), the first
    application round trip (``client.capabilities``, the same usable probe
    ``deploy/qa_gateway.py phases`` uses), then real ``message.delta`` events emitted
    through ``server._emit`` (seq-stamped into the replay ring like any turn's stream).
  * sleep     — the socket closes (the loopback stand-in for the connection the OS
    discarded while asleep) and no frame is read for --sleep-s.
  * wake      — detect_ms: the scripted client's first liveness attempt (``ping``) on
    the dead generation, measured until it fails. On loopback the close was delivered,
    so this is ~0 BY CONSTRUCTION — exactly like qa_gateway's footer says. The fat the
    brief measured from production logs (backoff ladder + heartbeat deadline, 7.9–8.9 s)
    is not reachable on loopback without netem; see the netem variant below. The real
    per-user detect_ms now rides the Desktop's ``reconnect-episode`` log line.
  * reconnect — dial → ready → replay barrier (``session.events.since`` with the
    watermark the client held across the sleep — events emitted while the socket was
    dead come back through it, so the barrier carries REAL replayed frames, which no
    gateway-side driver can produce) → usable.

pre_ipc_ms (revalidate + getConnection + HTTP ticket mint) lives on the web surface,
not on this mount: ``deploy/qa_gateway.py phases --base-url http://127.0.0.1:10119``
measures it against the real test instance, in the same column names.

Netem-delayed variant (the acceptance card; run on the QA box as root; turns the
loopback legs into a wire):
    tc qdisc add dev lo root netem delay 15ms 5ms
    python evals/liveness/wake_latency.py REPO --cycles 20
    tc qdisc del dev lo root

Usage:
    python evals/liveness/wake_latency.py REPO [--cycles 12] [--sleep-s 0.05] [--json]

Green means: every cycle completed its chain, the replay barrier carried at least one
genuinely-replayed event, and every server teardown line carried the C1 close fields
(``life_s`` / ``silent_s`` / ``initiated``) — C2a reads those from production gui.log.

No ``from __future__ import annotations`` here, on purpose: the endpoint's ``WebSocket``
annotation is imported inside ``main()``, and FastAPI resolves endpoint annotations against
the module globals — a stringized annotation that misses that scope turns every upgrade into
a route-level rejection (403 before the handler runs).
"""

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

PHASES = ("detect_ms", "pre_ipc_ms", "dial_ms", "ready_ms", "replay_barrier_ms", "usable_ms")


def wait_for(predicate, label, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(label)


class Agent:
    """Offline model boundary (the ws_orphan_reconnect.py line): construction and model
    execution are doubles; the transport, RPC, replay ring and teardown are production."""

    def __init__(self):
        self.interrupted = threading.Event()

    def interrupt(self, *args, **kwargs):
        self.interrupted.set()

    def get_activity_summary(self):
        return {"seconds_since_activity": 100000}


class CloseLineFilter(logging.Handler):
    """Captures the production ``ws closed`` teardown lines C1 just extended.

    Append-only list + plain reads: no hand-rolled lock (the interpreter may be a
    free-threaded build where the classic Handler lock reentry pattern parks forever —
    ``list.append`` and ``len`` are individually thread-safe here, and a lost read of a
    line that is one scheduler tick late is corrected by the next poll).
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        text = record.getMessage()
        if text.startswith("ws closed "):
            self.lines.append(text)

    def snapshot(self):
        return list(self.lines)


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def fmt(value):
    return "n/a" if value is None else f"{value:.1f}"


def parse_field(line, name):
    for part in line.split():
        if part.startswith(name + "="):
            return part[len(name) + 1:]
    return None


def main(argv):
    parser = argparse.ArgumentParser(prog="wake_latency.py", description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="path to the hermes-agent checkout to mount")
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--sleep-s", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    home = tempfile.mkdtemp(prefix="hermes-wake-latency-")
    os.environ.clear()
    os.environ.update(HOME=home, HERMES_HOME=home + "/.hermes", PATH="/usr/bin:/bin")
    sys.path.insert(0, str(repo))
    from tui_gateway import server as s
    from tui_gateway.ws import handle_ws
    from fastapi import FastAPI, WebSocket
    import uvicorn
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    def build(sid):
        session = s._sessions[sid]
        session["agent"] = Agent()
        session["agent_ready"].set()

    s._schedule_agent_build = build
    s._run_prompt_submit = lambda *args, **kwargs: None
    # Long grace: the session must survive the whole cycle run detached (a woken client
    # is exactly what the grace window exists for); the harness reaps nothing itself.
    s._WS_ORPHAN_REAP_GRACE_S = 3600.0

    close_lines = CloseLineFilter()
    ws_logger = logging.getLogger("tui_gateway.ws")
    ws_logger.addHandler(close_lines)
    ws_logger.setLevel(logging.INFO)

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
    watermark = {}  # sid -> last seq the CLIENT saw (the desktop's recordEventSeq rule)

    def rpc(ws, method, params):
        nonlocal sequence
        sequence += 1
        my_id = sequence
        ws.send(json.dumps({"jsonrpc": "2.0", "id": my_id, "method": method, "params": params}))
        return my_id

    def read_until(ws, pred, label, budget=15.0):
        """Pump frames until pred(obj) matches; track watermarks on the way, like any client."""
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(label)
            for line in ws.recv(timeout=remaining).splitlines():
                obj = json.loads(line)
                params = obj.get("params") or {}
                if obj.get("method") == "event":
                    sid, seq = params.get("session_id"), params.get("seq")
                    if isinstance(sid, str) and isinstance(seq, int):
                        watermark[sid] = max(watermark.get(sid, 0), seq)
                if pred(obj):
                    return obj

    def is_response(obj, my_id):
        return obj.get("id") == my_id and ("result" in obj or "error" in obj)

    rows = []
    previous = None
    sid = None
    try:
        for cycle in range(1, args.cycles + 1):
            row = {name: None for name in PHASES}
            row["events_replayed"] = 0

            # wake: the first liveness attempt on the dead generation (detect ends when it fails).
            t0 = time.monotonic()
            if previous is None:
                row["detect_ms"] = 0.0  # no prior generation: 0.0 by construction (see docstring)
            else:
                try:
                    previous.ping()
                    raise AssertionError(f"cycle {cycle}: dead socket answered a wake ping")
                except ConnectionClosed:
                    pass  # detection: the first probe failed
                except OSError:
                    pass  # stack-level: send on a discarded socket failed
                row["detect_ms"] = (time.monotonic() - t0) * 1000.0

            # Loopback carries no HTTP ticket surface: pre_ipc_ms is qa_gateway's column
            # on the real surface; never fabricate it here.
            row["pre_ipc_ms"] = None

            # dial: TCP + WS upgrade to the real handle_ws.
            t0 = time.monotonic()
            ws = connect(url, open_timeout=10, close_timeout=2, ping_interval=None)
            row["dial_ms"] = (time.monotonic() - t0) * 1000.0

            # ready: the NAMED gateway.ready frame (C1 semantics: qa_gateway reads the first
            # frame; on this mount the first frame IS gateway.ready — same number, stricter rule).
            read_until(ws, lambda o: (o.get("params") or {}).get("type") == "gateway.ready",
                       f"cycle {cycle}: gateway.ready never arrived")
            row["ready_ms"] = (time.monotonic() - t0) * 1000.0

            # replay barrier: the watermark held across the sleep, resumed on the new socket.
            if sid is not None:
                t_replay = time.monotonic()
                my_id = rpc(ws, "session.events.since", {"session_id": sid, "last_seen": watermark.get(sid, 0)})
                replay = read_until(ws, lambda o: is_response(o, my_id), f"cycle {cycle}: replay reply missing")
                row["replay_barrier_ms"] = (time.monotonic() - t_replay) * 1000.0
                result = replay.get("result") or {}
                row["events_replayed"] = int(result.get("count") or 0)
                if isinstance(result.get("latest_seq"), int):
                    watermark[sid] = max(watermark.get(sid, 0), result["latest_seq"])

            # usable: one real application round trip (qa_gateway's probe).
            t_rpc = time.monotonic()
            my_id = rpc(ws, "client.capabilities", {"capabilities": {}})
            read_until(ws, lambda o: is_response(o, my_id), f"cycle {cycle}: capabilities reply missing")
            row["usable_ms"] = (time.monotonic() - t_rpc) * 1000.0

            if sid is None:
                my_id = rpc(ws, "session.create", {})
                created = read_until(ws, lambda o: is_response(o, my_id), "session.create reply missing")
                sid = created["result"]["session_id"]

            # Traffic for the NEXT cycle's barrier: real _emit events (seq-stamped, ringed).
            # Once the first sleep passed, the session's transport is the detached sentinel,
            # so these land in the ring only — exactly what a reconnecting client replays.
            for i in range(2):
                s._emit("message.delta", sid, {"text": f"wake-cycle-{cycle}-{i}"})

            rows.append(row)
            # sleep: the socket the next cycle wakes to is already dead — a close delivered
            # while nobody reads, which is what sleep does to a loopback TCP connection. Failing
            # to close it is that same death, not a harness fault.
            with contextlib.suppress(Exception):
                ws.close()
            previous = ws
            time.sleep(args.sleep_s)
            wait_for(lambda: len(close_lines.snapshot()) >= cycle,
                     f"cycle {cycle}: server teardown line missing")

        closes = close_lines.snapshot()
        assert len(closes) >= args.cycles, f"{len(closes)} close lines for {args.cycles} cycles"
        for line in closes:
            for field in ("life_s=", "silent_s=", "initiated="):
                assert field in line, f"C1 teardown field {field} missing: {line}"
        replayed = [row["events_replayed"] for row in rows]
        assert any(count > 0 for count in replayed[1:]), f"no cycle ever replayed a frame: {replayed}"

        summary = {}
        for name in PHASES:
            values = [row[name] for row in rows if row[name] is not None]
            summary[name] = {
                "p50": round(percentile(values, 0.50), 3) if values else None,
                "p95": round(percentile(values, 0.95), 3) if values else None,
                "n": len(values),
            }
        silent = [float(parse_field(line, "silent_s")) for line in closes]
        life = [float(parse_field(line, "life_s")) for line in closes]
        initiated = {}
        for line in closes:
            word = parse_field(line, "initiated")
            initiated[word] = initiated.get(word, 0) + 1
        trace = {
            "source": s.__file__,
            "home": home,
            "fidelity": "actual WebSocket, real handle_ws/RPC/replay ring, offline model boundary",
            "cycles": args.cycles,
            "rows": rows,
            "summary": summary,
            "server_closes": {
                "count": len(closes),
                "initiated": initiated,
                "life_s_p50": round(percentile(life, 0.50), 3),
                "silent_s_p50": round(percentile(silent, 0.50), 3),
            },
        }
        if args.json:
            # sys.__stdout__: tui_gateway.server wraps sys.stdout for its stdio transport,
            # which would swallow the report (the orphan eval prints its trace to stderr for
            # the same reason). The real stdout is what callers of this script read.
            print(json.dumps(trace, indent=2), file=sys.__stdout__)
        else:
            def out(*parts):
                print(*parts, file=sys.__stdout__)

            header = "cycle".ljust(6) + "".join(name.rjust(19) for name in PHASES)
            out(header)
            out("-" * len(header))
            for i, row in enumerate(rows, start=1):
                out(str(i).ljust(6) + "".join(fmt(row[name]).rjust(19) for name in PHASES))
            out("-" * len(header))
            for label, q in (("p50", 0.50), ("p95", 0.95)):
                cells = []
                for name in PHASES:
                    values = [row[name] for row in rows if row[name] is not None]
                    cells.append(fmt(percentile(values, q) if values else None).rjust(19))
                out(label.ljust(6) + "".join(cells))
            out(f"\nn={len(rows)} cycles; all values are milliseconds.")
            out(f"server closes: {len(closes)}  initiated: {initiated}  "
                f"silent_s p50 {percentile(silent, 0.50):.3f}  life_s p50 {percentile(life, 0.50):.3f}")
            out("detect_ms is ~0 by construction: loopback delivers the close, so the wake probe")
            out("fails on the first attempt. Backoff + heartbeat-deadline detection (the 7.9-8.9 s")
            out("production story) measures via the Desktop reconnect-episode log line, and with the")
            out("netem variant in this file's docstring.")
            out("pre_ipc_ms is n/a here: this mount has no HTTP ticket surface; deploy/qa_gateway.py")
            out("phases measures it on the real test instance (:10119), same column names.")
            out(f"result: PASS ({sum(1 for row in rows if row['events_replayed'] > 0)} cycles replayed events)")
        return 0
    finally:
        ws_logger.removeHandler(close_lines)
        for orphan in list(s._sessions):
            s._cancel_ws_orphan_reap(orphan)
        uv.should_exit = True
        thread.join(5)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
