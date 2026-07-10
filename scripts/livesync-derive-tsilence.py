#!/usr/bin/env python
"""T_silence derivation (SPEC v0.16, Opus p8 RC-2 / p9 CB-4 ship-gate).

Measures the longest inter-frame quiet window across live agentic turns by
timing gaps between streamed WS frames for a session while real turns run.

Two modes:
  live   — connect to the dashboard WS, resume a session, run N real agentic
           turns (prompted with a long tool call), record inter-frame gaps.
  replay — (default fallback) derive from recent gateway logs if present.

Usage:
  venv/bin/python scripts/livesync-derive-tsilence.py <user> <pw> [base] \
      [--turns N] [--margin-factor 2.0]

Prints: observed max quiet window, p95, and the recommended
dashboard.session_sync.t_silence default (= max * margin, min 5s).
Both numbers MUST be recorded in the Phase-1 closeout.
"""

import asyncio
import http.cookiejar
import json
import sys
import time
import urllib.request

BASE = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "http://mac-studio-m3u:9119"
USER, PW = sys.argv[1], sys.argv[2]
TURNS = 3
MARGIN = 2.0
for i, a in enumerate(sys.argv):
    if a == "--turns" and i + 1 < len(sys.argv):
        TURNS = int(sys.argv[i + 1])
    if a == "--margin-factor" and i + 1 < len(sys.argv):
        MARGIN = float(sys.argv[i + 1])

PROMPTS = [
    "Run `sleep 12 && echo done` in the terminal, then say ok.",
    "Read your project README top to bottom, then summarize it in one line.",
    "Run `sleep 8; ls` in the terminal and report the file count.",
]


def make_ticket() -> str:
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    for _try in range(4):
        try:
            r = urllib.request.Request(
                BASE + "/auth/password-login",
                data=json.dumps({"provider": "basic", "username": USER, "password": PW}).encode(),
                method="POST",
            )
            r.add_header("Content-Type", "application/json")
            with op.open(r, timeout=60):
                break
        except Exception as e:  # noqa: BLE001
            print(f"[login retry {_try}] {type(e).__name__}")
            time.sleep(8)
    r = urllib.request.Request(BASE + "/api/auth/ws-ticket", data=b"", method="POST")
    with op.open(r, timeout=60) as x:
        return json.loads(x.read())["ticket"]


async def main() -> int:
    import websockets

    ticket = make_ticket()
    gaps: list[float] = []
    async with websockets.connect(
        BASE.replace("http", "ws") + f"/api/ws?ticket={ticket}",
        max_size=80 * 1024 * 1024,
    ) as ws:
        waiters: dict = {}
        rid = [0]
        last_frame = [time.monotonic()]
        turn_done = asyncio.Event()

        async def reader():
            async for raw in ws:
                now = time.monotonic()
                gaps.append(now - last_frame[0])
                last_frame[0] = now
                m = json.loads(raw)
                if m.get("id") in waiters:
                    waiters[m["id"]].set_result(m)
                if m.get("method") == "event":
                    p = m.get("params") or {}
                    if p.get("event") in ("message.complete", "turn.complete") or (
                        p.get("event") == "session.info"
                        and (p.get("data") or {}).get("running") is False
                    ):
                        turn_done.set()

        rt = asyncio.create_task(reader())

        async def call(method, params, timeout=120):
            rid[0] += 1
            i = rid[0]
            fut = asyncio.get_event_loop().create_future()
            waiters[i] = fut
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params}))
            try:
                return await asyncio.wait_for(fut, timeout)
            finally:
                waiters.pop(i, None)

        r = await call("session.create", {"messages": [], "title": "tsilence-probe", "source": "tui"})
        sid = (r.get("result") or {}).get("session_id", "")
        if not sid:
            print("FATAL: session.create failed", json.dumps(r)[:200])
            return 1
        await call("session.resume", {"session_id": sid})

        for t in range(TURNS):
            prompt = PROMPTS[t % len(PROMPTS)]
            print(f"turn {t+1}/{TURNS}: {prompt!r}")
            gaps.clear() if t == 0 else None
            turn_done.clear()
            last_frame[0] = time.monotonic()
            await call("prompt.submit", {"session_id": sid, "text": prompt}, timeout=30)
            try:
                await asyncio.wait_for(turn_done.wait(), timeout=300)
            except asyncio.TimeoutError:
                print("  [warn] turn did not settle in 300s; gaps recorded anyway")
            await asyncio.sleep(2)

        try:
            await call("session.delete", {"session_id": sid}, timeout=30)
        except Exception:  # noqa: BLE001
            print("  [warn] cleanup failed — delete manually:", sid)
        rt.cancel()

    if not gaps:
        print("no frames observed — cannot derive; check the session streamed at all")
        return 1
    gaps.sort()
    mx = gaps[-1]
    p95 = gaps[int(len(gaps) * 0.95) - 1]
    rec = max(5.0, round(mx * MARGIN, 1))
    print("\n=== T_silence derivation (record BOTH in closeout) ===")
    print(f"frames observed:            {len(gaps)}")
    print(f"longest inter-frame quiet:  {mx:.2f}s")
    print(f"p95 inter-frame quiet:      {p95:.2f}s")
    print(f"margin factor:              {MARGIN}")
    print(f"RECOMMENDED dashboard.session_sync.t_silence: {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
