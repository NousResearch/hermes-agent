#!/usr/bin/env python
"""RC-1 ship-gate probe (SPEC v0.16, Opus p8 RC-1 / p9 CB-1+CB-3 / p10 CB-C).

Two live WS clients resume ONE session; client-2 STEALS it (session.resume,
WATCH-ONLY — no send). Assert on client-1's own socket:
  (a) ROUTING: client-1's `session.changes` and `session.active_list` replies
      still round-trip to client-1 after the steal.
  (b) SEMANTIC: after the steal (frames diverted to client-2), client-1's
      `session.active_list` status for the session settles to idle/waiting.

Usage:
  ~/.hermes/runtime/hermes-agent/venv/bin/python scripts/livesync-rc1-probe.py \
      <username> <password> [base_url]
Default base: http://mac-studio-m3u:9119 (bound hostname, never raw IP).
Exit 0 = PASS (both assertions), 1 = FAIL.
"""

import asyncio
import http.cookiejar
import json
import sys
import time
import urllib.request

BASE = sys.argv[3] if len(sys.argv) > 3 else "http://mac-studio-m3u:9119"
USER, PW = sys.argv[1], sys.argv[2]


def make_ticket() -> str:
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    last = None
    for _try in range(4):
        try:
            r = urllib.request.Request(
                BASE + "/auth/password-login",
                data=json.dumps(
                    {"provider": "basic", "username": USER, "password": PW}
                ).encode(),
                method="POST",
            )
            r.add_header("Content-Type", "application/json")
            with op.open(r, timeout=60):
                break
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[login retry {_try}] {type(e).__name__}")
            time.sleep(8)
    else:
        raise SystemExit(f"login failed: {last}")
    r = urllib.request.Request(BASE + "/api/auth/ws-ticket", data=b"", method="POST")
    with op.open(r, timeout=60) as x:
        return json.loads(x.read())["ticket"]


class Client:
    def __init__(self, name: str):
        self.name = name
        self.ws = None
        self.waiters: dict = {}
        self.rid = 0
        self.reader_task = None

    async def connect(self):
        import websockets

        ticket = make_ticket()
        self.ws = await websockets.connect(
            BASE.replace("http", "ws") + f"/api/ws?ticket={ticket}",
            max_size=80 * 1024 * 1024,
        )
        self.reader_task = asyncio.create_task(self._reader())

    async def _reader(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if m.get("id") in self.waiters:
                self.waiters[m["id"]].set_result(m)

    async def call(self, method: str, params: dict, timeout: float = 60):
        self.rid += 1
        i = self.rid
        fut = asyncio.get_event_loop().create_future()
        self.waiters[i] = fut
        await self.ws.send(
            json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
        )
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.waiters.pop(i, None)

    async def close(self):
        if self.reader_task:
            self.reader_task.cancel()
        if self.ws:
            await self.ws.close()


def row(label: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


async def main() -> int:
    c1, c2 = Client("client-1"), Client("client-2")
    await c1.connect()
    await c2.connect()
    ok_all = True
    sid = ""
    try:
        # Disposable seeded session (per dashboard-ws-rpc-harness).
        seed = [
            {"role": r_, "content": f"{r_} {i}: probe filler " * 5}
            for i in range(3)
            for r_ in ("user", "assistant")
        ]
        r = await c1.call(
            "session.create", {"messages": seed, "title": "rc1-probe", "source": "tui"}
        )
        sid = (r.get("result") or {}).get("session_id", "")
        if not sid:
            print(f"FATAL: session.create failed: {json.dumps(r)[:300]}")
            return 1
        print(f"probe session: {sid}")

        # Client-1 resumes (becomes transport holder), takes a cursor.
        r = await c1.call("session.resume", {"session_id": sid})
        ok_all &= row("c1 session.resume", "result" in r)
        r = await c1.call(
            "session.changes", {"session_id": sid, "since_message_id": 0}
        )
        pre_ids = [
            m.get("id")
            for m in (r.get("result") or {}).get("messages", [])
            if m.get("id") is not None
        ]
        ok_all &= row(
            "c1 pre-steal session.changes", "result" in r, f"{len(pre_ids)} rows"
        )
        cursor = (r.get("result") or {}).get("last_id", 0)

        # THE STEAL: client-2 resumes the same session. WATCH-ONLY — no send.
        r = await c2.call("session.resume", {"session_id": sid})
        ok_all &= row("c2 steal (session.resume)", "result" in r)
        await asyncio.sleep(1.0)

        # (a) ROUTING: c1's polls/probes still round-trip on ITS socket.
        r = await c1.call(
            "session.changes",
            {"session_id": sid, "since_message_id": cursor},
            timeout=15,
        )
        ok_all &= row(
            "ROUTING: c1 post-steal session.changes reply on c1 socket",
            "result" in r,
        )
        r = await c1.call("session.active_list", {}, timeout=15)
        sessions = (r.get("result") or {}).get("sessions", [])
        ok_all &= row(
            "ROUTING: c1 post-steal session.active_list reply on c1 socket",
            "result" in r,
            f"{len(sessions)} live sessions",
        )

        # (b) SEMANTIC: status for the stolen session settles idle/waiting
        # (no turn is running; a lost frame must read as settled, not working).
        status = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            r = await c1.call("session.active_list", {}, timeout=15)
            for s in (r.get("result") or {}).get("sessions", []):
                if s.get("session_id") == sid or s.get("id") == sid:
                    status = s.get("status")
            if status in ("idle", "waiting"):
                break
            await asyncio.sleep(2)
        ok_all &= row(
            "SEMANTIC: session.active_list status settles idle/waiting",
            status in ("idle", "waiting"),
            f"status={status!r}",
        )
    finally:
        if sid:
            try:
                await c1.call("session.delete", {"session_id": sid}, timeout=30)
            except Exception:  # noqa: BLE001
                print("  [warn] probe session cleanup failed — delete manually:", sid)
        await c1.close()
        await c2.close()

    print("\nRC-1 PROBE:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
