"""A1 live proof: cross-process commit→fetch latency through session.changes.

Polls the dashboard's session.changes (the desktop's poll path) at the shipped
2.5s interval against a session the MESSAGING GATEWAY (other process) is
writing. For each newly returned row, latency = fetch_time - row.timestamp.
This is the gateway-written/dashboard-read A1 proof with measured numbers.
"""
import asyncio, json, sys, time
sys.path.insert(0, 'scripts')
probe = __import__('livesync-rc1-probe')

SID = sys.argv[4] if len(sys.argv) > 4 else ""
DURATION = 90

if not SID:
    raise SystemExit(
        "usage: livesync-a1-watch.py <user> <pw> <base_url> <stored_session_id>\n"
        "stored_session_id is REQUIRED (state.db key of the session being "
        "written by the messaging gateway)."
    )

async def main():
    c = probe.Client('a1')
    await c.connect()
    r = await c.call("session.changes", {"session_id": SID, "since_message_id": 0})
    res = r.get("result") or {}
    cursor = res.get("last_id", 0)
    print(f"A1 watch start: session={SID} cursor={cursor} preloaded={len(res.get('messages', []))} rows")
    t_end = time.time() + DURATION
    lat = []
    while time.time() < t_end:
        await asyncio.sleep(2.5)
        r = await c.call("session.changes", {"session_id": SID, "since_message_id": cursor})
        res = r.get("result") or {}
        rows = res.get("messages", [])
        now = time.time()
        for m in rows:
            ts = m.get("timestamp")
            if isinstance(ts, (int, float)) and ts > 1e9:
                d = now - ts
                lat.append(d)
                print(f"  row id={m.get('id')} role={m.get('role')} commit→fetch={d:.2f}s")
        cursor = res.get("last_id", cursor)
    await c.close()
    if lat:
        print(f"\nA1 RESULT: {len(lat)} rows; max latency {max(lat):.2f}s; "
              f"median {sorted(lat)[len(lat)//2]:.2f}s; SLO ≤3s: "
              f"{'PASS' if max(lat) <= 3.0 else 'CHECK (relax to ≤4s per spec if 3<x≤4)'}")
    else:
        print("\nA1: no new rows observed in window — re-run while the session is active")

asyncio.run(main())
