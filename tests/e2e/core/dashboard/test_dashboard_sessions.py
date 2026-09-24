"""Dashboard session browser against a live ``state.db`` while another process writes to it.

Users hit this as "the Sessions page 500s / shows 'database is locked' while a chat is running" or
"the session count jumps backwards / double-counts" (the dashboard reads the same SQLite file the
agent, gateway and cron write). Harness: the real ``hermes dashboard`` process and a real writer
process (``hermes_state.SessionDB``, the agent's own persistence layer) appending sessions and
messages as fast as it can. Readers hammer the list, stats, detail and messages routes in parallel.

Invariants while the writer runs: every read is a 200 (never a lock/busy error), the list total
never goes backwards across sequential reads and never exceeds what the writer has committed, a
session detail's ``message_count`` equals the messages the messages route returns for it. After the
writer stops, the API total equals the raw row count in SQLite.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import pytest

from . import _helpers as H

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX dashboard process")

SEEDED = 40
WRITE_SECONDS = 8.0

# Writes sessions + messages in the agent's own persistence layer; prints the committed count after
# every session so the harness knows an upper bound at any instant.
_WRITER = r"""
import sys, time
from hermes_state import SessionDB
deadline = time.monotonic() + float(sys.argv[1])
db = SessionDB()
n = 0
while time.monotonic() < deadline:
    sid = f"live-{n:05d}"
    db.create_session(sid, source="cli", model="writer")
    for j in range(3):
        db.append_message(sid, "user" if j % 2 == 0 else "assistant", f"{sid} message {j}")
    n += 1
    print(n, flush=True)
db.close()
print("DONE", n, flush=True)
"""


class Writer:
    def __init__(self, sb: H.Sandbox, p: H.Profile) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(WRITE_SECONDS)], env=sb.env({"HERMES_HOME": str(p.home)}),
            cwd=str(sb.home), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        self.committed = 0
        self.done: int | None = None
        threading.Thread(target=self._pump, daemon=True, name="dash-writer").start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            parts = line.split()
            if parts[:1] == ["DONE"]:
                self.done = int(parts[1])
            elif parts and parts[0].isdigit():
                self.committed = int(parts[0])


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):
    sb = H.make_sandbox(tmp_path_factory.mktemp("dash-sessions"))
    p = sb.profiles["default"]
    H.seed_sessions(sb, p, "seed", SEEDED)
    d = H.Dashboard(sb, sb.root / "dashboard.log")
    try:
        yield sb, p, d
    finally:
        d.close()
        sb.stop()


def _read_round(d: H.Dashboard, committed: Callable[[], int]) -> list[str]:
    """One list read + detail/messages consistency for its newest rows. The upper bound is read
    AFTER the response: everything the writer reported by then, plus the one session in flight."""
    problems: list[str] = []
    r = d.request("GET", "/api/sessions", params={"limit": 100, "order": "recent"})
    upper_bound = committed() + 1
    if r.status_code != 200:
        return [f"GET /api/sessions -> {r.status_code} {r.text[:200]}"]
    body = r.json()
    total = body["total"]
    if not SEEDED <= total <= SEEDED + upper_bound:
        problems.append(f"total {total} outside [{SEEDED}, {SEEDED + upper_bound}] (writer committed {upper_bound})")
    for row in body["sessions"][:2]:
        sid = row["id"]
        det = d.request("GET", f"/api/sessions/{sid}")
        msgs = d.request("GET", f"/api/sessions/{sid}/messages")
        if det.status_code != 200 or msgs.status_code != 200:
            problems.append(f"{sid}: detail {det.status_code} / messages {msgs.status_code} {msgs.text[:200]}")
            continue
        listed = msgs.json().get("messages", [])
        # The writer appends to at most the newest session while we read: the detail may lag or
        # lead the messages read by the rows appended in between, never by more.
        if abs(det.json()["message_count"] - len(listed)) > 3:
            problems.append(f"{sid}: detail message_count {det.json()['message_count']} vs {len(listed)} messages")
    stats = d.request("GET", "/api/sessions/stats")
    if stats.status_code != 200:
        problems.append(f"GET /api/sessions/stats -> {stats.status_code} {stats.text[:200]}")
    return problems + [f"@total={total}"]


def test_session_routes_stay_consistent_under_a_concurrent_writer(env) -> None:
    sb, p, d = env
    w = Writer(sb, p)
    H.poll(lambda: w.committed > 0, 60, "the writer's first commit")
    totals: list[int] = []
    problems: list[str] = []

    def reader() -> list[str]:
        out: list[str] = []
        while w.done is None and w.proc.poll() is None:
            got = _read_round(d, lambda: w.committed)
            out += [g for g in got if not g.startswith("@")]
            totals.extend(int(g[7:]) for g in got if g.startswith("@total="))
        return out

    # Sequential totals must be monotonic: one dedicated reader keeps its own ordered series.
    series: list[int] = []

    def ordered() -> list[str]:
        out: list[str] = []
        while w.done is None and w.proc.poll() is None:
            r = d.request("GET", "/api/sessions", params={"limit": 1})
            if r.status_code != 200:
                out.append(f"ordered read -> {r.status_code} {r.text[:200]}")
                continue
            series.append(r.json()["total"])
        return out

    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(reader) for _ in range(4)] + [pool.submit(ordered)]
        for f in futs:
            problems += f.result(timeout=WRITE_SECONDS + 120)
    w.proc.wait(timeout=60)
    err = w.proc.stderr.read() if w.proc.stderr else ""
    assert w.proc.returncode == 0 and w.done, f"writer failed rc={w.proc.returncode}: {err[-2000:]}"
    assert w.done >= 20, f"writer committed only {w.done} sessions: the contention window never opened"
    assert len(totals) >= 10 and len(series) >= 10, f"too few reads overlapped the writer ({len(totals)}, {len(series)})"
    assert not problems, f"{len(problems)} inconsistent read(s) while writing:\n  " + "\n  ".join(dict.fromkeys(problems[:30]))
    backwards = [(a, b) for a, b in zip(series, series[1:]) if b < a]
    assert not backwards, f"session total went backwards between sequential reads: {backwards[:5]}"

    raw = H.db_rows(p.db, "SELECT COUNT(*) FROM sessions WHERE parent_session_id IS NULL")[0][0]
    final = d.ok("GET", "/api/sessions", params={"limit": 1})["total"]
    assert final == raw == SEEDED + w.done, f"api total {final}, sqlite {raw}, expected {SEEDED + w.done}"
    newest = d.ok("GET", f"/api/sessions/live-{w.done - 1:05d}/messages")["messages"]
    assert [m["content"] for m in newest] == [f"live-{w.done - 1:05d} message {j}" for j in range(3)]
    log = d.log_tail(200_000).lower()
    assert "database is locked" not in log and "traceback" not in log, d.log_tail()
