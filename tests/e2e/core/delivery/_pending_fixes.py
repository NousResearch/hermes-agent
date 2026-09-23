"""Merge-order-safe expected failures for live gaps whose fix is an open PR.

A plain ``xfail(strict=True)`` turns main red the moment its fix merges (XPASS), and a
non-strict one guards nothing. Instead, each gap here has a PROBE: a few lines that reproduce
the defect's mechanism on the tree under test, in a throwaway interpreter with its own
``HOME``/``HERMES_HOME`` (no state leaks into the suite process). ``expect_gap`` applies a
strict xfail only while the probe still reproduces the defect; once the fix is in the tree the
cell runs as a plain test, so it must pass. Whichever lands first, suite or fix, main stays
green, and a probe that disagrees with the end-to-end cell still fails loudly (XPASS, or a
real failure) instead of hiding.

When a fix has landed, delete its entry and every ``expect_gap`` call naming it.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

_PRELUDE = "import json, os, sys\nsys.path.insert(0, os.getcwd())\n"

# PR -> (extra env, script). A script prints ``open`` while the defect reproduces, ``fixed`` once
# it no longer does; anything else (including a crash) fails the cell that asked.
PROBES: Dict[int, tuple] = {
    # The due gate compared same-zone wall clocks: 01:00 EST (fold=1) looked due at 01:01 EDT.
    120314: ({"HERMES_TIMEZONE": "America/New_York", "TZ": "UTC"}, r'''
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from cron import jobs
ny = ZoneInfo("America/New_York")
now = datetime(2026, 11, 1, 5, 1, tzinfo=timezone.utc).astimezone(ny)       # 01:01 EDT
scheduled = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc).astimezone(ny)  # 01:00 EST, 59 min later
jobs._hermes_now = lambda: now
jobs.save_jobs([{"id": "p", "name": "p", "prompt": "p", "schedule": {"kind": "interval", "minutes": 60},
                 "next_run_at": scheduled.isoformat(), "last_run_at": None, "enabled": True,
                 "state": "scheduled", "repeat": {"times": None, "completed": 0}, "deliver": "local"}])
print("open" if jobs.get_due_jobs() else "fixed")
'''),
    # No timezone configured: the next cron occurrence kept the base time's fixed UTC offset, so a
    # 09:00 job in a DST process zone fired at 10:00 local the day after spring-forward.
    119970: ({"TZ": "America/New_York"}, r'''
from datetime import datetime
from cron import jobs
# what an unconfigured clock returns: the process zone's offset of the moment, as a fixed offset
jobs._hermes_now = lambda: datetime.fromisoformat("2026-03-07T09:00:30-05:00")
nxt = jobs.compute_next_run({"kind": "cron", "expr": "0 9 * * *"})
print("fixed" if nxt == "2026-03-08T09:00:00-04:00" else "open")
'''),
    # A failed first stream send disabled edits but left no message id, so the next tick sent a
    # second first send: an uneditable partial preview stayed visible next to the final reply.
    120315: ({}, r'''
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

async def main():
    delivered, results = [], iter([SimpleNamespace(success=False, error="timeout")])
    async def send(**kw):
        r = next(results, None) or SimpleNamespace(success=True, message_id=f"m{len(delivered)}")
        if r.success:
            delivered.append(kw["content"])
        return r
    adapter = MagicMock()
    adapter.send = AsyncMock(side_effect=send)
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.MAX_MESSAGE_LENGTH = 4096
    c = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(edit_interval=0.01,
                                                                    buffer_threshold=5, cursor=""))
    c.on_delta("preview never landed ")
    task = asyncio.create_task(c.run())
    await asyncio.sleep(0.08)
    c.on_delta("and more streamed text ")
    await asyncio.sleep(0.08)
    c.on_delta("then the end.")
    c.finish()
    await asyncio.wait_for(task, timeout=10)
    return delivered

print("fixed" if asyncio.run(main()) == ["preview never landed and more streamed text then the end."]
      else "open")
'''),
    # Unclean startup ran the 120 s recency sweep: every recently active session was marked
    # resume_pending and auto-resumed, i.e. answered a second time.
    120377: ({}, r'''
import inspect
from gateway.run import GatewayRunner
src = inspect.getsource(GatewayRunner._recover_unclean_sessions)
print("open" if "suspend_recently_active" in src else "fixed")
'''),
    # Inbound de-duplication lived only on the adapter instance; the reconnect watcher builds a
    # fresh adapter without it, so a replay after a reconnect was processed again.
    120444: ({}, r'''
import inspect
from gateway.run import GatewayRunner
src = inspect.getsource(GatewayRunner._reconnect_failed_platform)
print("fixed" if "dedup" in src.lower() else "open")
'''),
    # The boot sweep claimed a 'pending' row without moving it to 'attempting', so a boot killed
    # inside that plain resend left 'pending' behind and the next boot resent it UNMARKED.
    120450: ({}, r'''
import sqlite3
from gateway import delivery_ledger as L
L.record_obligation(obligation_id="p", session_key="k", platform="telegram", chat_id="1",
                    thread_id=None, content="x")
with L._transaction() as conn:
    conn.execute("UPDATE delivery_obligations SET owner_pid=NULL, owner_started_at=NULL")
assert [r["obligation_id"] for r in L.sweep_recoverable()] == ["p"]
with L._transaction() as conn:
    state = conn.execute("SELECT state FROM delivery_obligations").fetchone()[0]
print("fixed" if state == "attempting" else "open")
'''),
}


@functools.lru_cache(maxsize=None)
def gap_open(pr: int) -> bool:
    """True while the defect PR ``pr`` fixes still reproduces on this tree."""
    extra_env, script = PROBES[pr]
    with tempfile.TemporaryDirectory(prefix=f"gap-{pr}-") as tmp:
        home = Path(tmp) / "home"
        (home / ".hermes").mkdir(parents=True)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PYTEST_", "HERMES_")) and not k.endswith("_API_KEY")}
        env.update({"HOME": str(home), "HERMES_HOME": str(home / ".hermes"),
                    "PYTHONPATH": str(REPO_ROOT), **extra_env})
        proc = subprocess.run([sys.executable, "-c", _PRELUDE + script], cwd=str(REPO_ROOT), env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    verdict = proc.stdout.strip().splitlines()[-1:] if proc.returncode == 0 else []
    if verdict not in (["open"], ["fixed"]):
        raise AssertionError(f"probe for #{pr} is broken (rc={proc.returncode}): "
                             f"{json.dumps(proc.stdout[-800:])} {proc.stderr[-2000:]}")
    return verdict == ["open"]


def expect_gap(request, pr: int, reason: str) -> None:
    """Strict xfail for this cell while #``pr``'s defect reproduces; a plain test once it doesn't."""
    assert f"#{pr}" in reason, f"reason for a #{pr} gap must name the PR: {reason!r}"
    if gap_open(pr):
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
