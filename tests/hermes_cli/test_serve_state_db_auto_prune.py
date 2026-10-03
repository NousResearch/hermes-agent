"""Invariant: the ``hermes serve`` maintenance tick runs ``sessions.auto_prune`` for its store,
and defers to a live gateway that owns that store.

``sessions.auto_prune`` (on by default) only ran from the classic CLI's startup and the gateway's
housekeeping tick. ``hermes serve`` — the Desktop app's backend — runs neither, so on a
Desktop-only install nothing ever pruned ``state.db`` or removed the ``request_dump_*`` files
that are only deleted together with their session.

Real store, real ``config.yaml``, the real ticker loop and the real ``_check_gateway_running``
predicate (the gateway stand-in holds ``gateway.lock`` under a ``hermes gateway run`` command line,
as in ``test_web_server_auto_archive_gateway_lock.py``). Only the curator/skill-sync chore is
replaced, by the tick-completion signal.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from queue import Queue

import pytest

from hermes_cli import web_server_sessions as wss
from hermes_cli.profiles import _check_gateway_running

_HOLDER = (
    "import fcntl, sys, time\n"
    "handle = open(sys.argv[1], 'a+', encoding='utf-8')\n"
    "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
    "sys.stdout.write('locked\\n')\n"
    "sys.stdout.flush()\n"
    "time.sleep(300)\n"
)


@pytest.fixture
def serve_home(monkeypatch):
    """This serve process's home (the per-test sandbox), whose OWN config keeps 30 days: one ended
    session idle 45 days (past that, but inside the 90-day default) and one idle 5 days, each with
    a request dump."""
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    home = get_hermes_home()
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "sessions:\n  auto_prune: true\n  retention_days: 30\n", encoding="utf-8")

    db = SessionDB(db_path=home / "state.db")
    try:
        for sid, idle_days in (("expired", 45), ("recent", 5)):
            db.create_session(sid, "tui")
            db.end_session(sid, end_reason="done")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                             (time.time() - idle_days * 86400, sid))
            db._conn.commit()
            (home / "sessions" / f"request_dump_{sid}_001.json").write_text("{}", encoding="utf-8")
    finally:
        db.close()

    ticks: Queue = Queue()
    monkeypatch.setattr(wss, "_maybe_run_skill_maintenance", lambda started_at: ticks.put(None))
    return home, ticks


async def _one_tick(ticks: Queue) -> None:
    """Run the real serve ticker for exactly one pass."""
    task = asyncio.create_task(wss._auto_archive_ticker_loop(interval_s=3600, initial_delay_s=0))
    try:
        await asyncio.to_thread(ticks.get, True, 20)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _sessions_left(home: Path) -> set:
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        return {sid for sid in ("expired", "recent") if db.get_session(sid)}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_serve_tick_prunes_sessions_past_the_profiles_retention(serve_home):
    home, ticks = serve_home

    await _one_tick(ticks)

    assert _sessions_left(home) == {"recent"}, "serve tick did not prune past sessions.retention_days"
    assert not (home / "sessions" / "request_dump_expired_001.json").exists()
    assert (home / "sessions" / "request_dump_recent_001.json").exists()


@pytest.mark.asyncio
@pytest.mark.platforms("posix")  # POSIX flock holder
@pytest.mark.spawns_gateway_lookalike  # a flock-holding stub this test reaps by PID
async def test_serve_tick_defers_prune_to_a_live_gateway(serve_home, tmp_path):
    home, ticks = serve_home
    # argv0 basename `hermes` + `gateway run` is what gateway.status's process-identity check reads.
    entrypoint = tmp_path / "hermes"
    entrypoint.write_text(_HOLDER, encoding="utf-8")
    holder = subprocess.Popen(
        [sys.executable, str(entrypoint), str(home / "gateway.lock"), "gateway", "run"],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
    )
    stdout = holder.stdout
    assert stdout is not None
    try:
        assert stdout.readline().strip() == "locked"
        (home / "gateway.pid").write_text(str(holder.pid), encoding="utf-8")
        assert _check_gateway_running(home), "probe did not look like a live gateway"

        await _one_tick(ticks)

        assert _sessions_left(home) == {"expired", "recent"}, "serve pruned a store its gateway owns"
    finally:
        holder.terminate()  # by PID: the process this test spawned
        holder.wait(timeout=10)
        stdout.close()

    # Gateway gone: the serve tick is the only maintainer left and must prune.
    deadline = time.monotonic() + 5
    while _check_gateway_running(home) and time.monotonic() < deadline:
        time.sleep(0.05)

    await _one_tick(ticks)

    assert _sessions_left(home) == {"recent"}, "serve must prune once no gateway owns the profile"
