"""C4 compaction atomicity: kill -9 a real child process mid-compaction, then prove state.db is intact
and the session is resumable.

A seeded session is built in-process on a real store; then a CHILD process (real AIAgent against the
same state.db and the same fake provider) runs ``/compress``. The parent SIGKILLs it at two points:

* ``summarizer`` — while the summary request is in flight (lease held, nothing written yet);
* ``commit``     — inside the ``archive_and_compact`` write transaction, after the archive UPDATE and the
  compacted-row INSERTs but before COMMIT (the child is parked there by a test hook).

Invariants after the kill: ``PRAGMA integrity_check`` is ok; the live rows are exactly the pre-compaction
rows (no half-archived session, nothing lost, no half-inserted summary); the session counter matches;
and a fresh agent resumed from state.db can compact (the dead holder's lease is reclaimed) and continue,
sending exactly the persisted history.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.e2e.core.compaction._helpers import (
    GOOD_SUMMARY_TOKEN,
    Hang,
    Text,
    active_rows,
    assert_db_matches_history,
    assert_db_recoverable,
    db_rows,
    generate_transcript,
    good_summary,
)

REPO = Path(__file__).resolve().parents[4]
HIGH_THRESHOLD = 60_000
KILL_DEADLINE_SECONDS = 120.0

CHILD = r'''
import os, sys, time
from pathlib import Path
repo, db_path, session_id, base_url, point, sentinel = sys.argv[1:7]
sys.path.insert(0, repo)
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import compaction_write_seam
from tests.fixtures.storage_read import conversation

if point == "commit":
    # Park inside archive_and_compact's write transaction (after archive + inserts, before COMMIT).
    in_commit = {"on": False}
    module, compact_name, insert_name = compaction_write_seam()
    orig_compact = getattr(module, compact_name)
    orig_insert = getattr(module, insert_name)

    def compact(*a, **k):
        in_commit["on"] = True
        return orig_compact(*a, **k)

    def insert(holder, conn, *a, **k):
        out = orig_insert(holder, conn, *a, **k)
        if in_commit["on"]:
            with open(sentinel, "w") as fh:
                fh.write(str(os.getpid()))
            time.sleep(3600)
        return out

    setattr(module, compact_name, compact)
    setattr(module, insert_name, insert)

from run_agent import AIAgent
from agent.conversation_compression_manual import compress_now, parse_compress_args

db = open_persistence(profile_scope(home=Path(db_path).parent))
agent = AIAgent(provider="custom", base_url=base_url, api_key="sk-fake-e2e", model="fake-model",
                persistence=db, session_id=session_id, quiet_mode=True, enabled_toolsets=["file"],
                skip_context_files=True, skip_memory=True)
with open_persistence(profile_scope(home=Path(db_path).parent)) as store:
    history = conversation(store, session_id)
result = compress_now(agent, history, parse_compress_args(""))
print("CHILD-FINISHED", result.status, flush=True)
'''


def _child_env(home: Path) -> dict:
    # Probe hygiene: tmp HOME/HERMES_HOME, no provider keys. The in-test db guard detects pytest by
    # process ancestry and would read this tmp $HOME/.hermes as the production root; the documented
    # child-process escape hatch is safe because every path here lives under tmp_path.
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update(HOME=str(home), HERMES_HOME=str(home / ".hermes"), PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1",
               HERMES_STATE_DB_GUARD_BYPASS="1")
    return env


def _poll(pred, deadline: float, what: str, child: subprocess.Popen) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if pred():
            return
        if child.poll() is not None:
            out = child.stdout.read() if child.stdout else ""
            raise AssertionError(f"child exited ({child.returncode}) before {what}: {out[-2000:]}")
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _integrity(db_path: Path) -> str:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.parametrize("point", ("summarizer", "commit"))
def test_kill9_mid_compaction_leaves_state_consistent_and_resumable(make_scenario, provider, tmp_path, point):
    server, dispatch = provider
    home = tmp_path / "home"
    sc = make_scenario("good", threshold_tokens=HIGH_THRESHOLD, name="home/.hermes")
    specs = generate_transcript(41, 9, tmp_path / "work", kinds=("tools", "parallel", "chat"))
    for spec in specs[:6]:
        sc.run_turn(spec)
    sc.close()  # hand the session to the child: no open writer in this process
    rows_before = db_rows(sc.db_path, sc.session_id)

    in_flight = threading.Event()

    def summarizer(rec):
        sc.summary_calls.append(rec)
        if point == "summarizer":
            in_flight.set()
            return Hang(KILL_DEADLINE_SECONDS)
        return Text(good_summary(len(sc.summary_calls)))

    dispatch.aux = summarizer
    sentinel = tmp_path / "in-commit"
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(script), str(REPO), str(sc.db_path), sc.session_id, server.base_url, point, str(sentinel)],
        env=_child_env(home), cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        if point == "summarizer":
            _poll(in_flight.is_set, KILL_DEADLINE_SECONDS, "the summary request", child)
        else:
            _poll(sentinel.exists, KILL_DEADLINE_SECONDS, "the commit transaction", child)
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
    assert child.returncode == -signal.SIGKILL

    # (3) atomicity: the store is intact and exactly the pre-compaction state.
    assert _integrity(sc.db_path) == "ok"
    rows_after = db_rows(sc.db_path, sc.session_id)
    assert rows_after == rows_before, (
        f"kill -9 at {point} left a partial compaction: {len(rows_before)} rows before, {len(rows_after)} after; "
        f"live {len(active_rows(rows_before))} -> {len(active_rows(rows_after))}")
    conn = sqlite3.connect(f"file:{sc.db_path}?mode=ro", uri=True)
    try:
        count = conn.execute("SELECT message_count FROM sessions WHERE id = ?", (sc.session_id,)).fetchone()[0]
    finally:
        conn.close()
    assert count == len(active_rows(rows_after)), "session message_count disagrees with the live rows"

    # Resumable: a fresh agent reclaims the dead holder's lease, compacts, and continues.
    from tui_gateway.server import _compress_session_history

    sc.install()
    sc.resume_from_db()
    session = {"agent": sc.agent, "history": list(sc.history), "history_lock": threading.Lock(),
               "history_version": 1, "session_key": sc.session_id}
    removed, _ = _compress_session_history(session, "")
    sc.history = session["history"]
    assert removed > 0 and sc.committed_compactions() > 0, "the session could not be compacted after the crash"
    assert any(GOOD_SUMMARY_TOKEN in str(m.get("content")) for m in sc.history)
    rows = db_rows(sc.db_path, sc.session_id)
    assert_db_recoverable(rows, sc.sent_markers, "after crash + recompress")
    assert_db_matches_history(rows, sc.history, "after crash + recompress")
    for spec in specs[6:8]:
        sc.run_turn(spec)
    sc.resume_from_db()
    sc.run_turn(specs[8])
