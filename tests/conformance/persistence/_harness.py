"""Shared harness for the crash/resume persistence conformance cells (#80921).

Method (per the spot-probes in the tracking issue): real ``SessionDB``
against an isolated temp database, real ``SIGKILL`` delivered to a separate
OS process mid-write, deterministic and LLM-free. Every wait has a hard
deadline so a wedged child can never hang the suite; coordination uses file
barriers, never bare sleeps.

Journal-mode policy mirrors the issue's caveat: cells run on the mode the
repo's own ``resolve_journal_mode()`` selects for this interpreter/filesystem
(recorded per cell), plus an explicit ``DELETE`` run; an explicit ``WAL`` run
is attempted and skipped when the resolver's downgrade gates trip (e.g. the
WAL-reset interpreter bug), so WAL semantics are probed exactly where they
are actually deployable.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Generous deadlines: xdist-loaded CI boxes stall; correctness never depends
# on these being tight, they only bound a hung child.
CHILD_DEADLINE = 60.0
POLL_INTERVAL = 0.02


def spawn_child(script_body: str, *, cwd: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    """Run ``script_body`` in a fresh interpreter with the repo importable."""
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = str(REPO_ROOT)
    child_env["PYTHONUNBUFFERED"] = "1"
    if env:
        child_env.update(env)
    return subprocess.Popen(
        [sys.executable, "-c", script_body],
        cwd=str(cwd or REPO_ROOT),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait_for(predicate, *, deadline: float = CHILD_DEADLINE, what: str = "condition") -> None:
    """Poll ``predicate`` until true or fail loudly at the deadline."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"deadline ({deadline}s) waiting for {what}")


def kill9_and_reap(proc: subprocess.Popen, *, deadline: float = CHILD_DEADLINE) -> None:
    """SIGKILL ``proc`` and reap it within ``deadline``."""
    try:
        proc.kill()  # SIGKILL on POSIX
    except ProcessLookupError:
        pass
    proc.wait(timeout=deadline)


def reap(proc: subprocess.Popen, *, deadline: float = CHILD_DEADLINE) -> tuple[int, str, str]:
    """Wait for a child to exit on its own; kill + fail if it doesn't."""
    try:
        out, err = proc.communicate(timeout=deadline)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate(timeout=10)
        raise AssertionError(
            f"child did not exit within {deadline}s; stderr:\n"
            f"{err.decode(errors='replace')[-2000:]}"
        )
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def on_disk_journal_mode(db_path: Path) -> str:
    """Record the journal mode actually in effect for a cell result."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else "unknown"
    finally:
        conn.close()


def force_journal_mode(db_path: Path, mode: str) -> str:
    """Set an explicit journal mode on a fresh DB; return the effective mode.

    SQLite may refuse the switch (downgrade gates, live readers); callers
    compare the returned mode and ``pytest.skip`` when the request didn't
    stick — mirroring the resolver's behavior instead of fighting it.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(f"PRAGMA journal_mode={mode}").fetchone()
        return str(row[0]) if row else "unknown"
    finally:
        conn.close()


def integrity_ok(db_path: Path) -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return bool(row) and str(row[0]).lower() == "ok"
    finally:
        conn.close()
