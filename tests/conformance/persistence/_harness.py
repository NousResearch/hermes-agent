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
    # Prepend, don't clobber: CI images may rely on an inherited PYTHONPATH
    # for dependencies — losing it would fail child imports while the parent
    # collects fine, surfacing only as an opaque wait_for deadline.
    inherited = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{inherited}" if inherited else str(REPO_ROOT)
    )
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


def wait_for(
    predicate,
    *,
    deadline: float = CHILD_DEADLINE,
    what: str = "condition",
    child: "subprocess.Popen | None" = None,
) -> None:
    """Poll ``predicate`` until true or fail loudly at the deadline.

    When ``child`` is given and it exits before the predicate turns true,
    fail IMMEDIATELY with its captured stderr — a crashed writer must be an
    instant diagnostic, not a 60s opaque deadline.
    """
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return
        if child is not None and child.poll() is not None:
            err = b""
            if child.stderr is not None:
                err = child.stderr.read() or b""
            raise AssertionError(
                f"child exited early (rc={child.returncode}) while waiting "
                f"for {what}; stderr:\n{err.decode(errors='replace')[-2000:]}"
            )
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


def integrity_ok(db_path: Path) -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return bool(row) and str(row[0]).lower() == "ok"
    finally:
        conn.close()


def make_hermes_home(base: Path, journal_mode: str) -> Path:
    """Create an isolated HERMES_HOME whose config pins ``database.journal_mode``.

    The journal-mode matrix legs must steer the CHILD's own resolver:
    pre-seeding the DB file alone is not enough, because ``SessionDB.__init__``
    runs ``apply_wal_with_fallback()`` which upgrades a non-WAL file to WAL
    whenever the configured mode (default ``wal``) says so — on healthy
    SQLite the DELETE leg would silently run in WAL.
    """
    home = base / f"hermes-home-{journal_mode}"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"database:\n  journal_mode: {journal_mode}\n", encoding="utf-8"
    )
    return home


def effective_mode_or_skip(db_path: Path, requested_mode: str | None) -> str:
    """Record the on-disk journal mode; skip if a requested leg wasn't honored.

    A leg that ran in a different mode than advertised must never count as
    green evidence for the advertised mode.
    """
    import pytest

    mode = on_disk_journal_mode(db_path)
    if requested_mode is not None and mode.upper() != requested_mode.upper():
        pytest.skip(
            f"journal_mode={requested_mode} not honored by this environment "
            f"(effective {mode}) — matrix leg not probeable here"
        )
    return mode
