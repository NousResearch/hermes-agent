"""`hermes doctor` must bound its read-only state.db health probe (#72441).

`_db_opens_cleanly` runs `PRAGMA integrity_check`, which walks every page of
the file. On a large `state.db` that turned `hermes doctor` into an indefinite
stall right after it printed `state.db exists (N sessions)`.

Two properties are pinned here, and the second is the one that makes the fix
safe rather than merely fast:

1. The command returns on a bounded wall clock even when the probe blocks.
2. A probe deadline is **never** reported as corruption and never escalates to
   `repair_state_db_schema()`. `sqlite3.OperationalError("interrupted")` — what
   a progress-handler abort raises — is a `sqlite3.DatabaseError`, so the naive
   implementation returns "interrupted" as an unhealthy *reason*; under
   `hermes doctor --fix` that rewrites the database (its last strategy drops
   the whole `messages_fts%` schema and VACUUMs). A healthy-but-large state.db
   would be sent into destructive repair merely for being slow.
"""

import contextlib
import io
import sqlite3
import time
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import doctor as doctor_mod


# How long the fake probe blocks for in the elapsed-time regression. Large
# enough that "waited for the probe" and "did not wait for the probe" cannot be
# confused by CI jitter.
_BLOCK_SECONDS = 20.0


def _isolate_home(monkeypatch, home: Path) -> None:
    """Point doctor at a temp HERMES_HOME.

    ``run_doctor`` reads the module-level ``HERMES_HOME`` constant cached at
    import time, NOT the env var, so ``setenv`` alone leaves doctor probing the
    real ``~/.hermes`` — which on a dev machine is exactly the multi-minute
    ``PRAGMA integrity_check`` this module is about. Same idiom as
    ``tests/hermes_cli/test_doctor.py::TestGitHubTokenCheck._isolate_home``.
    """
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))


def _make_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    return home


def _make_state_db(path: Path, sessions: int = 2, messages: int = 0) -> Path:
    """Create a minimal, genuinely healthy state.db."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, started_at REAL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                content TEXT, timestamp REAL
            );
            """
        )
        conn.executemany(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            [(f"s{i}", "test", 0.0) for i in range(sessions)],
        )
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            [("s0", "user", "x" * 200, 0.0) for _ in range(messages)],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _run_doctor(**kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False, ack=None, **kwargs))
    return buf.getvalue()


def _shrink_deadline(monkeypatch, deadline: float = 0.5, grace: float = 0.5) -> None:
    monkeypatch.setattr(doctor_mod, "_STATE_DB_PROBE_DEADLINE_S", deadline)
    monkeypatch.setattr(doctor_mod, "_STATE_DB_PROBE_ABANDON_GRACE_S", grace)


def test_doctor_returns_without_waiting_for_a_blocked_probe(monkeypatch, tmp_path):
    """The elapsed-time regression: doctor must not wait out a wedged probe.

    Fails on `origin/main` (no bound at all) and on the thread-timeout-only
    shape, where leaving a `with ThreadPoolExecutor(...)` block calls
    `shutdown(wait=True)` and blocks until the probe finishes anyway.
    """
    home = _make_home(tmp_path)
    _isolate_home(monkeypatch, home)
    _make_state_db(home / "state.db")

    import hermes_state

    started = []

    def _blocking_probe(db_path, *, deadline_seconds=None):
        started.append(time.monotonic())
        time.sleep(_BLOCK_SECONDS)
        return None

    monkeypatch.setattr(hermes_state, "_db_opens_cleanly", _blocking_probe)
    _shrink_deadline(monkeypatch)

    start = time.monotonic()
    out = _run_doctor()
    elapsed = time.monotonic() - start

    assert started, "the state.db health probe never ran"
    assert elapsed < _BLOCK_SECONDS / 2, (
        f"run_doctor waited {elapsed:.1f}s on a probe that blocks for "
        f"{_BLOCK_SECONDS:.0f}s — the deadline did not bound the command"
    )
    assert "timed out" in out
    # The worker is abandoned mid-flight, so the session count never comes
    # back — but the file demonstrably exists, and that row must not silently
    # vanish from the report.
    assert "state.db exists (session count not read" in out


def test_probe_timeout_is_reported_as_skipped_not_as_corruption(monkeypatch, tmp_path):
    home = _make_home(tmp_path)
    _isolate_home(monkeypatch, home)
    _make_state_db(home / "state.db", sessions=3)

    import hermes_state

    def _times_out(db_path, *, deadline_seconds=None):
        raise hermes_state.DBHealthProbeTimeout("deadline reached")

    monkeypatch.setattr(hermes_state, "_db_opens_cleanly", _times_out)
    _shrink_deadline(monkeypatch)

    out = _run_doctor()

    assert "state.db exists (3 sessions)" in out
    assert "timed out" in out
    assert "not modified" in out
    # The corruption verdict must not be printed for a timeout.
    assert "FTS index may be corrupt" not in out


def test_probe_timeout_does_not_trigger_repair_under_fix(monkeypatch, tmp_path):
    """The data-safety invariant: a deadline never escalates to a rewrite."""
    home = _make_home(tmp_path)
    _isolate_home(monkeypatch, home)
    _make_state_db(home / "state.db")

    import hermes_state

    def _times_out(db_path, *, deadline_seconds=None):
        raise hermes_state.DBHealthProbeTimeout("deadline reached")

    repairs = []

    def _spy_repair(db_path, **kwargs):
        repairs.append(db_path)
        return {"repaired": True, "strategy": "spy", "backup_path": None}

    monkeypatch.setattr(hermes_state, "_db_opens_cleanly", _times_out)
    monkeypatch.setattr(hermes_state, "repair_state_db_schema", _spy_repair)
    _shrink_deadline(monkeypatch)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=True, ack=None))

    assert repairs == [], (
        "a health-probe timeout escalated to repair_state_db_schema() — a "
        "healthy-but-large state.db would be rewritten for being slow"
    )
    assert "timed out" in buf.getvalue()


def test_progress_handler_cancels_a_real_integrity_check(tmp_path):
    """No mocks: the deadline is cancellable at the SQLite level.

    A thread timeout cannot stop `PRAGMA integrity_check`; a progress handler
    can. The same file probes healthy with no deadline, which is what proves
    the timeout is a cancellation and not a corruption finding.
    """
    from hermes_state import DBHealthProbeTimeout, _db_opens_cleanly

    db_path = _make_state_db(tmp_path / "state.db", sessions=1, messages=3000)

    start = time.monotonic()
    with pytest.raises(DBHealthProbeTimeout) as excinfo:
        _db_opens_cleanly(db_path, deadline_seconds=0)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    # Callers classify corruption by catching sqlite3.DatabaseError (and
    # OperationalError("interrupted") is one). The timeout signal must sit
    # outside that hierarchy or it gets swallowed as a corruption reason.
    assert not isinstance(excinfo.value, sqlite3.Error)

    assert _db_opens_cleanly(db_path) is None


def test_no_deadline_preserves_existing_behaviour(tmp_path):
    """The five `repair_state_db_schema` callers pass no deadline and must be
    byte-for-byte unaffected: healthy still returns None, damaged still returns
    the reason string rather than raising."""
    from hermes_state import _db_opens_cleanly

    healthy = _make_state_db(tmp_path / "state.db", sessions=1, messages=50)
    assert _db_opens_cleanly(healthy) is None

    broken = tmp_path / "broken.db"
    broken.write_bytes(b"SQLite format 3\x00" + b"\x00" * 512)
    reason = _db_opens_cleanly(broken)
    assert isinstance(reason, str) and reason


def test_doctor_reports_a_healthy_db_normally(monkeypatch, tmp_path):
    """Non-regression: the bound is invisible when the probe finishes."""
    home = _make_home(tmp_path)
    _isolate_home(monkeypatch, home)
    _make_state_db(home / "state.db", sessions=4, messages=100)
    _shrink_deadline(monkeypatch, deadline=30.0, grace=5.0)

    out = _run_doctor()

    assert "state.db exists (4 sessions)" in out
    assert "timed out" not in out
    assert "FTS index may be corrupt" not in out
