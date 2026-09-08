"""Behavioral contention contract for the shared state.db helper-writer primitive.

``hermes_state_common.state_db_begin_immediate`` is the ``BEGIN IMMEDIATE``
discipline wired into ``tools.async_delegation._transaction`` and
``gateway.delivery_ledger._transaction``. The contract this module proves is
that the *application* retry loop is genuinely exercised when a competing
writer holds the WAL write lock, that it succeeds once the hold is released,
and that it exhausts according to its bounded patience budget — without ever
changing journal mode (the primitive only does ``BEGIN IMMEDIATE`` /
``COMMIT`` / ``ROLLBACK``).

False-green pattern (the bug this style of test must never reintroduce): a
test that uses the helper's own ``sqlite3.connect(timeout=10)`` busy handler
to ride out a hold, then asserts only the END state. SQLite's deterministic
busy wait would satisfy the same end state on its own; the application-level
``BEGIN IMMEDIATE`` retry loop could be silently dead. Instead:

  1. The probe forces ``busy_timeout=0`` so SQLite's internal busy wait
     CANNOT satisfy contention — the first ``BEGIN IMMEDIATE`` raises
     ``SQLITE_BUSY`` immediately.
  2. The holder signals readiness ONLY after its own ``BEGIN IMMEDIATE``
     returned (no thread-scheduling race).
  3. A delegating counter proxy counts ``BEGIN IMMEDIATE`` invocations on
     the real call path through the primitive (the literal attempt count).
  4. The body call site is counted independently and must run exactly once.
  5. Final value is verified via a fresh verifier connection.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest


pytest.importorskip("hermes_state_common")

from hermes_state_common import (  # noqa: E402
    state_db_begin_immediate,
)


@pytest.fixture
def tmp_db_path(tmp_path):
    """Throwaway DB path; callers create their own connections."""
    return tmp_path / "test_state.db"


def _open_probe(path, *, busy_timeout_s: float = 0.0):
    """Open a probe connection with a known busy timeout.

    ``busy_timeout_s=0`` (default) disables SQLite's internal busy wait so
    the application retry loop is the ONLY thing standing between the probe
    and ``SQLITE_BUSY``.  Higher values are exposed for negative controls.
    """
    return sqlite3.connect(str(path), timeout=busy_timeout_s, isolation_level=None)


def _seed_wal(path):
    """Create a real DB with WAL journal mode and one row."""
    seed = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    try:
        seed.execute("CREATE TABLE IF NOT EXISTS probe (v INTEGER)")
        seed.execute("INSERT OR IGNORE INTO probe (v) VALUES (0)")
        seed.execute("PRAGMA journal_mode=WAL")
    finally:
        seed.close()


class _BeginCounter:
    """Count BEGIN IMMEDIATE / COMMIT / ROLLBACK on a probe connection.

    ``sqlite3.Connection`` is a static C type — its ``execute`` cannot be
    monkey-patched, so we wrap it in a delegating proxy that intercepts the
    literal SQL before delegating to the real connection.  This is a
    behavioral witness: it sees the real statements the primitive issues.
    """

    def __init__(self, conn):
        self._real = conn
        self.begin_attempts = 0
        self.commit_attempts = 0
        self.rollback_attempts = 0

    def execute(self, sql, *args, **kwargs):
        text = sql.strip().upper() if isinstance(sql, str) else ""
        if text == "BEGIN IMMEDIATE":
            self.begin_attempts += 1
        elif text == "COMMIT":
            self.commit_attempts += 1
        elif text == "ROLLBACK":
            self.rollback_attempts += 1
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _hold_write_lock_until(db_path, conn_box, ready, release) -> None:
    """Holder thread: acquire the WAL write lock, signal ready, then hold."""
    conn = sqlite3.connect(
        str(db_path), timeout=10, isolation_level=None, check_same_thread=False,
    )
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn_box.append(conn)
        ready.set()
        release.wait()
    finally:
        try:
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
        conn.close()


def test_uncontended_begin_immediate_success(tmp_db_path):
    """An uncontended BEGIN IMMEDIATE acquires, runs the body, commits once."""
    _seed_wal(tmp_db_path)
    real = _open_probe(tmp_db_path)
    probe = _BeginCounter(real)
    try:
        body_call_count = 0

        def body():
            nonlocal body_call_count
            body_call_count += 1
            probe.execute("UPDATE probe SET v = ? WHERE v = 0", (7,))

        with state_db_begin_immediate(probe):
            body()

        assert probe.begin_attempts == 1
        assert body_call_count == 1
        assert probe.commit_attempts == 1
        assert probe.rollback_attempts == 0
    finally:
        probe.close()

    verifier = sqlite3.connect(str(tmp_db_path), timeout=10, isolation_level=None)
    try:
        row = verifier.execute("SELECT v FROM probe").fetchone()
    finally:
        verifier.close()
    assert row is not None and row[0] == 7


def test_application_retry_loop_genuinely_exercised(tmp_db_path):
    """The primitive MUST retry BEGIN IMMEDIATE while the lock is held."""
    _seed_wal(tmp_db_path)

    holder_conns = []
    holder_ready = threading.Event()
    holder_release = threading.Event()

    holder = threading.Thread(
        target=_hold_write_lock_until,
        args=(tmp_db_path, holder_conns, holder_ready, holder_release),
        daemon=True,
    )
    holder.start()

    assert holder_ready.wait(timeout=2.0), "holder never acquired the lock"
    assert len(holder_conns) == 1

    hold_seconds = 0.3
    holder_release.clear()

    def _release_after():
        time.sleep(hold_seconds)
        holder_release.set()

    release_thread = threading.Thread(target=_release_after, daemon=True)
    release_thread.start()

    real_probe = _open_probe(tmp_db_path, busy_timeout_s=0.0)
    probe = _BeginCounter(real_probe)
    try:
        body_call_count = 0

        def body():
            nonlocal body_call_count
            body_call_count += 1
            probe.execute("UPDATE probe SET v = ? WHERE v = 0", (42,))

        with state_db_begin_immediate(probe):
            body()

        # 1. Application retry loop fired (BEGIN IMMEDIATE > 1 attempt).
        assert probe.begin_attempts > 1, (
            f"expected BEGIN IMMEDIATE retried >1 times, got "
            f"begin_attempts={probe.begin_attempts}"
        )
        # 2. Body ran exactly once (no replay on retry).
        assert body_call_count == 1
        # 3. Exactly one successful COMMIT.
        assert probe.commit_attempts == 1
        # 4. No rollback on the success path.
        assert probe.rollback_attempts == 0
    finally:
        probe.close()

    holder.join(timeout=2.0)
    release_thread.join(timeout=2.0)

    verifier = sqlite3.connect(str(tmp_db_path), timeout=10, isolation_level=None)
    try:
        row = verifier.execute("SELECT v FROM probe").fetchone()
    finally:
        verifier.close()
    assert row is not None and row[0] == 42


def test_retry_exhaustion_raises_and_rolls_back(monkeypatch, tmp_db_path):
    """When persistence budget is exhausted, the primitive re-raises the
    lock error and does not leave a dangling transaction.

    We shrink the module-level patience constant so the test does not wait
    the real 20 s budget.
    """
    import hermes_state_common as hsc

    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_PATIENCE_S", 0.25)
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_RETRY_MIN_S", 0.005)
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_RETRY_MAX_S", 0.010)
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_RETRY_SLOW_AFTER_S", 0.05)
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_RETRY_SLOW_MIN_S", 0.005)
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_RETRY_SLOW_MAX_S", 0.010)

    _seed_wal(tmp_db_path)

    holder_conns = []
    holder_ready = threading.Event()
    # Never release — the probe must exhaust its (short) patience budget.
    holder_release = threading.Event()

    holder = threading.Thread(
        target=_hold_write_lock_until,
        args=(tmp_db_path, holder_conns, holder_ready, holder_release),
        daemon=True,
    )
    holder.start()

    assert holder_ready.wait(timeout=2.0), "holder never acquired the lock"

    probe = _open_probe(tmp_db_path, busy_timeout_s=0.0)
    try:
        body_call_count = 0

        def body():
            nonlocal body_call_count
            body_call_count += 1
            probe.execute("UPDATE probe SET v = ? WHERE v = 0", (99,))

        with pytest.raises(sqlite3.OperationalError) as excinfo:
            with state_db_begin_immediate(probe):
                body()

        msg = str(excinfo.value).lower()
        assert "locked" in msg or "busy" in msg
        assert body_call_count == 0, "body must never run on retry exhaustion"
    finally:
        probe.close()

    holder_release.set()
    holder.join(timeout=2.0)

    # The value must be untouched by the exhausted write.
    verifier = sqlite3.connect(str(tmp_db_path), timeout=10, isolation_level=None)
    try:
        row = verifier.execute("SELECT v FROM probe").fetchone()
    finally:
        verifier.close()
    assert row is not None and row[0] == 0


def test_non_retryable_operational_error_propagates_immediately(tmp_db_path, monkeypatch):
    """A non-lock ``OperationalError`` from the BODY must propagate without
    jitter retry. The primitive may issue a best-effort ROLLBACK on the way
    out (it catches ``BaseException`` in the body slot) — that is part of
    failure cleanup, NOT a retry — but BEGIN IMMEDIATE itself is never
    retried for a non-lock error.
    """
    import hermes_state_common as hsc

    # Shrink patience to ensure the test fails fast if retry sneaks in.
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_PATIENCE_S", 5.0)

    _seed_wal(tmp_db_path)

    real = _open_probe(tmp_db_path)
    probe = _BeginCounter(real)
    try:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            with state_db_begin_immediate(probe):
                # Body raises a non-retryable error after BEGIN succeeded.
                probe.execute("SELECT * FROM does_not_exist")
        # BEGIN was issued exactly once (no jitter retry on a non-lock error).
        assert probe.begin_attempts == 1
        msg = str(excinfo.value).lower()
        assert "no such table" in msg
    finally:
        probe.close()


def test_primitive_never_changes_journal_mode(tmp_db_path):
    """state_db_begin_immediate must not touch journal mode — it only issues
    BEGIN IMMEDIATE / COMMIT / ROLLBACK.  A DELETE-mode DB stays DELETE after
    the primitive runs.
    """
    # Seed in DELETE mode (explicitly, matching the authoritative contract's
    # "seed journal_mode=DELETE while config requests WAL" scenario).
    seed = sqlite3.connect(str(tmp_db_path), timeout=10, isolation_level=None)
    try:
        seed.execute("CREATE TABLE IF NOT EXISTS probe (v INTEGER)")
        seed.execute("INSERT OR IGNORE INTO probe (v) VALUES (0)")
        assert str(seed.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower() == "delete"
    finally:
        seed.close()

    probe = sqlite3.connect(str(tmp_db_path), timeout=10, isolation_level=None)
    try:
        with state_db_begin_immediate(probe):
            probe.execute("UPDATE probe SET v = 5 WHERE v = 0")
        mode = str(probe.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        assert mode == "delete", f"primitive changed journal mode to {mode!r}"
    finally:
        probe.close()


def test_no_more_rows_classified_as_retryable(tmp_db_path, monkeypatch):
    """The ``no more rows available`` transient must be treated as retryable
    by the primitive — same class as ``locked``/``busy`` — so the WAL-contended
    append case doesn't surface to callers.
    """
    import hermes_state_common as hsc

    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_PATIENCE_S", 0.30)
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_RETRY_MIN_S", 0.005)
    monkeypatch.setattr(hsc, "_STATE_DB_WRITE_RETRY_MAX_S", 0.010)

    _seed_wal(tmp_db_path)

    real = _open_probe(tmp_db_path)
    counter = _BeginCounter(real)
    base_execute = counter.execute
    injected_attempts = {"n": 0}

    def injected_execute(sql, *args, **kwargs):
        text = sql.strip().upper() if isinstance(sql, str) else ""
        if text == "BEGIN IMMEDIATE":
            injected_attempts["n"] += 1
            if injected_attempts["n"] == 1:
                raise sqlite3.InterfaceError("no more rows available")
        return base_execute(sql, *args, **kwargs)

    counter.execute = injected_execute
    try:
        with state_db_begin_immediate(counter):
            counter.execute("UPDATE probe SET v = ? WHERE v = 0", (11,))
        # The injection fires on the first BEGIN IMMEDIATE (raising before
        # base_execute runs, so the counter does not see the first attempt).
        # The primitive treats the InterfaceError as retryable, the next
        # BEGIN succeeds, and the body sees exactly one BEGIN. The second
        # BEGIN did reach base_execute, so counter.begin_attempts == 1.
        # The total number of BEGIN attempts (success + transient) is
        # ``injected_attempts["n"]`` and must be >= 2.
        assert injected_attempts["n"] >= 2
        assert counter.begin_attempts == 1
        assert counter.commit_attempts == 1
    finally:
        real.close()

    verifier = sqlite3.connect(str(tmp_db_path), timeout=10, isolation_level=None)
    try:
        row = verifier.execute("SELECT v FROM probe").fetchone()
    finally:
        verifier.close()
    assert row is not None and row[0] == 11
