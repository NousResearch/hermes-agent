"""Retry behaviour of ``adopt_claimed_execution`` under transient lock contention.

Regression for #135487: ``cron/executions.db`` queues ``synchronous=FULL``
commits during batch-fire ticks, so an external restart-safe worker whose adopt
hit the 5 s busy timeout died with ``sqlite3.OperationalError: database is
locked`` and the whole dispatch was lost before the acknowledgement was written.

These tests pin:

1. Transient ``database is locked`` errors are retried with backoff and the
   adoption succeeds instead of killing the dispatch.
2. ``rowcount == 0`` (another winner, or the claim is gone) returns ``None``
   without retrying.
3. Non-contention errors (e.g. ``no such table``) surface immediately.
4. The budget is the handoff grace still left when adoption starts, measured from
   ``handoff_started_at`` — not a fixed attempt count (#145590 review finding).
5. A late attempt stamps its own ``started_at`` instead of reusing one taken
   before the backoff.
6. Real SQLite writer contention (a second connection holding ``BEGIN IMMEDIATE``)
   is survived, not only the faked control flow.
"""

import logging
import sqlite3
import threading
import time
import types

import pytest

import cron.executions as ex


class _FakeCursor:
    def __init__(self, rowcount, row=None):
        self.rowcount = rowcount
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, script):
        self._script = list(script) if script is not None else []
        self._fetch = None

    def execute(self, *a, **k):
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def fetchone(self):
        return self._fetch

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _CallLog(list):
    """The connection list the tests unpack, plus the statements each attempt ran."""

    def __init__(self):
        super().__init__()
        self.calls = []


def _patch(monkeypatch, script, fetch_row=(("id", "x"),), anchor_age=0.0):
    conns = _CallLog()
    # One stack shared by every connection: each retry opens a fresh connection.
    queue = list(script)

    class _SharedConn(_FakeConn):
        def __init__(self):
            super().__init__(None)
            self._script = queue  # shared, not a copy

        def execute(self, *a, **k):
            conns.calls.append(a)
            return super().execute(*a, **k)

    def fake_transaction():
        conns.append(_SharedConn())
        conns[-1]._fetch = fetch_row
        return conns[-1]

    class _Anchor:
        """The read ``_adopt_retry_deadlines`` does: the claim's handoff stamp."""

        def execute(self, *a, **k):
            return types.SimpleNamespace(fetchone=lambda: (time.time() - anchor_age,))

        def close(self):
            pass

    monkeypatch.setattr(ex, "_transaction", fake_transaction)
    monkeypatch.setattr(ex, "_connect", lambda: _Anchor())
    monkeypatch.setattr(ex, "_emit_execution_state", lambda *a, **k: None)
    sleeps = []
    monkeypatch.setattr(ex.time, "sleep", lambda s: sleeps.append(s))
    return conns, sleeps


def test_adopt_retries_transient_lock_and_succeeds(monkeypatch):
    locked = sqlite3.OperationalError("database is locked")
    script = [locked, locked, _FakeCursor(1), _FakeCursor(0, {"id": "x"})]
    conns, sleeps = _patch(monkeypatch, script)
    rec = ex.adopt_claimed_execution("deadbeef")
    assert rec is not None, "adopt must still succeed after transient lock contention"
    assert len(conns) == 3, "3 attempts (2x locked, 1x success)"
    assert sleeps == [0.5, 1.0], "backoff 0.5 s then 1.0 s"


def test_adopt_lost_race_returns_none_without_retry(monkeypatch):
    script = [_FakeCursor(0), _FakeCursor(0, {"id": "x"})]
    conns, sleeps = _patch(monkeypatch, script)
    assert ex.adopt_claimed_execution("deadbeef") is None
    assert len(conns) == 1, "rowcount 0 = another winner, no retry"
    assert sleeps == []


def test_adopt_non_contention_error_raises_immediately(monkeypatch):
    broken = sqlite3.OperationalError("no such table: executions")
    script = [broken]
    conns, _sleeps = _patch(monkeypatch, script)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        ex.adopt_claimed_execution("deadbeef")
    assert len(conns) == 1, "schema errors are not retried"


def test_retry_budget_is_the_grace_left_when_adoption_starts(monkeypatch):
    """A fresh claim has (grace - margin) of silent retrying ahead of it, no more."""
    _patch(monkeypatch, [])
    now = time.monotonic()
    grace, hard = ex._adopt_retry_deadlines("deadbeef")
    expected = ex.HANDOFF_ADOPTION_GRACE_SECONDS - ex._ADOPT_RETRY_MARGIN_SECONDS
    assert abs((grace - now) - expected) < 1.0, (grace - now, expected)
    assert abs((hard - grace) - ex._ADOPT_RETRY_OVERRUN_SECONDS) < 0.5


def test_spent_grace_promises_no_further_silent_window(monkeypatch):
    """A worker that starts after the window still gets only the bounded overrun."""
    _patch(monkeypatch, [], anchor_age=ex.HANDOFF_ADOPTION_GRACE_SECONDS)
    grace, hard = ex._adopt_retry_deadlines("deadbeef")
    assert grace <= time.monotonic() + 0.1, "a spent grace must not promise more window"
    assert abs((hard - grace) - ex._ADOPT_RETRY_OVERRUN_SECONDS) < 0.5


def test_adopt_gives_up_when_the_window_cannot_afford_another_attempt(monkeypatch):
    locked = sqlite3.OperationalError("database is locked")
    conns, sleeps = _patch(monkeypatch, [locked])
    monkeypatch.setattr(
        ex, "_adopt_retry_deadlines", lambda _id: (time.monotonic(), time.monotonic())
    )
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        ex.adopt_claimed_execution("deadbeef")
    assert len(conns) == 1, "no window left = no further attempt"
    assert sleeps == []


def test_adopt_overruns_a_spent_grace_loudly(monkeypatch, caplog):
    """Stopping exactly at the grace boundary would hand the waiter a dead owner, which
    it terminalizes as ``unknown`` and the tick is lost — so contention is retried to the
    hard cap, reported once."""
    locked = sqlite3.OperationalError("database is locked")
    script = [locked] * ex._ADOPT_RETRY_ATTEMPTS_MAX
    conns, sleeps = _patch(monkeypatch, script)
    monkeypatch.setattr(
        ex,
        "_adopt_retry_deadlines",
        lambda _id: (time.monotonic() - 1.0, time.monotonic() + 60.0),
    )
    with caplog.at_level(logging.WARNING, logger="cron.executions"):
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            ex.adopt_claimed_execution("deadbeef")
    assert len(conns) == ex._ADOPT_RETRY_ATTEMPTS_MAX, "bounded by the attempt backstop"
    assert sleeps[0] == 0.5 and max(sleeps) == 2.0, "backoff capped at 2.0 s"
    warned = [r for r in caplog.records if "past the handoff grace" in r.getMessage()]
    assert len(warned) == 1, "the overrun is reported once, not per attempt"


def test_started_at_is_stamped_per_attempt(monkeypatch):
    """A late attempt must not record a start seconds older than the run it describes."""
    locked = sqlite3.OperationalError("database is locked")
    ticks = iter([f"2026-11-14T00:00:0{i}" for i in range(6)])
    monkeypatch.setattr(
        ex, "_hermes_now", lambda: types.SimpleNamespace(isoformat=lambda: next(ticks))
    )
    script = [locked, locked, _FakeCursor(1), _FakeCursor(0, {"id": "x"})]
    conns, _sleeps = _patch(monkeypatch, script)
    ex.adopt_claimed_execution("deadbeef")
    stamps = [
        params[3]
        for stmt, params in ((call[0], call[1]) for call in conns.calls if len(call) > 1)
        if stmt.lstrip().upper().startswith("UPDATE")
    ]
    assert stamps == ["2026-11-14T00:00:00", "2026-11-14T00:00:01", "2026-11-14T00:00:02"]


def test_lock_classifier(monkeypatch):
    assert ex._is_lock_contention(sqlite3.OperationalError("database is locked"))
    assert ex._is_lock_contention(sqlite3.OperationalError("database is busy"))
    assert not ex._is_lock_contention(
        sqlite3.OperationalError("database disk image is malformed")
    )
    assert not ex._is_lock_contention(ValueError("database is locked"))


def test_real_wal_contention_is_survived(tmp_path, monkeypatch):
    """The fakes above pin control flow only. This drives real writer contention: a
    second connection holds the write lock through one busy timeout, and the adoption
    must still take the claim once the lock clears."""
    from hermes_cli.sqlite_util import open_db

    db = tmp_path / "executions.db"
    monkeypatch.setattr(ex, "EXECUTIONS_FILE", db)

    def connect(**kw):
        return open_db(
            db,
            db_label="cron/executions.db",
            synchronous_full=True,
            busy_timeout_ms=200,  # keeps the test fast; the knob itself is what is honoured
            initialize=ex._initialize_schema,
            **kw,
        )

    seed = connect()
    seed.execute(
        "INSERT INTO executions (id, job_id, source, process_id, pid, status,"
        " handoff_pending, handoff_started_at, claimed_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("exec-1", "job-1", "cli", "other-process", 111, "claimed", 1, time.time(),
         "2026-11-14T00:00:00"),
    )
    seed.commit()
    seed.close()

    monkeypatch.setattr(ex, "_connect", connect)
    monkeypatch.setattr(ex, "_emit_execution_state", lambda *a, **k: None)

    blocker = connect(check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")

    def release():
        # The hold duration IS the behaviour under test: it must outlast one
        # busy_timeout (200 ms) so the adopt really sees ``database is locked``.
        time.sleep(0.35)  # health: allow HX013 -- contention window under test
        blocker.rollback()

    holder = threading.Thread(target=release)
    holder.start()
    try:
        record = ex.adopt_claimed_execution("exec-1")
    finally:
        holder.join()
        blocker.close()

    assert record is not None, "contention that clears inside the window must not lose the run"
    assert record["status"] == "running"
