"""Resilience tests for the embedded kanban dispatcher in
``GatewayRunner._kanban_dispatcher_watcher``.

Background: issue #28464 (the legacy ``session_id`` migration trap, fixed
in #28781 / #28754) put a long-running gateway into a per-tick SQL-error
loop after an in-place upgrade. The dispatcher's exception handler
logged a full ``logger.exception`` traceback every tick — on a multi-
board install that surfaced as 20+ tracebacks per second per gateway,
saturating journald and pushing the gateway RSS to multi-GB over hours.

The schema bug itself is fixed at the migration layer (#28781). These
tests pin the *dispatcher-layer* defense added afterwards so the next
additive-column bug of the same shape self-heals instead of degrading
the whole gateway:

* ``no such column`` / ``no such table`` triggers one ``init_db``-based
  schema repair attempt per slug per gateway lifetime — gated by the
  ``schema_repair_attempted`` set so the per-tick race that #21378
  fixed cannot return.
* If the retry succeeds, the board stays enabled.
* If the retry fails, the board is added to ``disabled_schema_boards``
  (mtime-fingerprinted, mirroring the corrupt-board path) and skipped
  on subsequent ticks until the file changes or the gateway restarts.
* Any other persistent SQL error is rate-limited by
  ``_log_tick_exception`` to at most one ``logger.exception`` per
  ``(slug, exc_class)`` per ``_TICK_EXC_LOG_WINDOW_SECONDS``, with
  suppressed counts surfaced in the next permitted line.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

import pytest


def _common_dispatcher_setup(monkeypatch, tmp_path, *, board_slug: str = "default"):
    """Wire up the minimum surface so ``_kanban_dispatcher_watcher`` can be
    invoked in a test without a real Hermes home or DB. Returns the runner
    and the path the test should ``stat`` for fingerprint mtimes.
    """
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb

    runner = object.__new__(GatewayRunner)
    runner._running = True

    db_path = tmp_path / "kanban.db"
    # Real on-disk file so ``_board_db_fingerprint`` can stat() it. Contents
    # don't matter — the test stubs ``_kb.connect``.
    db_path.write_bytes(b"")

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(
        _kb,
        "list_boards",
        lambda include_archived=False: [{"slug": board_slug}],
    )
    monkeypatch.setattr(
        _kb,
        "read_board_metadata",
        lambda slug: {"slug": slug},
    )
    monkeypatch.setattr(_kb, "kanban_db_path", lambda board=None: db_path)
    return runner, db_path


def _stub_asyncio(monkeypatch, runner, *, stop_after_ticks: int):
    """Make ``asyncio.to_thread`` run synchronously and stop the watcher
    after ``stop_after_ticks`` calls.
    """
    calls = {"to_thread": 0}

    async def _to_thread(fn, *args, **kwargs):
        calls["to_thread"] += 1
        result = fn(*args, **kwargs)
        if calls["to_thread"] >= stop_after_ticks:
            runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)
    return calls


def test_schema_drift_triggers_init_db_repair_and_retry_succeeds(
    monkeypatch, tmp_path, caplog
):
    """First tick: ``connect`` raises ``no such column: session_id``. The
    dispatcher must call ``init_db`` once, retry the tick, and on success
    leave the board enabled. No ``tick failed`` error log.
    """
    import hermes_cli.kanban_db as _kb

    runner, db_path = _common_dispatcher_setup(monkeypatch, tmp_path)

    connect_calls = {"count": 0}
    init_db_calls = {"count": 0}

    class _FakeConn:
        def close(self):
            return None

    def _connect(*args, **kwargs):
        connect_calls["count"] += 1
        if connect_calls["count"] == 1:
            raise sqlite3.OperationalError("no such column: session_id")
        return _FakeConn()

    def _init_db(*args, **kwargs):
        init_db_calls["count"] += 1
        return db_path

    def _dispatch_once(conn, **kwargs):
        # Return a value with ``.spawned`` falsy so the watcher's quiet-by
        # default log path is taken.
        return type("R", (), {"spawned": []})()

    monkeypatch.setattr(_kb, "connect", _connect)
    monkeypatch.setattr(_kb, "init_db", _init_db)
    monkeypatch.setattr(_kb, "dispatch_once", _dispatch_once)
    monkeypatch.setattr(_kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(_kb, "has_spawnable_review", lambda conn: False)

    _stub_asyncio(monkeypatch, runner, stop_after_ticks=2)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(
                runner._kanban_dispatcher_watcher(),
                timeout=3.0,
            )
        )

    assert init_db_calls["count"] == 1, (
        "init_db must be called exactly once during schema-drift recovery"
    )
    messages = [r.getMessage() for r in caplog.records]
    # The warn line that announces the repair is the contract for operators.
    assert any(
        "reports missing session_id" in m and "retrying" in m for m in messages
    ), f"missing schema-repair warning; got: {messages}"
    # And no traceback was logged.
    assert not any(r.exc_info for r in caplog.records), (
        "no exception tracebacks should be logged when the repair succeeds"
    )
    assert not any("tick failed on board" in m for m in messages)


def test_schema_drift_persistent_failure_disables_board(
    monkeypatch, tmp_path, caplog
):
    """If the retry also fails, the board moves into
    ``disabled_schema_boards`` and subsequent ticks short-circuit without
    further ``init_db`` calls.
    """
    import hermes_cli.kanban_db as _kb

    runner, _ = _common_dispatcher_setup(monkeypatch, tmp_path)

    connect_calls = {"count": 0}
    init_db_calls = {"count": 0}

    def _connect(*args, **kwargs):
        connect_calls["count"] += 1
        raise sqlite3.OperationalError("no such column: session_id")

    def _init_db(*args, **kwargs):
        init_db_calls["count"] += 1
        return None

    monkeypatch.setattr(_kb, "connect", _connect)
    monkeypatch.setattr(_kb, "init_db", _init_db)
    monkeypatch.setattr(_kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(_kb, "has_spawnable_review", lambda conn: False)

    _stub_asyncio(monkeypatch, runner, stop_after_ticks=5)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(
                runner._kanban_dispatcher_watcher(),
                timeout=3.0,
            )
        )

    # One repair attempt across the whole lifetime of the watcher — not
    # one per tick. The schema_repair_attempted guard is the contract.
    assert init_db_calls["count"] == 1, (
        f"init_db must be called exactly once across multiple failing "
        f"ticks; got {init_db_calls['count']}"
    )
    # The disabled-board log line must mention the missing identifier so
    # operators know which column to add to the migration.
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "still failing after schema repair" in m and "session_id" in m
        for m in messages
    ), f"missing schema-disable error; got: {messages}"


def test_persistent_unknown_sql_error_is_rate_limited(
    monkeypatch, tmp_path, caplog
):
    """A persistent SQL error that's *not* a schema-drift error (and so
    can't be self-healed) must be logged at most once per
    ``(slug, exc_class)`` per window, with suppressed counts surfaced.
    """
    import hermes_cli.kanban_db as _kb

    runner, _ = _common_dispatcher_setup(monkeypatch, tmp_path)

    def _connect(*args, **kwargs):
        # Not a schema-drift signature — so the schema branch is skipped
        # and the generic rate-limited path takes over.
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(_kb, "connect", _connect)
    monkeypatch.setattr(_kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(_kb, "has_spawnable_review", lambda conn: False)

    _stub_asyncio(monkeypatch, runner, stop_after_ticks=10)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(
                runner._kanban_dispatcher_watcher(),
                timeout=3.0,
            )
        )

    tick_failed_records = [
        r for r in caplog.records if "tick failed on board" in r.getMessage()
    ]
    # Multiple ticks raised — must be at most one logged inside the 60s
    # window. The throttle uses ``time.monotonic`` so the entire test runs
    # well within a single window.
    assert len(tick_failed_records) <= 1, (
        f"expected ≤1 tick-failed log line under the rate-limit window; "
        f"got {len(tick_failed_records)}: "
        f"{[r.getMessage() for r in tick_failed_records]}"
    )


def test_schema_drift_recovery_state_clears_on_fingerprint_change(
    monkeypatch, tmp_path, caplog
):
    """A board disabled by schema-drift must come back when its DB file
    mtime changes — symmetric with the corrupt-board re-enable path.
    """
    import time as _time
    import hermes_cli.kanban_db as _kb

    runner, db_path = _common_dispatcher_setup(monkeypatch, tmp_path)

    # First and second connects fail (drift + failed repair). Third onwards
    # succeed — but the third only fires after the fingerprint changes.
    connect_calls = {"count": 0}

    class _FakeConn:
        def close(self):
            return None

    def _connect(*args, **kwargs):
        connect_calls["count"] += 1
        if connect_calls["count"] <= 2:
            raise sqlite3.OperationalError("no such column: session_id")
        return _FakeConn()

    def _init_db(*args, **kwargs):
        return None

    def _dispatch_once(conn, **kwargs):
        return type("R", (), {"spawned": []})()

    monkeypatch.setattr(_kb, "connect", _connect)
    monkeypatch.setattr(_kb, "init_db", _init_db)
    monkeypatch.setattr(_kb, "dispatch_once", _dispatch_once)
    monkeypatch.setattr(_kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(_kb, "has_spawnable_review", lambda conn: False)

    tick_count = {"n": 0}

    async def _to_thread(fn, *args, **kwargs):
        tick_count["n"] += 1
        result = fn(*args, **kwargs)
        if tick_count["n"] == 2:
            # Fingerprint change between tick 2 and tick 3: simulate the
            # user touching the DB (e.g., ``hermes kanban init``).
            _time.sleep(0.01)  # ensure mtime advances
            db_path.write_bytes(b"\x00")
        if tick_count["n"] >= 6:
            runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(
                runner._kanban_dispatcher_watcher(),
                timeout=3.0,
            )
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "database changed; retrying dispatch" in m for m in messages
    ), (
        "Schema-disabled board must re-enable on fingerprint change. "
        f"Messages: {messages}"
    )
