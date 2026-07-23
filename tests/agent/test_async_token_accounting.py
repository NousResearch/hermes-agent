"""Async token accounting — SessionDB background writer queue.

queue_token_counts() must take the per-call sessions UPDATE off the turn
thread while preserving update_token_counts() semantics exactly:

1. Deltas apply in enqueue order.
2. Coalescing consecutive same-route deltas is sum-equivalent to applying
   them one by one (sessions row AND session_model_usage breakdown).
3. flush_token_counts() gives readers read-your-writes (get_session and
   friends call it), and turn finalize / close() drain the queue.
4. A failing apply is logged by the writer and never raises into a turn.
"""

import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


def _totals(db, session_id):
    """Read token totals via raw SQL — bypasses get_session's flush so the
    read observes only what the writer has actually persisted."""
    with db._lock:
        row = db._conn.execute(
            "SELECT input_tokens, output_tokens, cache_read_tokens,"
            " cache_write_tokens, reasoning_tokens, api_call_count,"
            " estimated_cost_usd, actual_cost_usd, model, cost_status"
            " FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def _model_usage(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT model, input_tokens, output_tokens, api_call_count,"
            " estimated_cost_usd FROM session_model_usage"
            " WHERE session_id = ? ORDER BY model",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# =========================================================================
# Ordering
# =========================================================================


class TestOrdering:
    def test_deltas_apply_in_enqueue_order(self, db):
        """The writer applies deltas strictly in enqueue order, including
        across sessions (which never coalesce with each other)."""
        db.create_session("s-a", "test")
        db.create_session("s-b", "test")

        applied = []
        original = db.update_token_counts

        def recording(session_id, **kwargs):
            applied.append((session_id, kwargs.get("input_tokens", 0)))
            return original(session_id, **kwargs)

        db.update_token_counts = recording
        try:
            expected = []
            for i in range(1, 7):
                sid = "s-a" if i % 2 else "s-b"
                db.queue_token_counts(sid, input_tokens=i, api_call_count=1)
                expected.append((sid, i))
            assert db.flush_token_counts()
        finally:
            db.update_token_counts = original

        # Alternating sessions defeats coalescing, so every delta must be
        # applied individually, in order.
        assert applied == expected
        assert _totals(db, "s-a")["input_tokens"] == 1 + 3 + 5
        assert _totals(db, "s-b")["input_tokens"] == 2 + 4 + 6

    def test_absolute_delta_is_an_ordering_barrier(self, db):
        """incremental → absolute → incremental applies in order: the
        absolute overwrite wins over earlier increments, later increments
        stack on top of it."""
        db.create_session("s-abs", "test")
        db.queue_token_counts("s-abs", input_tokens=100, api_call_count=1)
        db.queue_token_counts(
            "s-abs", input_tokens=500, output_tokens=50,
            api_call_count=3, absolute=True,
        )
        db.queue_token_counts("s-abs", input_tokens=7, api_call_count=1)
        assert db.flush_token_counts()

        totals = _totals(db, "s-abs")
        assert totals["input_tokens"] == 507
        assert totals["output_tokens"] == 50
        assert totals["api_call_count"] == 4


# =========================================================================
# Coalescing
# =========================================================================


class TestCoalescing:
    def test_backlog_coalesces_and_sums_match(self, db):
        """When a backlog forms, same-route deltas merge into fewer applies
        while totals stay exact."""
        db.create_session("s-c", "test")

        apply_calls = []
        first_apply_started = threading.Event()
        release_first_apply = threading.Event()
        original = db.update_token_counts

        def gated(session_id, **kwargs):
            apply_calls.append(kwargs)
            if len(apply_calls) == 1:
                first_apply_started.set()
                # Hold the writer inside its first apply so the remaining
                # enqueues pile up into one batch.
                assert release_first_apply.wait(timeout=10)
            return original(session_id, **kwargs)

        db.update_token_counts = gated
        try:
            n = 20
            db.queue_token_counts(
                "s-c", input_tokens=1, output_tokens=1,
                estimated_cost_usd=0.001, model="m1",
                billing_provider="p1", api_call_count=1,
            )
            assert first_apply_started.wait(timeout=10)
            for _ in range(n - 1):
                db.queue_token_counts(
                    "s-c", input_tokens=1, output_tokens=1,
                    estimated_cost_usd=0.001, model="m1",
                    billing_provider="p1", api_call_count=1,
                )
            release_first_apply.set()
            assert db.flush_token_counts()
        finally:
            db.update_token_counts = original

        # The backlog collapses into far fewer UPDATEs than enqueues.
        assert len(apply_calls) < n
        totals = _totals(db, "s-c")
        assert totals["input_tokens"] == n
        assert totals["output_tokens"] == n
        assert totals["api_call_count"] == n
        assert totals["estimated_cost_usd"] == pytest.approx(0.001 * n)
        # Per-model attribution must also see the full sum.
        usage = _model_usage(db, "s-c")
        assert len(usage) == 1
        assert usage[0]["input_tokens"] == n
        assert usage[0]["api_call_count"] == n

    def test_coalesced_apply_equals_sequential_apply(self, db, tmp_path):
        """Applying a coalesced batch produces byte-identical session and
        per-model rows to applying the same deltas one at a time."""
        batch = [
            ("s-eq", dict(input_tokens=10, output_tokens=2, model="m1",
                          billing_provider="p1", estimated_cost_usd=0.01,
                          cost_status="estimated", api_call_count=1)),
            ("s-eq", dict(input_tokens=20, output_tokens=4, model="m1",
                          billing_provider="p1", estimated_cost_usd=0.02,
                          cost_status="estimated", api_call_count=1)),
            # /model switch mid-session — must not merge with the m1 run.
            ("s-eq", dict(input_tokens=5, output_tokens=1, model="m2",
                          billing_provider="p1", estimated_cost_usd=0.005,
                          cost_status="estimated", api_call_count=1)),
        ]

        db.create_session("s-eq", "test")
        db._apply_token_batch(list(batch))

        seq_db = SessionDB(db_path=tmp_path / "sequential.db")
        try:
            seq_db.create_session("s-eq", "test")
            for sid, kwargs in batch:
                seq_db.update_token_counts(sid, **kwargs)
            assert _totals(db, "s-eq") == _totals(seq_db, "s-eq")
            assert _model_usage(db, "s-eq") == _model_usage(seq_db, "s-eq")
        finally:
            seq_db.close()

    def test_coalesce_unit_rules(self, db):
        """_coalesce_token_deltas merge rules: same route merges, session /
        route changes and absolute deltas do not."""
        inc = dict(model="m1", billing_provider="p1")
        out = db._coalesce_token_deltas([
            ("a", dict(input_tokens=1, api_call_count=1, **inc)),
            ("a", dict(input_tokens=2, api_call_count=1, **inc)),
            ("b", dict(input_tokens=4, api_call_count=1, **inc)),
            ("a", dict(input_tokens=8, api_call_count=1, **inc)),
            ("a", dict(input_tokens=16, api_call_count=1, model="m2",
                       billing_provider="p1")),
            ("a", dict(input_tokens=32, absolute=True)),
            ("a", dict(input_tokens=64, absolute=True)),
        ])
        assert [(sid, kw.get("input_tokens")) for sid, kw in out] == [
            ("a", 3),   # merged 1+2
            ("b", 4),   # session change
            ("a", 8),   # session change back
            ("a", 16),  # model change
            ("a", 32),  # absolute never merges
            ("a", 64),
        ]
        assert out[0][1]["api_call_count"] == 2

    def test_coalesce_cost_none_preserved(self, db):
        """An all-None cost run stays None after merging (COALESCE in the
        UPDATE must keep the stored value untouched)."""
        out = db._coalesce_token_deltas([
            ("a", dict(input_tokens=1, estimated_cost_usd=None)),
            ("a", dict(input_tokens=1, estimated_cost_usd=None)),
        ])
        assert len(out) == 1
        assert out[0][1]["estimated_cost_usd"] is None

        out = db._coalesce_token_deltas([
            ("a", dict(input_tokens=1, estimated_cost_usd=None)),
            ("a", dict(input_tokens=1, estimated_cost_usd=0.5)),
        ])
        assert out[0][1]["estimated_cost_usd"] == pytest.approx(0.5)


# =========================================================================
# Read-your-writes
# =========================================================================


class TestReaderFlush:
    def test_get_session_sees_queued_deltas(self, db):
        """get_session drains the queue first, so readers observe exact
        totals even while the writer is mid-backlog."""
        db.create_session("s-r", "test")

        original = db.update_token_counts

        def slow(session_id, **kwargs):
            time.sleep(0.05)  # keep the writer visibly behind the reader
            return original(session_id, **kwargs)

        db.update_token_counts = slow
        try:
            for i in range(1, 5):
                sid_tokens = i
                # Alternate models to defeat coalescing — four real applies.
                db.queue_token_counts(
                    "s-r", input_tokens=sid_tokens,
                    model=f"m{i % 2}", api_call_count=1,
                )
            row = db.get_session("s-r")
        finally:
            db.update_token_counts = original

        assert row["input_tokens"] == 1 + 2 + 3 + 4
        assert row["api_call_count"] == 4

    def test_flush_empty_queue_is_cheap_noop(self, db):
        assert db.flush_token_counts()
        # No writer thread was ever started by a bare flush.
        assert db._token_writer_thread is None

    def test_flush_after_close_drains_on_caller_thread(self, db):
        """After close() stops the writer, a late flush still drains queued
        deltas synchronously instead of losing them."""
        db.create_session("s-late", "test")
        db.flush_token_counts()
        db._stop_token_writer()  # simulate a stopped writer with the conn open
        db._token_queue.append(("s-late", dict(input_tokens=9, api_call_count=1)))
        assert db.flush_token_counts()
        assert _totals(db, "s-late")["input_tokens"] == 9

    def test_flush_waits_for_stop_flagged_live_writer(self, db):
        """A stop-flagged but still-running writer owns the queue: flush must
        wait for it (its loop drains before exiting), never drain on the
        caller's thread — that would commit newer deltas before the writer's
        in-flight older batch and could return True with that batch
        unapplied."""
        db.create_session("s-stop", "test")

        applied = []
        gate = threading.Event()
        first_apply_started = threading.Event()
        original = db.update_token_counts

        def gated(session_id, **kwargs):
            applied.append(kwargs.get("input_tokens"))
            if len(applied) == 1:
                first_apply_started.set()
                assert gate.wait(timeout=10)
            return original(session_id, **kwargs)

        db.update_token_counts = gated
        try:
            db.queue_token_counts("s-stop", input_tokens=1, api_call_count=1)
            assert first_apply_started.wait(timeout=10)
            # close() has set the stop flag but the writer is mid-apply.
            db._token_writer_stop = True
            db._token_queue.append(
                ("s-stop", dict(input_tokens=2, api_call_count=1))
            )
            # The writer is alive, so flush waits — timing out, NOT applying
            # the newer delta on this thread ahead of the in-flight batch.
            assert db.flush_token_counts(timeout=0.3) is False
            assert applied == [1]
            gate.set()
            # Once released, the stop-flagged writer drains the queue itself
            # before exiting, preserving enqueue order.
            assert db.flush_token_counts()
        finally:
            db.update_token_counts = original

        assert applied == [1, 2]
        assert _totals(db, "s-stop")["input_tokens"] == 3

    def test_concurrent_flush_waits_for_caller_drain(self, db):
        """The dead-writer caller-drain claims busy: a second flush must not
        report drained (fast path or locked path) while the first flusher's
        popped batch is still being applied outside the condition lock."""
        db.create_session("s-cc", "test")
        db.flush_token_counts()
        db._stop_token_writer()  # writer dead, connection still open

        applied = threading.Event()
        gate = threading.Event()
        original = db.update_token_counts

        def gated(session_id, **kwargs):
            applied.set()
            assert gate.wait(timeout=10)
            return original(session_id, **kwargs)

        db.update_token_counts = gated
        try:
            db._token_queue.append(
                ("s-cc", dict(input_tokens=4, api_call_count=1))
            )
            results = {}
            t_a = threading.Thread(
                target=lambda: results.__setitem__(
                    "a", db.flush_token_counts()
                )
            )
            t_a.start()
            assert applied.wait(timeout=10)
            # Flusher A is mid-apply with the queue already popped: B must
            # wait on the claimed busy flag, not return True.
            assert db.flush_token_counts(timeout=0.3) is False
            gate.set()
            t_a.join(timeout=10)
            assert results.get("a") is True
            assert db.flush_token_counts()
        finally:
            db.update_token_counts = original

        assert _totals(db, "s-cc")["input_tokens"] == 4

    def test_enqueue_after_writer_stop_applies_synchronously(self, db):
        """Once the writer is stopped for good, queue_token_counts falls back
        to the synchronous path instead of parking deltas on a queue no
        writer will ever drain."""
        db.create_session("s-sync", "test")
        db.queue_token_counts("s-sync", input_tokens=1, api_call_count=1)
        db._stop_token_writer()  # writer dead, connection still open

        db.queue_token_counts("s-sync", input_tokens=2, api_call_count=1)

        # Applied inline — nothing queued, no writer restarted.
        assert not db._token_queue
        assert db._token_writer_thread is None or not db._token_writer_thread.is_alive()
        totals = _totals(db, "s-sync")
        assert totals["input_tokens"] == 3
        assert totals["api_call_count"] == 2

    def test_enqueue_after_close_raises_at_call_site(self, tmp_path):
        """After close() the synchronous fallback surfaces the failure to the
        caller (whose try/except logs it) — the pre-queue contract — instead
        of silently dropping the delta."""
        db = SessionDB(db_path=tmp_path / "closed.db")
        db.create_session("s-closed", "test")
        db.queue_token_counts("s-closed", input_tokens=1, api_call_count=1)
        db.close()

        with pytest.raises(Exception):
            db.queue_token_counts("s-closed", input_tokens=2, api_call_count=1)
        assert not db._token_queue  # not parked on a dead queue either


# =========================================================================
# Ordering vs synchronous route writes (/model switch)
# =========================================================================


class TestRouteSwitchBarrier:
    def test_model_switch_applies_queued_deltas_first(self, db):
        """update_session_model / update_session_billing_route bypass the
        queue, so they must flush it first: a still-queued first delta
        carries the pre-switch route, and applying it after the switch
        UPDATE trips first_accounted_route (api_call_count == 0 + route
        mismatch) and resurrects the old model/provider on the row."""
        db.create_session("s-sw", "test")
        # First delta of the session, queued but not yet applied (writer
        # not started — same state as a backlogged writer).
        db._token_queue.append(("s-sw", dict(
            input_tokens=10, model="m1", billing_provider="p1",
            api_call_count=1,
        )))

        db.update_session_model("s-sw", "m2")
        db.update_session_billing_route(
            "s-sw", provider="p2", base_url="https://p2.example"
        )

        totals = _totals(db, "s-sw")
        # The switch wins on the session row…
        assert totals["model"] == "m2"
        assert _model_usage(db, "s-sw")[0]["model"] == "m1"
        with db._lock:
            row = db._conn.execute(
                "SELECT billing_provider FROM sessions WHERE id = ?",
                ("s-sw",),
            ).fetchone()
        assert row["billing_provider"] == "p2"
        # …and the queued delta was applied (before it), not dropped.
        assert totals["input_tokens"] == 10
        assert totals["api_call_count"] == 1


# =========================================================================
# Durability
# =========================================================================


class TestDurability:
    def test_close_drains_queue(self, tmp_path):
        """close() drains queued deltas before closing the connection, so a
        clean shutdown loses nothing."""
        db_path = tmp_path / "drain.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s-d", "test")
        for i in range(5):
            db.queue_token_counts("s-d", input_tokens=10, api_call_count=1)
        db.close()

        reopened = SessionDB(db_path=db_path)
        try:
            totals = _totals(reopened, "s-d")
            assert totals["input_tokens"] == 50
            assert totals["api_call_count"] == 5
        finally:
            reopened.close()

    def test_atexit_drain_is_idempotent_and_never_raises(self, db):
        db.create_session("s-x", "test")
        db.queue_token_counts("s-x", input_tokens=3, api_call_count=1)
        db._drain_token_queue_at_exit()
        db._drain_token_queue_at_exit()  # second call: writer already stopped
        assert _totals(db, "s-x")["input_tokens"] == 3

    def test_close_unregisters_atexit_hook(self, tmp_path):
        """close() must unregister the atexit drain hook: it holds a strong
        reference (bound method) that would otherwise pin every closed
        SessionDB — and its sqlite connection object — until interpreter
        exit in multi-open/close processes."""
        import gc
        import weakref

        db = SessionDB(db_path=tmp_path / "atexit.db")
        db.create_session("s-gc", "test")
        db.queue_token_counts("s-gc", input_tokens=1, api_call_count=1)
        db.close()

        ref = weakref.ref(db)
        del db
        gc.collect()
        assert ref() is None

    def test_persist_session_drains_queue(self, tmp_path, monkeypatch):
        """Turn finalize (_persist_session) flushes the accounting queue —
        the crash window is at most the in-flight call's delta."""
        import os
        monkeypatch.setitem(os.environ, "OPENROUTER_API_KEY", "test-key")
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "finalize.db")
        try:
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id="s-fin",
                skip_context_files=True,
                skip_memory=True,
            )
            agent._ensure_db_session()

            db.queue_token_counts(
                "s-fin", input_tokens=11, output_tokens=2, api_call_count=1
            )
            agent._persist_session(
                [{"role": "user", "content": "q"}],
                [],
            )
            # Raw read: the flush happened inside _persist_session itself.
            totals = _totals(db, "s-fin")
            assert totals["input_tokens"] == 11
            assert totals["api_call_count"] == 1
        finally:
            db.close()


# =========================================================================
# Failure isolation
# =========================================================================


class TestWriterFailure:
    def test_apply_failure_logs_and_does_not_raise(self, db, caplog):
        """A failing UPDATE is logged by the writer; enqueue/flush never
        raise into the turn, and the writer survives to apply later deltas."""
        db.create_session("s-f", "test")

        original = db.update_token_counts
        boom = {"raise": True}

        def flaky(session_id, **kwargs):
            if boom["raise"]:
                raise sqlite3.OperationalError("database is locked")
            return original(session_id, **kwargs)

        db.update_token_counts = flaky
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                db.queue_token_counts("s-f", input_tokens=5, api_call_count=1)
                assert db.flush_token_counts()
            assert any(
                "async token accounting" in rec.getMessage()
                for rec in caplog.records
            )

            # Writer thread survived the failure and keeps applying.
            boom["raise"] = False
            db.queue_token_counts("s-f", input_tokens=7, api_call_count=1)
            assert db.flush_token_counts()
        finally:
            db.update_token_counts = original

        assert _totals(db, "s-f")["input_tokens"] == 7
