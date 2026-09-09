"""A permanently-held state.db must not defer its FTS repair forever.

Deferring a stale-FTS rebuild while another process holds state.db is correct
while that holder might leave. When the holder is a supervised peer sharing one
HERMES_HOME -- a gateway alongside `hermes serve` -- it never leaves, the orphan
reap deliberately will not touch it, and every deferral keeps the FTS index
detached while the messages_fts triggers fail canonical writes. In #106393 that
ran for five days and 292 deferrals, losing every transcript in the window while
the process stayed healthy and the remedy was logged each time.

These tests pin the invariants of the exit condition, not its current numbers:
a holder that has not yet proven permanent is still deferred to, one that has is
not, and the rebuild's real safety (the cross-process rebuild flock) still holds
in both cases.
"""

import contextlib
import json
import multiprocessing
import os
import sqlite3
import time

import pytest

import hermes_state_schema
from hermes_state import SessionDB
from hermes_state_common import FTS_REBUILD_DEFERRAL_KEY, FTS_STALE_KEY

# Read the thresholds through getattr so an unpatched tree fails on BEHAVIOR
# (the rebuild never runs) rather than on an AttributeError at collection time.
# A test that errors before it exercises anything proves nothing about the bug.
FUTILE_ATTEMPTS = getattr(hermes_state_schema, "_FTS_HOLDER_FUTILE_ATTEMPTS", 10)
FUTILE_SECONDS = getattr(hermes_state_schema, "_FTS_HOLDER_FUTILE_SECONDS", 3600.0)


def _corrupt_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    raw.commit()
    raw.close()


def _meta_value(db_path, key):
    raw = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = raw.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        raw.close()


def _seed_stale_db(tmp_path, monkeypatch):
    """A closed state.db with a detached, stale FTS index and one live session."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    if not db._fts_enabled:
        db.close()
        pytest.skip("FTS5 unavailable in this build")
    db.create_session("s1", source="test")
    db.append_message("s1", "user", "seed calibration row")
    _corrupt_fts(db_path)
    monkeypatch.setattr(
        db, "rebuild_fts", lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("still corrupt"))
    )
    db.append_message("s1", "user", "written while fts was failing")
    db.close()
    return db_path


def _write_deferral(db_path, *, attempts, age_seconds, now, holder_pids=(4242,)):
    """Seed the deferral breadcrumb.

    `holder_pids` must match the holder the code will actually see: production
    compares the persisted set against the live scan and restarts the counters
    when they differ, so seeding a fictional PID would silently exercise the
    holder-changed path instead of the futility path.
    """
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (
            FTS_REBUILD_DEFERRAL_KEY,
            json.dumps({
                "first_seen": now - age_seconds,
                "last_seen": now,
                "attempts": attempts,
                "holder_pids": sorted(holder_pids),
            }),
        ),
    )
    raw.commit()
    raw.close()


def _pin_unreapable_holder(monkeypatch, db_path):
    """A holder that is real and that the orphan reap refuses to touch.

    This is the supervised-peer shape: `_reap_inactive_orphan_desktop_holders`
    returns nothing for it, exactly as it does for a systemd-managed
    `hermes serve` (non-orphan ppid, not an ephemeral backend, live clients).
    """
    monkeypatch.setattr(
        SessionDB, "_foreign_state_db_holders",
        lambda self: [(4242, str(db_path) + "-wal")], raising=False,
    )
    monkeypatch.setattr(
        SessionDB, "_reap_inactive_orphan_desktop_holders",
        lambda self, holders, *, min_age_seconds: [], raising=False,
    )


class TestUnprovableHolderStaysFailClosed:
    """The futility exit must never fire on a holder we could not identify.

    `foreign_state_db_holders()` returns `(-1, "open-file scan failed: ...")`
    when a /proc or open-file scan fails, or `(-1, "open-file scan
    unavailable")` when psutil is missing. Its docstring is explicit that
    structural maintenance must not assume quiescence in that state.

    That is precisely what `fts_rebuild_admission` cannot speak for: the flock
    serializes cooperating FTS rebuilders, it is not evidence that an
    uninspectable process -- or a process holding an unlinked SQLite generation
    -- has gone away. So an unknown holder keeps deferring past the threshold,
    forever if necessary, rather than trading a hang for possible corruption.
    """

    @pytest.mark.parametrize(
        "sentinel",
        [
            pytest.param((-1, "open-file scan failed: boom"), id="scan-failed"),
            pytest.param((-1, "open-file scan unavailable"), id="psutil-missing"),
        ],
    )
    def test_unknown_holder_past_the_threshold_still_defers(
        self, tmp_path, monkeypatch, sentinel
    ):
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        # Well past both thresholds; only the holder's identity differs.
        # holder_pids=() because production filters pid > 0, so a (-1, ...)
        # scan persists an EMPTY set. Seeding a fictional pid here would take
        # the holder-changed reset path and never reach the fail-closed branch.
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS * 5,
            age_seconds=FUTILE_SECONDS * 5,
            now=now,
            holder_pids=(),
        )
        monkeypatch.setattr(
            SessionDB, "_foreign_state_db_holders", lambda self: [sentinel], raising=False,
        )
        monkeypatch.setattr(
            SessionDB, "_reap_inactive_orphan_desktop_holders",
            lambda self, holders, *, min_age_seconds: [], raising=False,
        )
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True, (
                "an unprovable holder must not reach the futility exit"
            )
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
            # The history must be INTACT, not reset: an unchanged holder set
            # that is merely unprovable keeps accruing, and the exit declines
            # anyway. This is what distinguishes the fail-closed guard from the
            # holder-identity reset -- without the guard, this same state
            # reaches the exit and rebuilds.
            raw = _meta_value(db_path, FTS_REBUILD_DEFERRAL_KEY)
            assert raw is not None
            record = json.loads(raw)
            assert record["attempts"] > FUTILE_ATTEMPTS, (
                f"deferral history was reset (attempts={record['attempts']}), so this "
                "case is exercising the identity-change path, not the guard"
            )
            assert now - record["first_seen"] > FUTILE_SECONDS, (
                "first_seen was reset, so the futility thresholds were not actually met"
            )
            # Canonical writes stay available while deferred.
            reopened.append_message("s1", "user", "still writable while unprovable")
        finally:
            reopened.close()

    def test_unknown_holder_mixed_with_a_known_one_still_defers(
        self, tmp_path, monkeypatch
    ):
        """One unprovable entry poisons the whole set, not just its own row."""
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        # The known pid 4242 is what production persists for this mixed set
        # (the -1 entry is filtered out), so the identity check must agree and
        # the test must fail closed on the unprovable entry alone.
        _write_deferral(
            db_path, attempts=FUTILE_ATTEMPTS * 5, age_seconds=FUTILE_SECONDS * 5, now=now,
            holder_pids=(4242,),
        )
        monkeypatch.setattr(
            SessionDB, "_foreign_state_db_holders",
            lambda self: [(4242, str(db_path) + "-wal"), (-1, "open-file scan failed: x")],
            raising=False,
        )
        monkeypatch.setattr(
            SessionDB, "_reap_inactive_orphan_desktop_holders",
            lambda self, holders, *, min_age_seconds: [], raising=False,
        )
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            # Same distinction as above: the history must survive, proving the
            # guard declined rather than the identity check resetting it.
            raw = _meta_value(db_path, FTS_REBUILD_DEFERRAL_KEY)
            assert raw is not None
            record = json.loads(raw)
            assert record["attempts"] > FUTILE_ATTEMPTS
            assert now - record["first_seen"] > FUTILE_SECONDS
        finally:
            reopened.close()


class TestReapRescanReestablishesIdentity:
    """A reap that swaps the holder must not hand its window to the newcomer.

    The A->B protection runs before the orphan reap. If the reap removes A and
    the rescan returns a different positive-PID B, every downstream variable
    still describes A -- so the futility check can authorize B on its first
    deferral using A's accrued hour.
    """

    def test_holder_swapped_by_the_reap_restarts_the_window(self, tmp_path, monkeypatch):
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        # A has earned the full futility window.
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS * 5,
            age_seconds=FUTILE_SECONDS * 5,
            now=now,
            holder_pids=(4242,),
        )
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        scans = iter([
            [(4242, str(db_path) + "-wal")],   # pre-reap: A
            [(7777, str(db_path) + "-wal")],   # post-reap rescan: B, a different process
        ])
        monkeypatch.setattr(
            SessionDB, "_foreign_state_db_holders",
            lambda self: next(scans, [(7777, str(db_path) + "-wal")]),
            raising=False,
        )
        monkeypatch.setattr(
            SessionDB, "_reap_inactive_orphan_desktop_holders",
            lambda self, holders, *, min_age_seconds: [4242],
            raising=False,
        )

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True, (
                "B inherited A's window and was authorized on its first deferral"
            )
            raw = _meta_value(db_path, FTS_REBUILD_DEFERRAL_KEY)
            assert raw is not None
            record = json.loads(raw)
            assert record["holder_pids"] == [7777], (
                "the persisted identity must describe the post-reap holder"
            )
            assert record["attempts"] == 1, (
                f"the window must restart for B (attempts={record['attempts']})"
            )
            assert record["first_seen"] == now
        finally:
            reopened.close()


class TestReapClearedKeepsTheStartupBudget:
    """`_defer_stale_fts_for_holders` returns False for two different outcomes.

    Proven permanence means the holder is staying, so probing the rebuild lock
    non-blocking is right. A successful reap means the blocker is gone -- there
    the startup admission budget must survive, or transient contention postpones
    a repair that could have finished now.
    """

    def test_reap_cleared_holders_still_waits_for_admission(self, tmp_path, monkeypatch):
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        _write_deferral(
            db_path, attempts=3, age_seconds=120.0, now=now, holder_pids=(4242,),
        )
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        scans = iter([
            [(4242, str(db_path) + "-wal")],   # pre-reap
            [],                                # post-reap rescan: gone
        ])
        monkeypatch.setattr(
            SessionDB, "_foreign_state_db_holders",
            lambda self: next(scans, []), raising=False,
        )
        monkeypatch.setattr(
            SessionDB, "_reap_inactive_orphan_desktop_holders",
            lambda self, holders, *, min_age_seconds: [4242],
            raising=False,
        )

        seen = {}
        real = hermes_state_schema.fts_rebuild_admission

        def spy(db, *, timeout_seconds=None):
            seen["timeout"] = timeout_seconds
            return real(db, timeout_seconds=timeout_seconds)

        monkeypatch.setattr(hermes_state_schema, "fts_rebuild_admission", spy)

        reopened = SessionDB(db_path=db_path)
        try:
            assert seen.get("timeout") != 0.0, (
                "the reap-cleared path must keep the startup admission budget, "
                f"got timeout_seconds={seen.get('timeout')!r}"
            )
        finally:
            reopened.close()


class TestPermanenceBelongsToAHolderSet:
    """Accrued futility must not transfer from one holder to another.

    #106393's contract is "the SAME holder set has blocked N consecutive
    deferrals". Without comparing the persisted `holder_pids` against the
    current scan, holder A's accumulated hour lets a freshly-arrived holder B
    skip the wait on its very first deferral.
    """

    def test_a_new_holder_resets_the_accrued_history(self, tmp_path, monkeypatch):
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        # Holder A (pid 4242) accrued more than enough history to be futile.
        _write_deferral(
            db_path, attempts=FUTILE_ATTEMPTS, age_seconds=FUTILE_SECONDS, now=now
        )
        # ...but the holder present NOW is B (pid 7777).
        monkeypatch.setattr(
            SessionDB, "_foreign_state_db_holders",
            lambda self: [(7777, str(db_path) + "-wal")], raising=False,
        )
        monkeypatch.setattr(
            SessionDB, "_reap_inactive_orphan_desktop_holders",
            lambda self, holders, *, min_age_seconds: [], raising=False,
        )
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True, (
                "holder B inherited holder A's futility history"
            )
            # The breadcrumb must now describe B, with the counters restarted.
            raw = _meta_value(db_path, FTS_REBUILD_DEFERRAL_KEY)
            assert raw is not None, "the deferral breadcrumb was not written"
            record = json.loads(raw)
            assert record["holder_pids"] == [7777]
            assert record["attempts"] == 1
            assert record["first_seen"] == now
        finally:
            reopened.close()

    def test_the_same_holder_keeps_its_history(self, tmp_path, monkeypatch):
        """Control for the test above: an unchanged set must still reach the exit."""
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        _write_deferral(
            db_path, attempts=FUTILE_ATTEMPTS, age_seconds=FUTILE_SECONDS, now=now
        )
        _pin_unreapable_holder(monkeypatch, db_path)   # same pid 4242 as seeded
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is False
            assert _meta_value(db_path, FTS_REBUILD_DEFERRAL_KEY) is None
        finally:
            reopened.close()


class TestPermanentHolderExitCondition:
    def test_holder_not_yet_proven_permanent_is_still_deferred_to(self, tmp_path, monkeypatch):
        """Below the futility threshold nothing changes: the holder wins, cheaply."""
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        # One attempt short, and one second short, of proven-futile.
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS - 2,
            age_seconds=FUTILE_SECONDS - 1,
            now=now,
        )
        _pin_unreapable_holder(monkeypatch, db_path)
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
            # Canonical writes must stay available while deferred.
            reopened.append_message("s1", "user", "still writable while deferred")
        finally:
            reopened.close()

    def test_proven_permanent_holder_no_longer_blocks_the_rebuild(self, tmp_path, monkeypatch):
        """Past the threshold the rebuild proceeds and the breadcrumbs are cleared."""
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS,
            age_seconds=FUTILE_SECONDS,
            now=now,
        )
        _pin_unreapable_holder(monkeypatch, db_path)
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert _meta_value(db_path, FTS_REBUILD_DEFERRAL_KEY) is None
            # The rebuild is only useful if search actually works afterwards,
            # including for rows written while the index was detached.
            assert reopened.search_messages("written while fts was failing")
        finally:
            reopened.close()

    def test_futility_needs_both_attempts_and_elapsed_time(self, tmp_path, monkeypatch):
        """Many fast retries are not proof of permanence; the clock must agree too."""
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS * 10,
            age_seconds=1.0,  # a restart loop, not a permanent holder
            now=now,
        )
        _pin_unreapable_holder(monkeypatch, db_path)
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
        finally:
            reopened.close()

    def test_rebuild_flock_still_fails_closed_for_a_permanent_holder(self, tmp_path, monkeypatch):
        """The safety that replaces holder-absence must be load-bearing.

        Proven futility removes the holder gate, NOT the cross-process rebuild
        authority. With the flock unavailable the rebuild must still defer.
        """
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS,
            age_seconds=FUTILE_SECONDS,
            now=now,
        )
        _pin_unreapable_holder(monkeypatch, db_path)
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        @contextlib.contextmanager
        def _never_admitted(db_path, *, timeout_seconds=None):
            yield False

        monkeypatch.setattr(hermes_state_schema, "fts_rebuild_admission", _never_admitted)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        finally:
            reopened.close()

    def test_proceeding_past_a_permanent_holder_never_waits_on_the_flock(
        self, tmp_path, monkeypatch
    ):
        """Opening a permanently-blocked db must not inherit the admission budget.

        Before the exit condition existed, a database in this state returned at
        the holder gate and never reached `fts_rebuild_admission`. Now it does
        reach it, so a contended flock would add the full startup budget
        (_FTS_REBUILD_LOCK_TIMEOUT_SECONDS, 120s) to every open -- measured at
        120.1s before this was fixed. The holder is permanent by definition, so
        the acquire must be a probe: the breadcrumb guarantees the retry.
        """
        db_path = _seed_stale_db(tmp_path, monkeypatch)
        now = 1_000_000.0
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS,
            age_seconds=FUTILE_SECONDS,
            now=now,
        )
        _pin_unreapable_holder(monkeypatch, db_path)
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        seen_timeouts = []
        real_admission = hermes_state_schema.fts_rebuild_admission

        @contextlib.contextmanager
        def _recording_admission(path, *, timeout_seconds=None):
            seen_timeouts.append(timeout_seconds)
            with real_admission(path, timeout_seconds=timeout_seconds) as admitted:
                yield admitted

        monkeypatch.setattr(hermes_state_schema, "fts_rebuild_admission", _recording_admission)

        reopened = SessionDB(db_path=db_path)
        try:
            assert seen_timeouts, "the rebuild never reached the admission authority"
            # Non-blocking probe, not the startup budget (None) and not a wait.
            assert seen_timeouts[0] == 0.0, (
                f"proceeding past a permanent holder waited on the flock "
                f"(timeout={seen_timeouts[0]!r}); a contended lock would stall startup"
            )
        finally:
            reopened.close()


def _peer_writer(db_path, ready, stop, wrote, errors):
    """A second OS process holding the db read-write and committing throughout."""
    conns = []
    for _ in range(3):
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("SELECT count(*) FROM messages").fetchone()
        conns.append(conn)
    ready.set()
    i = 0
    while not stop.is_set():
        conn = conns[i % len(conns)]
        try:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?,?,?,?)",
                ("peer", "user", f"peer calibration row {i}", time.time()),
            )
            conn.commit()
            with wrote.get_lock():
                wrote.value += 1
        except sqlite3.Error:
            with errors.get_lock():
                errors.value += 1
        i += 1
        time.sleep(0.004)
    for conn in conns:
        conn.close()


@pytest.mark.linux_only
def test_rebuild_under_a_live_second_process_keeps_the_database_intact(tmp_path, monkeypatch):
    """E2E: the claim that makes the exit condition safe, against a real process.

    The holder gate was added because file-level surgery (WAL sidecar unlink,
    journal_mode flips) corrupts a live-WAL database under a concurrent holder
    (#90806, #90950). The stale-FTS rebuild does none of that -- it is one
    BEGIN IMMEDIATE transaction of SQL DDL/DML. This exercises that difference
    for real: a separate process holds the database and commits rows for the
    whole rebuild, and afterwards the database must be intact, no committed row
    may be missing, and search must work.
    """
    db_path = _seed_stale_db(tmp_path, monkeypatch)
    before = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    baseline = before.execute("SELECT count(*) FROM messages").fetchone()[0]
    before.close()

    ctx = multiprocessing.get_context("spawn")
    ready, stop = ctx.Event(), ctx.Event()
    wrote, errors = ctx.Value("i", 0), ctx.Value("i", 0)
    peer = ctx.Process(target=_peer_writer, args=(db_path, ready, stop, wrote, errors))
    peer.start()
    try:
        assert ready.wait(timeout=30), "peer process never became ready"

        # The peer must really be a foreign holder, or this proves nothing.
        held = set()
        for fd in os.listdir(f"/proc/{peer.pid}/fd"):
            try:
                target = os.readlink(f"/proc/{peer.pid}/fd/{fd}")
            except OSError:
                continue
            if os.path.basename(target).startswith("state.db"):
                held.add(os.path.basename(target))
        assert held, "peer is not holding state.db; the test would be vacuous"

        time.sleep(0.5)  # let the peer commit before the rebuild starts

        now = 1_000_000.0
        # Seed the history against the peer's REAL pid. Using a fictional one
        # would take the holder-changed reset path and never test futility.
        _write_deferral(
            db_path,
            attempts=FUTILE_ATTEMPTS,
            age_seconds=FUTILE_SECONDS,
            now=now,
            holder_pids=(peer.pid,),
        )
        monkeypatch.setattr(
            SessionDB, "_reap_inactive_orphan_desktop_holders",
            lambda self, holders, *, min_age_seconds: [], raising=False,
        )
        monkeypatch.setattr(hermes_state_schema.time, "time", lambda: now)

        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is False, "rebuild did not run under the live holder"
            time.sleep(0.5)  # peer keeps writing across the schema change
        finally:
            reopened.close()
    finally:
        stop.set()
        peer.join(timeout=30)
        if peer.is_alive():  # pragma: no cover - defensive
            peer.terminate()
            peer.join(timeout=10)

    peer_rows = wrote.value
    assert peer_rows > 0, "peer never committed; the test would be vacuous"
    assert errors.value == 0, f"peer writes failed during the rebuild: {errors.value}"

    check = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        total = check.execute("SELECT count(*) FROM messages").fetchone()[0]
        assert total == baseline + peer_rows, "committed rows were lost across the rebuild"
        # Every row is indexed, including those committed mid-rebuild.
        assert check.execute("SELECT count(*) FROM messages_fts").fetchone()[0] == total
        assert check.execute(
            "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'calibration'"
        ).fetchone()[0] > 0
    finally:
        check.close()
