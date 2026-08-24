"""A failed state.db schema repair must preserve the recovery image.

For rollback-journal databases the failed-repair invariant is byte identity:
the main file and its sidecars must be unchanged.  WAL mode has an important
qualification: SQLite may checkpoint committed frames when a connection is
opened or closed, so the main file's bytes are not themselves the recovery
image.  WAL tests therefore assert committed-row and recovery semantics,
while the byte-identity assertions remain on the DELETE/rollback-journal
path.

Reported incident: the automatic repair path deleted a user's transcripts and
then reported that it had failed.

``_repair_state_db_schema_locked`` ran its strategies on the live file, and
Strategy 2 ends in ``VACUUM``::

    PRAGMA writable_schema=ON
    DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'
    PRAGMA writable_schema=OFF
    VACUUM

VACUUM does not preserve what it cannot parse — it rebuilds the file from the
schema SQLite can still read. When the damage IS in the schema b-tree (page
1's child pointers aimed at data pages, which is exactly the ``malformed
database schema ()`` class this function exists to handle), the rebuild drops
every table hanging off the unreadable part. Measured on the reporting
install: ``state.db`` went from 3048 pages / 29 sessions / 2537 messages to
113 pages, in place.

The probe that follows then correctly reported the file was *still* malformed,
so the function returned ``repaired=False`` with "manual restore from backup
may be required" — after the only live copy had already been gutted.
Destroying the data and reporting the repair failed are not mutually exclusive
outcomes, and nothing in the code treated them as a contradiction.

The pre-repair backup (#69603) is a forensic artefact, not a recovery path:
nothing reads it back. So the invariant under test is the stronger one — a
repair that does not succeed must not change the file at all — plus the
structural assertion that makes it hold: the strategies never receive the live
database.

Fix under test: every strategy runs on a ``<db>.repair-scratch`` snapshot and
is copied back through SQLite's transactional backup API only once the result
is proven to open cleanly.

Mutation-checked: pointing ``_run_repair_strategies`` back at ``db_path``
instead of the scratch copy fails
``test_failed_repair_leaves_the_original_byte_identical`` and
``test_strategies_never_receive_the_live_database``.
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing
import os
import shutil
import sqlite3
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state
import hermes_state_repair
from hermes_state_repair import repair_state_db_schema

PAGE_SIZE = 4096


def _writer_after_stage(
    db_path: str,
    ready: "multiprocessing.synchronize.Event",
    start: "multiprocessing.synchronize.Event",
    result: "multiprocessing.queues.Queue",
) -> None:
    """Try one real cross-process write after staging has begun.

    The repair process owns the SQLite exclusion for the complete
    stage/strategy/promotion interval.  A writer is allowed to fail or to
    wait until that interval ends and commit afterwards; what is forbidden is
    a successful commit that promotion silently overwrites.
    """
    conn = sqlite3.connect(db_path, timeout=0.75, isolation_level=None)
    try:
        ready.set()
        if not start.wait(20):
            result.put(("not-started", "repair did not reach staging"))
            return
        try:
            conn.execute(
                "INSERT INTO messages (body) VALUES ('committed-after-stage')"
            )
            result.put(("committed", None))
        except sqlite3.Error as exc:
            result.put(("failed", str(exc)))
    finally:
        conn.close()


def _make_repair_test_db(path: Path, *, journal_mode: str = "delete") -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO sessions (name) VALUES ('seed')")
        conn.execute("INSERT INTO messages (body) VALUES ('seed')")
        actual = conn.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[0]
        if journal_mode == "wal" and actual != "wal":
            pytest.skip("SQLite build/filesystem does not support WAL")
    finally:
        conn.close()


def _leave_hot_wal_row(db_path: str) -> None:
    """Commit a WAL frame, then die without letting sqlite3 close the DB."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("INSERT INTO messages (body) VALUES ('committed-wal-row')")
    conn.commit()
    # Deliberately bypass Connection.close(): the next process must recover
    # the committed row from the hot WAL image, not from a pre-checkpoint main.
    os._exit(0)


def _probe_repair_lock_from_child(db_path: str, result) -> None:
    """Attempt the repair lock with a short timeout from another process."""
    hermes_state._REPAIR_LOCK_TIMEOUT_SECONDS = 0.5
    with hermes_state_repair._cross_process_repair_lock(Path(db_path)) as holding:
        result.put(holding)


def _write_populated_db(path: Path, *, sessions: int = 3, messages: int = 25) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA page_size={PAGE_SIZE}")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany(
        "INSERT INTO sessions (name) VALUES (?)",
        [(f"session-{i}",) for i in range(sessions)],
    )
    conn.executemany(
        "INSERT INTO messages (body) VALUES (?)",
        [(f"message body {i}" * 20,) for i in range(messages)],
    )
    conn.commit()
    conn.close()


def _break_the_schema_btree(path: Path) -> None:
    """Aim page 1's rightmost child at a data page.

    This is the shape of the reported corruption: ``sqlite_master``'s b-tree
    resolves to pages holding table content, so SQLite reports "malformed
    database schema ()" — the parentheses empty because the bogus row's name
    is not text.
    """
    data = bytearray(path.read_bytes())
    page_count = struct.unpack(">I", data[28:32])[0]
    assert page_count >= 3, "fixture needs a multi-page database"
    # Byte 100 is page 1's b-tree header; offset 108 is the rightmost pointer
    # on an interior page. Force page 1 to be interior and point it at the
    # last page, which holds table data rather than schema records.
    data[100] = 0x05
    struct.pack_into(">H", data, 103, 1)  # one cell
    struct.pack_into(">I", data, 108, page_count)  # rightmost -> data page
    struct.pack_into(">H", data, 112, PAGE_SIZE - 6)  # cell pointer
    struct.pack_into(">I", data, PAGE_SIZE - 6, page_count)
    path.write_bytes(bytes(data))


@pytest.fixture
def corrupt_db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    _write_populated_db(path)
    _break_the_schema_btree(path)
    with pytest.raises(sqlite3.DatabaseError):
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("SELECT * FROM sessions").fetchall()
        finally:
            conn.close()
    return path


# ---------------------------------------------------------------------------
# The structural guarantee
# ---------------------------------------------------------------------------


def test_strategies_never_receive_the_live_database(corrupt_db, monkeypatch):
    """Every strategy mutates its argument in place, so the property that
    makes them safe is simply that the argument is never the real file."""
    seen: list[Path] = []
    real = hermes_state_repair._run_repair_strategies

    def spy(path, report):
        seen.append(path)
        return real(path, report)

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", spy)
    repair_state_db_schema(corrupt_db)

    assert seen, "the repair path did not run at all"
    for path in seen:
        assert path != corrupt_db
        assert path.name.endswith(".repair-scratch")


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


def test_failed_repair_leaves_the_original_byte_identical(corrupt_db):
    before = hashlib.sha256(corrupt_db.read_bytes()).hexdigest()

    report = repair_state_db_schema(corrupt_db)

    after = hashlib.sha256(corrupt_db.read_bytes()).hexdigest()
    assert not report.get("repaired"), (
        "fixture precondition: this corruption is not automatically repairable"
    )
    assert before == after, (
        "a repair that FAILED rewrote the database anyway — this is the "
        "reported data loss: 29 sessions / 2537 messages became 113 pages "
        "while the function reported 'manual restore may be required'"
    )


def test_failed_repair_leaves_no_scratch_file_behind(corrupt_db):
    """A half-repaired file beside the DB is a trap for the next probe."""
    repair_state_db_schema(corrupt_db)
    leftovers = sorted(p.name for p in corrupt_db.parent.glob("*repair-scratch*"))
    assert leftovers == []


def test_failed_repair_still_takes_the_forensic_backup(corrupt_db):
    """Non-destructive repair does not make the #69603 backup redundant."""
    report = repair_state_db_schema(corrupt_db)
    assert report["backup_path"], "the pre-repair forensic copy is still required"
    assert Path(report["backup_path"]).exists()


# ---------------------------------------------------------------------------
# ...and a repair that DOES succeed must still land on the original path
# ---------------------------------------------------------------------------


def test_successful_repair_is_promoted_over_the_original(tmp_path, monkeypatch):
    """The scratch copy is a staging area, not a detour: a strategy that
    heals the copy must leave the healed bytes at ``db_path``."""
    db = tmp_path / "state.db"
    _write_populated_db(db)
    # Force the "already healthy" short-circuit off so the staging path runs,
    # and have the strategy pass mark a repair after writing a marker row.
    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda path: "forced-unhealthy"
    )

    def fake_strategies(scratch_path, report):
        conn = sqlite3.connect(str(scratch_path))
        conn.execute("INSERT INTO sessions (name) VALUES ('healed-on-scratch')")
        conn.commit()
        conn.close()
        report["repaired"] = True
        report["strategy"] = "test_strategy"
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", fake_strategies)

    report = repair_state_db_schema(db)
    assert report["repaired"] is True

    conn = sqlite3.connect(str(db))
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM sessions")]
    finally:
        conn.close()
    assert "healed-on-scratch" in names, (
        "the repaired copy was never promoted over the original"
    )
    assert not list(db.parent.glob("*repair-scratch*"))


@pytest.mark.parametrize("journal_mode", ("delete", "wal"))
def test_committed_writer_after_staging_is_never_lost(
    tmp_path, monkeypatch, journal_mode
):
    """A writer racing the repair lifecycle cannot be overwritten.

    This uses the real orchestration and both SQLite online-backup calls.  The
    strategy hook is only a deterministic latch: it writes the repaired
    marker to the real scratch database, then gives a separate process a
    chance to attempt its commit before promotion.  A successful writer must
    still be present after repair; a locked/timeout writer is also valid.
    """
    db = tmp_path / "state.db"
    _make_repair_test_db(db, journal_mode=journal_mode)

    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )
    ready = multiprocessing.get_context("spawn").Event()
    start = multiprocessing.get_context("spawn").Event()
    result = multiprocessing.get_context("spawn").Queue()
    writer = multiprocessing.get_context("spawn").Process(
        target=_writer_after_stage,
        args=(str(db), ready, start, result),
    )
    writer.start()
    assert ready.wait(20), "writer process did not initialize"

    def staged_strategy(scratch_path, report):
        with sqlite3.connect(str(scratch_path)) as conn:
            conn.execute(
                "INSERT INTO sessions (name) VALUES ('repaired-before-race')"
            )
            conn.commit()
        start.set()
        # Make the race deterministic: an unguarded implementation lets the
        # child commit here, after which promotion silently erases its row.
        time.sleep(1.0)
        report["repaired"] = True
        report["strategy"] = "race_test"
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", staged_strategy)
    report = repair_state_db_schema(db, backup=False)
    writer.join(20)
    if writer.is_alive():
        writer.terminate()
        writer.join(5)
        pytest.fail("writer process did not finish")

    outcome, detail = result.get(timeout=5)
    assert outcome in {"failed", "committed"}, (outcome, detail)
    assert report["repaired"] is True

    with sqlite3.connect(str(db)) as conn:
        rows = {
            body for (body,) in conn.execute("SELECT body FROM messages")
        }
        names = {
            name for (name,) in conn.execute("SELECT name FROM sessions")
        }
    assert "repaired-before-race" in names
    if outcome == "committed":
        assert "committed-after-stage" in rows, (
            "a writer reported a successful commit, but transactional repair "
            "promotion silently overwrote it"
        )


def test_environmental_aborts_do_not_burn_repair_ledger(tmp_path, monkeypatch):
    """Three disk/staging aborts leave the actual strategy budget untouched."""
    db = tmp_path / "state.db"
    _make_repair_test_db(db)
    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )
    monkeypatch.setattr(
        hermes_state_repair, "_repair_scratch_space_error",
        lambda _path: "temporary disk pressure",
    )

    for _ in range(3):
        report = repair_state_db_schema(db, backup=False)
        assert report["repaired"] is False
        assert report["error"] == "temporary disk pressure"

    ledger_path = hermes_state_repair._repair_ledger_path(db)
    assert not ledger_path.exists(), "environmental aborts must not consume attempts"

    monkeypatch.setattr(hermes_state_repair, "_repair_scratch_space_error", lambda _path: None)

    def successful_strategy(scratch_path, report):
        with sqlite3.connect(str(scratch_path)) as conn:
            conn.execute("INSERT INTO sessions (name) VALUES ('after-aborts')")
            conn.commit()
        report["repaired"] = True
        report["strategy"] = "after_environmental_aborts"
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", successful_strategy)
    report = repair_state_db_schema(db, backup=False)
    assert report["repaired"] is True
    assert report["strategy"] == "after_environmental_aborts"


def test_actual_strategy_failure_still_consumes_one_attempt(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _make_repair_test_db(db)
    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )

    def failed_strategy(_scratch_path, report):
        report["repaired"] = False
        report["strategy"] = None
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", failed_strategy)
    report = repair_state_db_schema(db, backup=False)
    assert report["repaired"] is False
    ledger = hermes_state_repair._read_repair_ledger(db)
    assert ledger["failed_attempts"] == 1


def test_repair_outcome_is_recorded_while_cross_process_lock_is_held(
    tmp_path, monkeypatch
):
    """Ledger publication must remain inside the repairer's critical section."""
    db = tmp_path / "state.db"
    _make_repair_test_db(db)
    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )

    def failed_strategy(_scratch_path, report):
        report["repaired"] = False
        report["strategy"] = None
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", failed_strategy)
    observed = []
    lock_released = threading.Event()
    real_repair_lock = hermes_state_repair._cross_process_repair_lock

    @contextlib.contextmanager
    def tracking_repair_lock(path):
        with real_repair_lock(path) as holding:
            yield holding
        lock_released.set()

    monkeypatch.setattr(
        hermes_state_repair, "_cross_process_repair_lock", tracking_repair_lock
    )

    def record_outcome(_db_path, *, repaired, fingerprint=None):
        assert not lock_released.is_set(), (
            "repair outcome was recorded after the cross-process lock released"
        )
        context = multiprocessing.get_context("spawn")
        result = context.Queue()
        probe = context.Process(
            target=_probe_repair_lock_from_child,
            args=(str(db), result),
        )
        probe.start()
        try:
            observed.append(result.get(timeout=5))
        finally:
            probe.join(5)
            if probe.is_alive():
                probe.terminate()
                probe.join(5)
        assert repaired is False

    monkeypatch.setattr(hermes_state_repair, "_record_repair_outcome", record_outcome)
    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert observed == [False], (
        "repair outcome was recorded after releasing the cross-process lock"
    )


def test_exhaustion_is_rechecked_after_acquiring_repair_lock(tmp_path, monkeypatch):
    """A queued repairer must not start surgery on a newly exhausted ledger."""
    db = tmp_path / "state.db"
    _make_repair_test_db(db)
    exhaustion_checks = []

    def exhaustion_probe(_db_path):
        exhaustion_checks.append(True)
        # The first caller's pre-lock view is stale; the queued view is
        # terminal because another repairer just recorded the final failure.
        return len(exhaustion_checks) >= 2

    monkeypatch.setattr(
        hermes_state_repair, "_persistent_repair_attempts_exhausted", exhaustion_probe
    )
    monkeypatch.setattr(hermes_state_repair, "_live_writer_holds_db", lambda _path: False)
    surgery_calls = []

    def unexpected_surgery(_db_path, *, backup, report):
        surgery_calls.append(True)
        return report

    monkeypatch.setattr(
        hermes_state_repair, "_repair_state_db_schema_locked", unexpected_surgery
    )

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert "automatic repair has already failed" in report["error"]
    assert len(exhaustion_checks) == 2
    assert surgery_calls == [], "surgery ran despite the under-lock exhaustion recheck"


def test_scratch_budget_counts_sidecars_in_vacuum_multiplier(tmp_path, monkeypatch):
    """A WAL-heavy snapshot must reserve 3x snapshot bytes, not 2x main."""
    db = tmp_path / "state.db"
    main_bytes = 100
    wal_bytes = 900
    db.write_bytes(b"m" * main_bytes)
    db.with_name(db.name + "-wal").write_bytes(b"w" * wal_bytes)
    total = 10_000_000_000
    headroom = hermes_state_repair._repair_backup_headroom_bytes(total)
    # This exactly satisfies the obsolete ``snapshot + 2*main + headroom``
    # calculation, but is below the corrected ``3*snapshot + headroom``.
    old_required = main_bytes + wal_bytes + (2 * main_bytes) + headroom
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=total, free=old_required),
    )

    error = hermes_state_repair._repair_scratch_space_error(db)

    assert error is not None
    assert "VACUUM may need another" in error


def test_environmental_promotion_failures_do_not_burn_ledger(
    tmp_path, monkeypatch
):
    """Transient promotion disk errors remain retriable across three passes."""
    db = tmp_path / "state.db"
    _make_repair_test_db(db)
    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )

    def successful_strategy(_scratch_path, report):
        report["repaired"] = True
        report["strategy"] = "promotion_environment_test"
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", successful_strategy)
    real_copy = hermes_state_repair._copy_database_snapshot
    copy_calls = 0

    def fail_promotion_three_times(source, destination, **kwargs):
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls in (2, 4, 6):
            raise sqlite3.OperationalError("database or disk is full")
        return real_copy(source, destination, **kwargs)

    monkeypatch.setattr(
        hermes_state_repair, "_copy_database_snapshot", fail_promotion_three_times
    )
    for _ in range(3):
        report = repair_state_db_schema(db, backup=False)
        assert report["repaired"] is False
        assert "could not be promoted" in report["error"]

    ledger = hermes_state_repair._read_repair_ledger(db)
    assert ledger.get("failed_attempts", 0) == 0

    monkeypatch.setattr(hermes_state_repair, "_copy_database_snapshot", real_copy)
    report = repair_state_db_schema(db, backup=False)
    assert report["repaired"] is True
    assert report["strategy"] == "promotion_environment_test"


def test_corrupt_promotion_failure_consumes_one_attempt(tmp_path, monkeypatch):
    """A deterministic malformed-image promotion failure is budgeted."""
    db = tmp_path / "state.db"
    _make_repair_test_db(db)
    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )

    def successful_strategy(_scratch_path, report):
        report["repaired"] = True
        report["strategy"] = "corrupt-promotion-test"
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", successful_strategy)
    real_copy = hermes_state_repair._copy_database_snapshot
    calls = 0

    def fail_promotion_with_corruption(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return real_copy(source, destination, **kwargs)

    monkeypatch.setattr(
        hermes_state_repair, "_copy_database_snapshot", fail_promotion_with_corruption
    )
    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert hermes_state_repair._read_repair_ledger(db)["failed_attempts"] == 1


def test_snapshot_includes_committed_wal_frames(tmp_path):
    """The recovery image includes commits not checkpointed to the WAL main."""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute("CREATE TABLE messages (body TEXT)")
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("INSERT INTO messages VALUES ('committed-only-in-wal')")
        assert db.with_name(db.name + "-wal").exists()

        scratch = tmp_path / "state.db.repair-scratch"
        hermes_state_repair._copy_database_snapshot(db, scratch)
        with sqlite3.connect(str(scratch)) as check:
            assert check.execute("SELECT body FROM messages").fetchall() == [
                ("committed-only-in-wal",)
            ]
    finally:
        conn.close()


@pytest.mark.requires_wal
def test_failed_wal_repair_preserves_committed_rows_semantically(
    tmp_path, monkeypatch
):
    """WAL correctness is about committed recovery state, not main-file bytes."""
    db = tmp_path / "state.db"
    _make_repair_test_db(db)
    context = multiprocessing.get_context("spawn")
    writer = context.Process(target=_leave_hot_wal_row, args=(str(db),))
    writer.start()
    writer.join(20)
    if writer.is_alive():
        writer.terminate()
        writer.join(5)
        pytest.fail("hot-WAL fixture process did not finish")
    assert writer.exitcode == 0, "hot-WAL fixture process failed"
    if not db.with_name(db.name + "-wal").exists():
        pytest.skip("SQLite/filesystem did not retain a hot WAL sidecar")

    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )

    strategy_calls = []

    def failed_strategy(_scratch_path, report):
        strategy_calls.append(True)
        report["repaired"] = False
        report["strategy"] = None
        return report

    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", failed_strategy)
    report = repair_state_db_schema(db, backup=False)
    assert report["repaired"] is False
    assert strategy_calls == [True], "the test must exercise strategy failure"

    with sqlite3.connect(str(db)) as conn:
        bodies = {
            body for (body,) in conn.execute("SELECT body FROM messages")
        }
    assert {"seed", "committed-wal-row"} <= bodies


def test_snapshot_deadline_has_a_floor_and_scales_with_source_size(
    tmp_path, monkeypatch
):
    """Large snapshots get more than the lock floor without huge fixtures."""
    deadline = getattr(hermes_state_repair, "_repair_snapshot_timeout_seconds", None)
    assert callable(deadline), "repair snapshots need a size-scaled deadline"
    # Lower the throughput only for this arithmetic test so a 72 MiB fixture
    # crosses the floor without allocating a multi-GB file.
    monkeypatch.setattr(
        hermes_state_repair, "_REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND", 256 * 1024
    )

    db = tmp_path / "state.db"
    db.write_bytes(b"x" * PAGE_SIZE)
    small = deadline(db)
    # A modest sparse-ish fixture is enough to cross the 120-second floor on
    # the production throughput constant; no multi-GB allocation is needed.
    with db.open("ab") as fh:
        fh.truncate(64 * 1024 * 1024)
    db.with_name(db.name + "-wal").write_bytes(b"w" * (8 * 1024 * 1024))
    large = deadline(db)
    assert small >= hermes_state._REPAIR_LOCK_TIMEOUT_SECONDS
    assert large > small


def test_transactional_promotion_preserves_a_live_wal_reader(tmp_path):
    """Promotion must not replace the inode behind an existing reader."""
    live = tmp_path / "state.db"
    scratch = tmp_path / "scratch.db"
    for path, value in ((live, "old"), (scratch, "repaired")):
        with sqlite3.connect(str(path)) as conn:
            conn.execute("CREATE TABLE messages (body TEXT)")
            conn.execute("INSERT INTO messages VALUES (?)", (value,))
            conn.commit()
            if path == live:
                assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"

    reader = sqlite3.connect(str(live), isolation_level=None)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT body FROM messages").fetchall() == [("old",)]

        hermes_state_repair._copy_database_snapshot(scratch, live)

        assert reader.execute("SELECT body FROM messages").fetchall() == [("old",)]
        with sqlite3.connect(str(live)) as fresh:
            assert fresh.execute("SELECT body FROM messages").fetchall() == [
                ("repaired",)
            ]
    finally:
        reader.execute("ROLLBACK")
        reader.close()


def test_interrupted_snapshot_rolls_back_destination(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    destination = tmp_path / "destination.db"
    with sqlite3.connect(str(source)) as conn:
        conn.execute("CREATE TABLE payloads (body BLOB)")
        conn.executemany(
            "INSERT INTO payloads VALUES (?)",
            [(b"x" * PAGE_SIZE,) for _ in range(400)],
        )
        conn.commit()
    with sqlite3.connect(str(destination)) as conn:
        conn.execute("CREATE TABLE marker (value TEXT)")
        conn.execute("INSERT INTO marker VALUES ('original')")
        conn.commit()

    ticks = iter((0.0, hermes_state._REPAIR_LOCK_TIMEOUT_SECONDS + 1.0))
    monkeypatch.setattr(hermes_state.time, "monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError):
        hermes_state_repair._copy_database_snapshot(source, destination)

    with sqlite3.connect(str(destination)) as conn:
        assert conn.execute("SELECT value FROM marker").fetchall() == [("original",)]


def test_failed_promotion_returns_failure_and_preserves_original(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _write_populated_db(db)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    real_copy = hermes_state_repair._copy_database_snapshot
    calls = 0

    def fail_second_copy(source, destination, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_copy(source, destination, **kwargs)
        raise sqlite3.OperationalError("destination is busy")

    def fake_strategies(_scratch, report):
        report["repaired"] = True
        report["strategy"] = "test_strategy"
        return report

    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )
    monkeypatch.setattr(hermes_state_repair, "_copy_database_snapshot", fail_second_copy)
    monkeypatch.setattr(hermes_state_repair, "_run_repair_strategies", fake_strategies)

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert report["strategy"] is None
    assert "could not be promoted" in report["error"]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob("*repair-scratch*"))


def test_scratch_space_guard_accounts_for_snapshot_and_vacuum(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"x" * 4096)
    headroom = hermes_state_repair._repair_backup_headroom_bytes(10_000_000_000)
    free = (3 * db.stat().st_size) + headroom - 1
    usage = SimpleNamespace(total=10_000_000_000, free=free)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: usage)

    error = hermes_state_repair._repair_scratch_space_error(db)

    assert error is not None
    assert "VACUUM may need" in error


def test_stale_scratch_is_removed_before_health_check(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _write_populated_db(db)
    scratch = tmp_path / "state.db.repair-scratch"
    scratch.write_bytes(b"crash debris" * 1000)
    checks: list[str] = []

    def fake_health(_path):
        checks.append("health")
        assert not scratch.exists()
        return None

    monkeypatch.setattr(hermes_state_repair, "_db_opens_cleanly", fake_health)

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is True
    assert report["strategy"] == "already_healthy"
    assert checks == ["health"]
    assert not scratch.exists()


def test_stale_scratch_is_removed_before_space_check(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _write_populated_db(db)
    scratch = tmp_path / "state.db.repair-scratch"
    scratch.write_bytes(b"crash debris" * 1000)
    checked_space = False

    monkeypatch.setattr(
        hermes_state_repair, "_db_opens_cleanly", lambda _path: "forced-unhealthy"
    )

    def fake_space_check(_path):
        nonlocal checked_space
        checked_space = True
        assert not scratch.exists()
        return "forced low space"

    monkeypatch.setattr(
        hermes_state_repair, "_repair_scratch_space_error", fake_space_check
    )

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert report["error"] == "forced low space"
    assert checked_space is True
    assert not scratch.exists()


def test_stale_scratch_cleanup_failure_aborts_before_probe(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _write_populated_db(db)
    probed = False

    def fake_health(_path):
        nonlocal probed
        probed = True
        return "forced-unhealthy"

    monkeypatch.setattr(hermes_state_repair, "_db_opens_cleanly", fake_health)
    monkeypatch.setattr(
        hermes_state_repair, "_unlink_db_triple", lambda _path: "scratch is locked"
    )
    report = {
        "repaired": False,
        "strategy": None,
        "backup_path": None,
        "error": None,
    }

    result = hermes_state_repair._repair_state_db_schema_locked(
        db, backup=False, report=report
    )

    assert result["repaired"] is False
    assert "stale repair snapshot" in result["error"]
    assert probed is False
