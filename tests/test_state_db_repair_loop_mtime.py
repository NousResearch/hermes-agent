"""Regression: the state.db repair-loop guards must survive an mtime change.

Incident (2026-08-17): a malformed-SCHEMA state.db sent Hermes into an
unbounded repair loop that wrote a fresh 98MB forensic copy every ~10s —
2.3GB in 20 minutes, disk heading to zero, whole agent fleet at risk.

The #86747 guards were already present and did NOT hold, because both keyed
on ``size:mtime_ns``:

* ``_db_fingerprint`` -> the ledger's attempt counter reset to 1 on every
  pass, so ``_MAX_PERSISTENT_REPAIR_ATTEMPTS`` was never reached;
* ``_backup_db_file``'s dedupe compared mtime, so it never matched and each
  pass wrote another full-size copy.

Unlike the b-tree damage of #86747, the malformed-SCHEMA class still opens
and accepts writes (only ``sqlite_master`` is unreadable), so live writers,
WAL checkpoints and the in-place repair strategies all move mtime between
passes. These tests pin the guards to content, not mtime, and add the
missing free-space refusal.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import hermes_state
from hermes_state import (
    _MAX_MALFORMED_BACKUPS,
    _MAX_PERSISTENT_REPAIR_ATTEMPTS,
    _REPAIR_BACKUP_MIN_FREE_BYTES,
    _backup_content_identity,
    _backup_db_file,
    _db_fingerprint,
    _existing_malformed_backups,
    _persistent_repair_attempts_exhausted,
    _record_repair_outcome,
    _repair_backup_headroom_bytes,
)


def _damaged_db(tmp_path: Path, size: int = 200_000) -> Path:
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00" + os.urandom(size))
    return db


# ---------------------------------------------------------------------------
# Fingerprint stability
# ---------------------------------------------------------------------------


def test_fingerprint_survives_mtime_change(tmp_path):
    """A touched-but-unchanged file keeps its identity (the incident's core)."""
    db = _damaged_db(tmp_path)
    before = _db_fingerprint(db)
    time.sleep(0.01)
    os.utime(db, None)  # live writer / WAL checkpoint / in-place repair pass
    assert _db_fingerprint(db) == before


def test_fingerprint_changes_when_contents_change(tmp_path):
    """Genuine recovery must still reset the attempt budget."""
    db = _damaged_db(tmp_path)
    before = _db_fingerprint(db)
    db.write_bytes(b"SQLite format 3\x00" + os.urandom(200_000))
    assert _db_fingerprint(db) != before


def test_fingerprint_changes_on_truncation(tmp_path):
    db = _damaged_db(tmp_path)
    before = _db_fingerprint(db)
    with open(db, "r+b") as fh:
        fh.truncate(1024)
    assert _db_fingerprint(db) != before


# ---------------------------------------------------------------------------
# Attempt ledger
# ---------------------------------------------------------------------------


def test_attempt_budget_exhausts_despite_mtime_churn(tmp_path):
    """The loop must terminate even when every pass touches the file."""
    db = _damaged_db(tmp_path)
    for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        assert not _persistent_repair_attempts_exhausted(db)
        _record_repair_outcome(db, repaired=False)
        time.sleep(0.01)
        os.utime(db, None)
    assert _persistent_repair_attempts_exhausted(db)


def test_successful_repair_clears_budget(tmp_path):
    db = _damaged_db(tmp_path)
    for _ in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        _record_repair_outcome(db, repaired=False)
    assert _persistent_repair_attempts_exhausted(db)
    _record_repair_outcome(db, repaired=True)
    assert not _persistent_repair_attempts_exhausted(db)


# ---------------------------------------------------------------------------
# Backup dedupe
# ---------------------------------------------------------------------------


def test_backup_dedupes_across_mtime_change(tmp_path):
    """Repeated passes over identical bytes must not each write a new copy."""
    db = _damaged_db(tmp_path)
    first, err = _backup_db_file(db)
    assert err is None and first is not None
    for _ in range(5):
        time.sleep(0.01)
        os.utime(db, None)
        again, err = _backup_db_file(db)
        assert err is None
        assert again == first, "a touched-but-identical DB was copied again"
    assert len(_existing_malformed_backups(db)) == 1


def test_backup_retention_cap_still_holds(tmp_path):
    """Genuinely different damaged states are kept, but bounded."""
    db = _damaged_db(tmp_path)
    for _ in range(_MAX_MALFORMED_BACKUPS + 3):
        db.write_bytes(b"SQLite format 3\x00" + os.urandom(200_000))
        _backup_db_file(db)
    assert len(_existing_malformed_backups(db)) <= _MAX_MALFORMED_BACKUPS


# ---------------------------------------------------------------------------
# Free-space guard
# ---------------------------------------------------------------------------


def test_backup_refused_when_disk_would_be_exhausted(tmp_path):
    """A nearly-full volume must not be finished off by the forensic copy."""
    db = _damaged_db(tmp_path)
    tight = type(
        "Usage",
        (),
        {"total": 10_000_000_000, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES // 2},
    )()
    with patch("shutil.disk_usage", return_value=tight):
        path, reason = _backup_db_file(db)
    assert path is None
    assert reason is not None and "free" in reason.lower()
    assert not _existing_malformed_backups(db)


def test_backup_allowed_on_small_volume_with_room(tmp_path):
    """A flat multi-GB floor would disable repair on small VMs/containers.

    50MB DB on a 10GB volume with 1.5GB free fits with ~30x headroom; the
    guard must allow it rather than hard-stopping repair forever.
    """
    # Sparse: this test DOES copy the file, but the guard and copy both care
    # about st_size, not content — 50MB of os.urandom would only cost CI time.
    db = tmp_path / "state.db"
    with open(db, "wb") as handle:
        handle.write(b"SQLite format 3\x00")
        handle.truncate(50_000_000)
    assert db.stat().st_size == 50_000_000
    small_vm = type(
        "Usage", (), {"total": 10_000_000_000, "used": 8_500_000_000, "free": 1_500_000_000}
    )()
    with patch("shutil.disk_usage", return_value=small_vm):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None


def test_headroom_scales_with_volume_size():
    """Big volumes reserve proportionally; small ones keep a modest floor."""
    assert _repair_backup_headroom_bytes(1_000_000_000) == _REPAIR_BACKUP_MIN_FREE_BYTES
    assert _repair_backup_headroom_bytes(1_000_000_000_000) > _REPAIR_BACKUP_MIN_FREE_BYTES


def test_disk_guard_accounts_for_sidecars(tmp_path):
    """The copy includes -wal/-shm, so the space check must count them."""
    db = _damaged_db(tmp_path, size=1_000_000)
    # Sparse: the guard reads st_size, so allocating 400MB of real bytes would
    # only buy CI cost (and an ENOSPC risk on tmpfs runners).
    wal = db.with_name(db.name + "-wal")
    with open(wal, "wb") as handle:
        handle.truncate(400_000_000)
    assert wal.stat().st_size == 400_000_000
    usage = type(
        "Usage",
        (),
        {"total": 10_000_000_000, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES + 300_000_000},
    )()
    with patch("shutil.disk_usage", return_value=usage):
        path, reason = _backup_db_file(db)
    assert path is None, "sidecar bytes were ignored by the free-space check"
    assert reason is not None


def test_failed_copy_leaves_no_countable_debris(tmp_path):
    """Prune only runs on success, so a failed copy must self-clean.

    Otherwise partials matching the backup prefix accumulate unbounded and,
    on a later successful pass, are KEPT (newest by name) while intact
    forensic copies get pruned away.
    """
    db = _damaged_db(tmp_path, size=1_000_000)
    db.with_name(db.name + "-wal").write_bytes(os.urandom(1_000_000))
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()
    real_copy2 = shutil.copy2

    def sidecar_fails(src, dst, *a, **kw):
        if str(src).endswith("-wal"):
            Path(dst).write_bytes(b"PARTIAL" * 100)
            raise OSError(28, "No space left on device")
        return real_copy2(src, dst, *a, **kw)

    with patch("shutil.disk_usage", return_value=roomy), \
            patch("shutil.copy2", sidecar_fails):
        for _ in range(6):
            _backup_db_file(db)
            time.sleep(0.01)
            os.utime(db, None)

    assert len(_existing_malformed_backups(db)) <= _MAX_MALFORMED_BACKUPS

    # a later successful pass must sweep any staging debris
    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None
    strays = list(tmp_path.glob("*.backup-staging-*")) + list(
        tmp_path.glob("*.incomplete*")
    )
    assert not strays, f"staging debris survived: {strays}"


def test_backup_allowed_with_ample_disk(tmp_path):
    db = _damaged_db(tmp_path)
    roomy = type(
        "Usage", (), {"total": 0, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES * 10}
    )()
    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None


def test_repair_aborts_when_backup_refused_for_disk(tmp_path):
    """Refused backup is a HARD STOP — never mutate the only damaged copy."""
    db = _damaged_db(tmp_path)
    tight = type(
        "Usage", (), {"total": 0, "used": 0, "free": _REPAIR_BACKUP_MIN_FREE_BYTES // 2}
    )()
    with patch("shutil.disk_usage", return_value=tight):
        report = hermes_state.repair_state_db_schema(db)
    assert not report.get("repaired")
    assert "free" in (report.get("error") or "").lower()


# ---------------------------------------------------------------------------
# Lock safety: the content fingerprint must not cancel POSIX advisory locks
# ---------------------------------------------------------------------------


def test_fingerprint_takes_no_raw_fd_while_a_connection_is_live(tmp_path):
    """The content read must not ``open()`` a DB that has a live connection.

    ``close()`` on ANY descriptor cancels every POSIX advisory lock this
    process holds on the file (https://sqlite.org/howtocorrupt.html), so a
    peer connection's RESERVED lock is silently dropped and another process
    can write into a file the holder still believes it owns. The exhaustion
    probe runs BEFORE ``_backup_db_file``'s ``has_live_connection`` guard, so
    the fingerprint has to guard itself.
    """
    import builtins

    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t(a)")
    conn.commit()
    conn.close()

    live = connect_tracked(db, isolation_level=None, check_same_thread=False)
    try:
        opened: list[str] = []
        real_open = builtins.open

        def spy(target, *a, **kw):
            if str(target).endswith("state.db"):
                opened.append(str(target))
            return real_open(target, *a, **kw)

        with patch.object(builtins, "open", spy):
            fp = _db_fingerprint(db)

        assert not opened, f"raw fd taken on a live DB: {opened}"
        # None is the correct answer here — see
        # test_budget_exhausts_when_liveness_alternates_across_passes for why a
        # substitute key shape would be worse than no key at all. The ledger
        # keeps counting against the key already on record.
        assert fp is None
    finally:
        live.close()


def test_live_connection_keeps_its_write_lock_across_a_repair_pass(tmp_path):
    """End-to-end: a peer must not be able to steal the holder's write lock.

    The peer runs in a SUBPROCESS on purpose. POSIX advisory locks are owned
    per-process, so a same-process peer shares the holder's lock ownership and
    cannot demonstrate the cancellation — it stays blocked either way, which
    makes the test vacuous.

    Rollback-journal mode only — WAL coordinates through ``-shm`` rather than
    POSIX advisory locks, so it is immune. DELETE mode is what Hermes falls
    back to on NFS/SMB/FUSE/ZFS and on SQLite builds vulnerable to the
    WAL-reset bug, so it is a real deployment shape, not a corner case.
    """
    import subprocess
    import sys
    import textwrap

    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE sessions(id TEXT)")
    conn.commit()
    conn.close()

    peer_script = tmp_path / "peer.py"
    peer_script.write_text(
        textwrap.dedent(
            """
            import sqlite3, sys
            con = sqlite3.connect(sys.argv[1], timeout=0.3, isolation_level=None)
            try:
                con.execute("BEGIN IMMEDIATE")
                con.execute("INSERT INTO sessions VALUES('peer')")
                con.execute("COMMIT")
                print("WROTE")
            except sqlite3.OperationalError:
                print("BLOCKED")
            """
        )
    )

    def _peer_can_write() -> bool:
        out = subprocess.run(
            [sys.executable, str(peer_script), str(db)],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        assert out in {"WROTE", "BLOCKED"}, f"unexpected peer output: {out!r}"
        return out == "WROTE"

    live = connect_tracked(db, isolation_level=None, check_same_thread=False)
    try:
        live.execute("BEGIN IMMEDIATE")
        live.execute("INSERT INTO sessions VALUES('holder')")
        assert not _peer_can_write(), "peer wrote before the fingerprint (bad fixture)"

        _db_fingerprint(db)

        assert not _peer_can_write(), (
            "the fingerprint cancelled the holder's POSIX advisory lock"
        )
        live.execute("COMMIT")
    finally:
        live.close()


# ---------------------------------------------------------------------------
# Staging must never be mistaken for a forensic backup
# ---------------------------------------------------------------------------


def test_staging_name_is_outside_the_backup_prefix(tmp_path):
    """Whatever staging name the code picks must not be counted as a backup.

    Observes the REAL staging path (captured from the copy call) rather than
    hardcoding it, so the assertion binds to the invariant instead of to
    today's spelling. ``_existing_malformed_backups`` matches
    ``startswith(f"{db}.malformed-backup-")`` and excludes only ``-wal``/
    ``-shm``, so a staging name derived from the backup name sorts NEWEST and
    prune keeps partials while deleting intact copies.
    """
    db = _damaged_db(tmp_path, size=20_000)
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()
    real_copy2 = shutil.copy2
    staging_names: list[str] = []

    def capture(src, dst, *a, **kw):
        staging_names.append(Path(dst).name)
        return real_copy2(src, dst, *a, **kw)

    with patch("shutil.disk_usage", return_value=roomy), \
            patch("shutil.copy2", capture):
        path, reason = _backup_db_file(db)

    assert reason is None and path is not None
    assert staging_names, "no copy was made (fixture problem)"
    prefix = f"{db.name}.malformed-backup-"
    for name in staging_names:
        assert not name.startswith(prefix), (
            f"staging name {name!r} matches the backup prefix — it would be "
            "counted by _existing_malformed_backups, sort NEWEST, and let "
            "prune keep partials while deleting intact forensic copies"
        )


def test_orphaned_staging_is_never_returned_as_the_backup_path(tmp_path):
    """A kill mid-copy leaves a byte-identical staging file; the dedupe must
    not hand it back as the official ``backup_path``.

    It would pass the #69603 hard-stop gate — repair then runs destructive
    surgery believing a forensic copy exists — and the next pass's sweep
    deletes that very file.
    """
    db = _damaged_db(tmp_path, size=20_000)
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()

    # Discover the staging name the implementation actually uses, then plant an
    # orphan under it — so this binds to the code's scheme, not to a literal.
    real_copy2 = shutil.copy2
    seen: list[Path] = []

    def capture(src, dst, *a, **kw):
        seen.append(Path(dst))
        return real_copy2(src, dst, *a, **kw)

    with patch("shutil.disk_usage", return_value=roomy), \
            patch("shutil.copy2", capture):
        first, _ = _backup_db_file(db)
    assert first is not None
    Path(first).unlink(missing_ok=True)
    orphan = seen[0]
    shutil.copy2(db, orphan)  # identical bytes => fingerprint matches
    assert orphan.exists()

    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)

    assert reason is None and path is not None
    assert Path(path) != orphan, f"staging returned as the backup: {path}"
    assert not str(path).endswith(".incomplete")
    assert "staging" not in Path(path).name
    assert Path(path).exists()
    assert not orphan.exists(), "stale staging debris was not swept"


def test_backup_refused_when_free_space_cannot_be_determined(tmp_path):
    """Fail CLOSED: a nearly-full volume is where disk_usage is likeliest to
    fail, and proceeding is the multi-GB copy that finishes off the disk."""
    db = _damaged_db(tmp_path)
    with patch("shutil.disk_usage", side_effect=OSError("statvfs failed")):
        path, reason = _backup_db_file(db)
    assert path is None
    assert reason is not None and "free space" in reason.lower()
    assert not _existing_malformed_backups(db)


def test_budget_exhausts_when_liveness_alternates_across_passes(tmp_path):
    """A peer connection must not reset the attempt budget.

    ``_db_fingerprint`` returns None when a live connection makes the content
    read unsafe. If the ledger treated that as "no identity" (skip the record)
    or substituted a differently-shaped key (``size:mtime_ns``), then a gateway
    peer connecting and disconnecting between passes would reset the counter to
    1 forever — the exact unbounded loop this whole ledger exists to stop.
    """
    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t(a)")
    conn.commit()
    conn.close()

    for index in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        live = None
        if index % 2 == 1:  # a peer holds the DB on alternate passes
            live = connect_tracked(db, isolation_level=None, check_same_thread=False)
        try:
            assert not _persistent_repair_attempts_exhausted(db)
            _record_repair_outcome(db, repaired=False)
        finally:
            if live is not None:
                live.close()

    assert _persistent_repair_attempts_exhausted(db), (
        "alternating live/offline passes reset the repair budget"
    )
    # And an exhausted budget must stay visible even while a peer is connected.
    live = connect_tracked(db, isolation_level=None, check_same_thread=False)
    try:
        assert _persistent_repair_attempts_exhausted(db)
    finally:
        live.close()


def test_fingerprint_returns_none_rather_than_a_mtime_shaped_key(tmp_path):
    """Never mint a second key SHAPE — the ledger compares for equality."""
    from hermes_cli.sqlite_safe_read import connect_tracked

    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t(a)")
    conn.commit()
    conn.close()

    offline = _db_fingerprint(db)
    assert offline is not None
    live = connect_tracked(db, isolation_level=None, check_same_thread=False)
    try:
        assert _db_fingerprint(db) is None, (
            "a live connection produced a fingerprint; if its shape differs "
            "from the offline key the ledger can never match across passes"
        )
    finally:
        live.close()


# ---------------------------------------------------------------------------
# The content sample must exclude SQLite's commit counters
# ---------------------------------------------------------------------------


def _populated_db(path: Path, journal_mode: str, rows: int = 600) -> None:
    """A DB comfortably larger than the fingerprint sample window (~270KB)."""
    conn = sqlite3.connect(str(path))
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute("CREATE TABLE sessions(id TEXT, blob TEXT)")
    conn.executemany(
        "INSERT INTO sessions VALUES(?,?)",
        [(str(i), "x" * 400) for i in range(rows)],
    )
    conn.commit()
    conn.close()


def test_ordinary_commit_does_not_rekey_the_fingerprint(tmp_path):
    """A malformed-SCHEMA DB still accepts writes, so commits must not re-key.

    In rollback-journal (DELETE) mode a commit writes the main file directly and
    bumps the header's file change counter (bytes 24-27) and version-valid-for
    (92-95). Those live inside the head sample, so an unmasked fingerprint
    changed on every ordinary session write — resetting the repair budget to 1
    forever, which is exactly the unbounded loop this suite exists to pin.
    """
    for journal_mode in ("DELETE", "WAL"):
        db = tmp_path / f"state_{journal_mode}.db"
        _populated_db(db, journal_mode)
        before = _db_fingerprint(db)

        writer = sqlite3.connect(str(db), isolation_level=None)
        try:
            writer.execute("UPDATE sessions SET blob='peer' WHERE id='20000'")
        finally:
            writer.close()

        assert _db_fingerprint(db) == before, (
            f"{journal_mode} mode: an ordinary commit re-keyed the ledger"
        )


def test_budget_exhausts_while_a_writer_commits_between_passes(tmp_path):
    """End-to-end shape of the original incident, in DELETE mode."""
    db = tmp_path / "state.db"
    _populated_db(db, "DELETE")

    for index in range(_MAX_PERSISTENT_REPAIR_ATTEMPTS):
        assert not _persistent_repair_attempts_exhausted(db)
        _record_repair_outcome(db, repaired=False)
        writer = sqlite3.connect(str(db), isolation_level=None)
        try:
            writer.execute("UPDATE sessions SET blob=? WHERE id='20000'", (f"v{index}",))
        finally:
            writer.close()

    assert _persistent_repair_attempts_exhausted(db), (
        "a live writer's commits reset the repair budget every pass"
    )


def test_genuine_recovery_still_resets_the_budget(tmp_path):
    """Masking the commit counters must not blind us to real repair."""
    db = tmp_path / "state.db"
    _populated_db(db, "DELETE")

    def _mutate(sql: str) -> None:
        conn = sqlite3.connect(str(db), isolation_level=None)
        try:
            conn.execute(sql)
        finally:
            conn.close()

    for label, sql in (
        ("sqlite_master rewrite", "CREATE TABLE healed(x)"),
        ("index rebuild", "CREATE INDEX ix_sessions_id ON sessions(id)"),
        ("VACUUM", "VACUUM"),
    ):
        before = _db_fingerprint(db)
        _mutate(sql)
        assert _db_fingerprint(db) != before, f"{label} left the fingerprint unchanged"

    before = _db_fingerprint(db)
    with open(db, "r+b") as handle:
        handle.truncate(4096)
    assert _db_fingerprint(db) != before, "truncation left the fingerprint unchanged"


def test_forensic_backup_includes_the_rollback_journal(tmp_path):
    """DELETE mode leaves a hot -journal, and that file interprets the damage.

    Rollback-journal mode is Hermes's fallback on NFS/SMB/FUSE/ZFS and on
    WAL-reset-vulnerable SQLite builds. A forensic copy without the journal
    cannot be rolled back to a consistent state by hand.
    """
    db = _damaged_db(tmp_path, size=20_000)
    for suffix, payload in (("-wal", b"WALDATA"), ("-journal", b"JOURNALDATA")):
        db.with_name(db.name + suffix).write_bytes(payload)

    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()
    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None

    journal_copy = path.with_name(path.name + "-journal")
    assert journal_copy.exists(), "the rollback journal was left out of the backup"
    assert journal_copy.read_bytes() == b"JOURNALDATA"
    assert path.with_name(path.name + "-wal").read_bytes() == b"WALDATA"

    # Sidecar copies must not inflate the retention count.
    assert len(_existing_malformed_backups(db)) == 1


def test_prune_removes_journal_sidecars_too(tmp_path):
    """Otherwise the retention cap leaks one -journal per pruned backup."""
    db = _damaged_db(tmp_path, size=20_000)
    db.with_name(db.name + "-journal").write_bytes(b"J")
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()
    for _ in range(_MAX_MALFORMED_BACKUPS + 2):
        db.write_bytes(b"SQLite format 3\x00" + os.urandom(20_000))
        with patch("shutil.disk_usage", return_value=roomy):
            _backup_db_file(db)

    kept = _existing_malformed_backups(db)
    assert len(kept) <= _MAX_MALFORMED_BACKUPS

    # Assert on what is ON DISK rather than on the paths returned earlier: a
    # same-second stamp collision means an earlier return value can name a file
    # a later pass legitimately recreated.
    kept_names = {p.name for p in kept}
    orphans = [
        p.name
        for p in tmp_path.iterdir()
        if p.name.endswith("-journal")
        and ".malformed-backup-" in p.name
        and p.name[: -len("-journal")] not in kept_names
    ]
    assert not orphans, f"pruned backups left journals behind: {orphans}"
    # And every surviving backup keeps its journal.
    for survivor in kept:
        assert survivor.with_name(survivor.name + "-journal").exists()


# ---------------------------------------------------------------------------
# Backup identity vs repair-epoch fingerprint are different equivalence
# relations (the forensic dedupe must NOT reuse _db_fingerprint).
# ---------------------------------------------------------------------------


def test_backup_not_deduped_after_interior_page_write(tmp_path):
    """An interior-page write must force a fresh forensic backup.

    ``_db_fingerprint`` deliberately samples only head/tail and masks commit
    counters so an ordinary write does not re-key the repair budget. If the
    forensic dedupe reused THAT identity, a live writer committing new rows
    into an interior page (size preserved, first/last 64KiB untouched) would
    be handed the STALE earlier backup as "identical" — a recovery point that
    predates real user data. The dedupe must use ``_backup_content_identity``
    (whole file), which detects the interior change.
    """
    # Larger than 2x the 64KiB head/tail sample so a middle region exists
    # outside the sampled windows.
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00" + os.urandom(300_000))
    size = db.stat().st_size
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()

    with patch("shutil.disk_usage", return_value=roomy):
        first, err = _backup_db_file(db)
    assert err is None and first is not None

    # Mutate an interior byte far from both sampled windows; keep size + mtime.
    raw = bytearray(db.read_bytes())
    mid = len(raw) // 2
    raw[mid] ^= 0xFF
    st = db.stat()
    db.write_bytes(bytes(raw))
    os.utime(db, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert db.stat().st_size == size

    # Guard the test's own premise: the repair-epoch fingerprint is BLIND to
    # this change (that is why it must not be the dedupe key), while the
    # backup-content identity SEES it.
    assert _backup_content_identity(db) != _backup_content_identity(first)

    with patch("shutil.disk_usage", return_value=roomy):
        second, err = _backup_db_file(db)
    assert err is None and second is not None
    assert second != first, "an interior-page write was wrongly deduped to a stale backup"
    assert len(_existing_malformed_backups(db)) == 2


def test_publication_failure_leaves_no_countable_partial_bundle(tmp_path):
    """A mid-publish os.replace failure must not leave a countable main backup.

    The bundle is published sidecars-first, main-DB-last (the main name is the
    commit marker ``_existing_malformed_backups`` counts). If a promotion after
    the first fails, cleanup must roll back every already-published
    destination — otherwise an incomplete bundle (main present, a sidecar
    missing) survives, passes the #69603 hard stop, and is deduped/reused as a
    legitimate forensic copy on the next pass.

    Distinct from ``test_failed_copy_leaves_no_countable_debris``, which fails
    during ``copy2`` (before any ``os.replace``); this exercises the
    publication window.
    """
    db = _damaged_db(tmp_path, size=200_000)
    db.with_name(db.name + "-wal").write_bytes(os.urandom(50_000))
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()

    real_replace = os.replace
    calls = {"n": 0}

    def replace_fails_after_first(src, dst, *a, **kw):
        # Let the first promotion (a sidecar) land, fail the next one.
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *a, **kw)

    with patch("shutil.disk_usage", return_value=roomy), \
            patch("os.replace", replace_fails_after_first):
        try:
            _backup_db_file(db)
        except OSError:
            pass  # the failure is re-raised by design; we assert on-disk state

    # No countable main backup, and no orphaned promoted sidecar, may survive.
    assert not _existing_malformed_backups(db), "a partial bundle was left countable"
    promoted = [
        p for p in tmp_path.iterdir()
        if ".malformed-backup-" in p.name and p.name != "state.db"
    ]
    assert not promoted, f"partial promoted files survived: {promoted}"

    # A later clean pass must still succeed and must not dedupe onto debris.
    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)
    assert reason is None and path is not None
    strays = list(tmp_path.glob("*.backup-staging-*"))
    assert not strays, f"staging debris survived: {strays}"


# ---------------------------------------------------------------------------
# Orphaned sidecars left by a hard kill mid-PUBLISH (not mid-copy).
#
# ``test_publication_failure_leaves_no_countable_partial_bundle`` above only
# exercises a *raising* os.replace() failure, where the except-block rolls
# back every already-published destination. A hard kill (SIGKILL, OOM,
# power loss) between two os.replace() calls raises nothing — the process
# just stops — so the rollback never runs and a published sidecar can be
# left with no matching main backup. These tests exercise that shape
# directly: plant the orphan a hard kill would leave, then assert the next
# pass's sweep removes it.
# ---------------------------------------------------------------------------


def test_orphaned_published_sidecar_is_swept_on_next_pass(tmp_path):
    """A sidecar promoted to its final name with no matching main backup
    (a hard kill between the sidecar and main os.replace() calls) must be
    swept on the next pass — it is never counted by
    ``_existing_malformed_backups`` (which excludes sidecar suffixes), so
    without an explicit sweep it survives forever.
    """
    db = _damaged_db(tmp_path, size=20_000)
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()

    # Plant an orphan exactly as a hard kill would leave one: a published
    # sidecar (final name, no staging/incomplete marker) with no main file.
    orphan = db.with_name(f"{db.name}.malformed-backup-20260101_000000-wal")
    orphan.write_bytes(os.urandom(1_000))
    assert orphan.exists()
    assert not orphan.with_name(orphan.name[: -len("-wal")]).exists()

    # The orphan must never be countable as a backup (root cause of why it
    # would otherwise survive every prune pass).
    assert not _existing_malformed_backups(db)

    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)

    assert reason is None and path is not None
    assert not orphan.exists(), "orphaned published sidecar was not swept"


def test_orphaned_sidecar_sweep_does_not_touch_a_healthy_backup(tmp_path):
    """The sweep must only remove a sidecar whose main file is ABSENT — a
    sidecar belonging to a genuine, complete forensic backup must survive.
    """
    db = _damaged_db(tmp_path, size=20_000)
    db.with_name(db.name + "-wal").write_bytes(os.urandom(5_000))
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()

    with patch("shutil.disk_usage", return_value=roomy):
        first, reason = _backup_db_file(db)
    assert reason is None and first is not None
    first_sidecar = first.with_name(first.name + "-wal")
    assert first_sidecar.exists()

    # A second pass must not disturb the first backup's sidecar — its main
    # file is present, so the sweep must leave it alone.
    with patch("shutil.disk_usage", return_value=roomy):
        second, reason = _backup_db_file(db)
    assert reason is None and second is not None
    assert first.exists(), "sweep removed a healthy backup's main file"
    assert first_sidecar.exists(), "sweep removed a healthy backup's sidecar"


def test_orphaned_sidecar_sweep_covers_all_sidecar_suffixes(tmp_path):
    """The sweep must cover every sidecar suffix, not just ``-wal``."""
    db = _damaged_db(tmp_path, size=20_000)
    roomy = type(
        "Usage", (), {"total": 500_000_000_000, "used": 0, "free": 400_000_000_000}
    )()

    orphans = [
        db.with_name(f"{db.name}.malformed-backup-20260101_000000{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
    ]
    for orphan in orphans:
        orphan.write_bytes(os.urandom(100))

    with patch("shutil.disk_usage", return_value=roomy):
        path, reason = _backup_db_file(db)

    assert reason is None and path is not None
    for orphan in orphans:
        assert not orphan.exists(), f"orphan survived: {orphan.name}"
