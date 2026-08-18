"""state.db must not run in-file repair against a file that was replaced (#89332).

`SessionDB._execute_write` has a three-rung recovery ladder for corruption-class
errors: reopen-and-retry, a one-shot in-place FTS rebuild, and finally detaching
the FTS indexes so canonical writes can continue. Every rung assumes the damage
is *inside the file we opened*.

When something replaces `state.db` at the same path -- a restore script, a
maintenance job, any write-temp-then-rename -- that assumption is false, and all
three rungs are guaranteed to fail. The third is not inert: it drops the FTS sync
triggers, so rows written afterwards create an index gap. #89332 reports what that
costs in practice: 17 minutes of every `append_message` failing, with nothing in
the log naming the actual cause.

The reconnect rung's own docstring already anticipated "the backing file was
replaced/truncated by a sibling process" -- it was simply wired to only one of the
two error classes that produces.

These tests pin the guard and, just as importantly, pin its fail-open contract:
a guard that fires on absent evidence would turn a healthy store into the outage
it exists to prevent.

SCOPE NOTE, deliberately pinned below: file identity is `(st_dev, st_ino)`, so
this catches the rename/replace variant. The truncate-in-place variant (`cp` onto
the live path, same inode, new content) keeps its identity and is NOT caught --
that needs an in-file generation stamp, which changes the on-disk artifact and is
a maintainer decision. See `test_truncate_in_place_is_not_detected_by_identity`.
"""

import os
import sqlite3

import pytest

from hermes_state import SessionDB, _stat_file_identity


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _replace_by_rename(db_path):
    """Swap in a different file at the same path (new inode).

    Only usable with no live handle on *db_path*: Windows refuses ``os.replace``
    onto an open file. The tests that need a live SessionDB use
    ``_simulate_swap`` instead.
    """
    other = db_path.parent / "other.db"
    raw = sqlite3.connect(str(other))
    raw.execute("CREATE TABLE t (x)")
    raw.commit()
    raw.close()
    os.replace(str(other), str(db_path))


def _simulate_swap(db):
    """Put *db* in the state a replacement leaves it in.

    A swap is observable to the guard as exactly one thing: the identity
    recorded at open no longer matches the identity on disk. That the OS
    really produces that divergence is proven separately and for real in
    ``TestFileIdentityHelper.test_replacing_the_file_changes_its_identity``;
    driving it from the recorded side here keeps the ladder tests portable,
    since Windows will not let a file be renamed over while a SessionDB holds
    it open.
    """
    real = _stat_file_identity(db.db_path)
    assert real is not None, "fixture database should have a readable identity"
    db._file_identity = (real[0], real[1] + 1)


class TestFileIdentityHelper:
    """`_stat_file_identity` -- the fail-open primitive everything rests on."""

    def test_a_real_file_has_an_identity(self, tmp_path):
        p = tmp_path / "f"
        p.write_bytes(b"x")
        assert _stat_file_identity(p) is not None

    def test_a_missing_path_has_no_opinion(self, tmp_path):
        assert _stat_file_identity(tmp_path / "nope") is None

    def test_the_same_file_keeps_its_identity_across_writes(self, tmp_path):
        """Identity is of the FILE, not its contents.

        If ordinary writes moved it, the guard would fire on every busy
        database -- the exact false positive that would make this change worse
        than the bug.
        """
        p = tmp_path / "f"
        p.write_bytes(b"x")
        before = _stat_file_identity(p)
        p.write_bytes(b"completely different contents, much longer")
        assert _stat_file_identity(p) == before

    def test_replacing_the_file_changes_its_identity(self, tmp_path):
        p = tmp_path / "f"
        p.write_bytes(b"x")
        before = _stat_file_identity(p)
        _replace_by_rename(p)
        assert _stat_file_identity(p) != before

    def test_a_filesystem_without_inodes_has_no_opinion(self, tmp_path, monkeypatch):
        """`st_ino == 0` happens on some network and FUSE mounts.

        Fail open. An unknowable identity must never read as "replaced".
        """
        p = tmp_path / "f"
        p.write_bytes(b"x")
        real = os.stat

        class _NoIno:
            st_dev = 1
            st_ino = 0

        monkeypatch.setattr(os, "stat", lambda *a, **k: _NoIno())
        try:
            assert _stat_file_identity(p) is None
        finally:
            monkeypatch.setattr(os, "stat", real)


class TestTheGuardFailsOpen:
    """Every unknown must answer "not replaced". This is the load-bearing half."""

    def test_no_recorded_identity_disarms_the_guard(self, db):
        db._file_identity = None
        assert db._backing_file_was_replaced() is False

    def test_an_unstatable_path_now_disarms_the_guard(self, db, monkeypatch):
        """A transiently missing path is not evidence of a swap.

        Answering True here would refuse repair on a database that is merely
        mid-rotation, converting a recoverable state into a hard failure.
        """
        monkeypatch.setattr(
            "hermes_state._stat_file_identity", lambda *a, **k: None
        )
        assert db._backing_file_was_replaced() is False

    def test_a_written_database_is_never_reported_as_replaced(self, db):
        """Ordinary write traffic must not look like a swap."""
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)",
                ("identity-guard-probe", "1"),
            )
        )
        assert db._backing_file_was_replaced() is False


class TestTheGuardFires:

    def test_a_replaced_file_is_detected(self, db):
        _simulate_swap(db)
        assert db._backing_file_was_replaced() is True

    def test_the_refusal_names_the_swap_and_the_original_error(self, db, caplog):
        _simulate_swap(db)
        exc = sqlite3.DatabaseError("database disk image is malformed")

        with caplog.at_level("ERROR", logger="hermes_state"):
            refused = db._refuse_repair_on_replaced_file(exc)

        assert refused is True
        blob = "\n".join(r.getMessage() for r in caplog.records)
        assert "REPLACED" in blob
        assert "database disk image is malformed" in blob, (
            "the original error must survive into the log -- #89332's whole "
            "complaint is that the cause was never named"
        )

    def test_a_healthy_database_is_not_refused(self, db):
        """Behaviour preservation: ordinary corruption still gets the ladder."""
        exc = sqlite3.DatabaseError("database disk image is malformed")
        assert db._refuse_repair_on_replaced_file(exc) is False


class TestTheLadderIsSkipped:
    """The point of the guard: no in-file repair on a file that isn't ours."""

    def test_no_repair_rung_runs_after_a_swap(self, db, monkeypatch):
        called = []

        def _rung(name):
            # One-shot: the real rungs are one-shot too, and a mock that
            # always says "retry" would turn a regression here into an
            # infinite retry loop instead of a failing assertion.
            def _fn(*_a, **_k):
                first = name not in called
                called.append(name)
                return first
            return _fn

        monkeypatch.setattr(db, "_reconnect_after_notadb", _rung("reconnect"))
        monkeypatch.setattr(db, "_try_runtime_fts_rebuild", _rung("fts_rebuild"))
        monkeypatch.setattr(db, "_enter_fts_fail_open", _rung("fail_open"))
        _simulate_swap(db)

        def _boom(conn):
            raise sqlite3.DatabaseError("database disk image is malformed")

        with pytest.raises(sqlite3.DatabaseError):
            db._execute_write(_boom)

        assert called == [], (
            "the swapped-file path must reach none of the three rungs; "
            f"it reached {called}"
        )

    def test_the_original_error_still_propagates(self, db):
        """Refusing to repair is not swallowing.

        The write must still fail -- it fails immediately and by name rather
        than after three futile attempts.
        """
        _simulate_swap(db)

        def _boom(conn):
            raise sqlite3.DatabaseError("database disk image is malformed")

        with pytest.raises(sqlite3.DatabaseError, match="malformed"):
            db._execute_write(_boom)

    def test_an_unswapped_database_still_reaches_the_ladder(self, db, monkeypatch):
        """The regression that would matter most.

        Refusing a legitimate FTS repair would be a worse bug than the one
        this fixes, so the untouched case must be provably unaffected.
        """
        called = []
        monkeypatch.setattr(
            db, "_try_runtime_fts_rebuild",
            lambda exc: called.append("fts_rebuild") or False,
        )
        monkeypatch.setattr(
            db, "_enter_fts_fail_open",
            lambda exc: called.append("fail_open") or False,
        )

        def _boom(conn):
            raise sqlite3.DatabaseError("database disk image is malformed")

        with pytest.raises(sqlite3.DatabaseError):
            db._execute_write(_boom)

        assert called == ["fts_rebuild", "fail_open"]


class TestIdentityIsRebaselined:

    def test_the_open_path_records_an_identity(self, db):
        assert db._file_identity == _stat_file_identity(db.db_path)

    def test_a_successful_reconnect_re_records_it(self, db):
        """Otherwise the swap stays "detected" forever.

        `_reconnect_after_notadb` exists precisely because the backing file may
        have been replaced, so the file it opens is the new baseline. Without
        the rebaseline, every later corruption error on a reconciled database
        would be blamed on a swap that has already been handled.
        """
        _simulate_swap(db)
        stale = db._file_identity
        assert db._backing_file_was_replaced() is True

        assert db._reconnect_after_notadb() is True

        assert db._file_identity != stale
        assert db._backing_file_was_replaced() is False


class TestScopeIsPinnedHonestly:

    def test_truncate_in_place_is_not_detected_by_identity(self, db):
        """`cp` onto the live path keeps the inode, so identity cannot see it.

        #89332 names both variants. This guard covers the rename/replace one.
        Covering the truncate-in-place one requires an in-file generation stamp
        (the issue suggests `PRAGMA application_id` / `user_version`), which
        changes the on-disk artifact every existing install carries -- a
        maintainer decision, deliberately not made here.

        This test exists so the limitation is a stated contract rather than an
        unnoticed hole: if a generation stamp is ever added, this test fails
        and should be rewritten, not deleted.
        """
        other = db.db_path.parent / "other.db"
        raw = sqlite3.connect(str(other))
        raw.execute("CREATE TABLE t (x)")
        raw.commit()
        raw.close()

        before = _stat_file_identity(db.db_path)
        # Truncate + rewrite in place: same inode, entirely new content.
        with open(other, "rb") as src, open(db.db_path, "r+b") as dst:
            dst.truncate(0)
            dst.write(src.read())

        assert _stat_file_identity(db.db_path) == before
        assert db._backing_file_was_replaced() is False
