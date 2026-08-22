"""The post-update state.db corruption message must name a repair (#88252).

`hermes update` verifies state.db afterwards and, when the check fails, tries
to restore a whole-database snapshot.  When there is no usable snapshot it
used to stop there — the reporter saw

    ⚠ state.db is corrupted after update: integrity check failed:
      malformed inverted index for FTS5 table main.messages_fts_trigram
      ⚠ No pre-update snapshot was taken

on every update, concluded their history was damaged, and eventually ran the
FTS5 ``'rebuild'`` by hand.  It was not damaged: that message names a derived
search index, and ``hermes sessions repair`` already rebuilds it in place.
"""
from __future__ import annotations

import inspect
import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import backup, update_cmd


# The exact string PRAGMA integrity_check produced on the reporter's database
# (Windows 11 26200, Hermes 0.20.2, state.db ~207 MB), as quoted in #88252.
REPORTED_MESSAGE = (
    "integrity check failed: malformed inverted index for FTS5 table "
    "main.messages_fts_trigram"
)


class TestDamageClassification:
    """Only FTS damage may be described as leaving the transcript intact."""

    @pytest.mark.parametrize(
        "message",
        [
            REPORTED_MESSAGE,
            "malformed inverted index for FTS5 table main.messages_fts",
            # The newer-SQLite wording of the same class, already recognised
            # elsewhere by SessionDB._is_fts_write_corruption_error.
            'fts5: corrupt structure record for table "messages_fts"',
            # Callers hand us whatever verify_sqlite_integrity returned, and
            # SQLite's own casing has changed between releases.
            "MALFORMED INVERTED INDEX FOR FTS5 TABLE main.messages_fts_cjk",
        ],
    )
    def test_fts_damage_is_recognised(self, message):
        assert update_cmd._state_db_damage_is_fts_only(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            # Genuine b-tree damage in a canonical table — the transcript
            # really may be gone, so we must not say otherwise.
            "row 12 missing from index sqlite_autoindex_sessions_1",
            "wrong # of entries in index idx_messages_session",
            "integrity check failed: database disk image is malformed",
            "header check failed: not a database",
            "unknown error",
            "",
        ],
    )
    def test_non_fts_damage_is_not_claimed_as_fts(self, message):
        assert update_cmd._state_db_damage_is_fts_only(message) is False

    @pytest.mark.parametrize(
        "message",
        [
            # Both kinds at once, in either order.  verify_sqlite_integrity
            # joins every finding PRAGMA integrity_check returned into one
            # string, so this is what a doubly-damaged database looks like
            # by the time it reaches us -- not a hypothetical.
            "integrity check failed: malformed inverted index for FTS5 table "
            "main.messages_fts_trigram; row 12 missing from index "
            "sqlite_autoindex_sessions_1",
            "integrity check failed: row 12 missing from index "
            "sqlite_autoindex_sessions_1; malformed inverted index for FTS5 "
            "table main.messages_fts_trigram",
            # The page damage is in the middle of a run of FTS findings,
            # which is where an "is an FTS phrase present" test is blindest.
            "integrity check failed: malformed inverted index for FTS5 table "
            "main.messages_fts; wrong # of entries in index "
            "idx_messages_session; malformed inverted index for FTS5 table "
            "main.messages_fts_cjk",
        ],
    )
    def test_mixed_damage_is_not_claimed_as_fts_only(self, message):
        """The false-reassurance entry path this classifier must not have.

        Telling someone whose b-tree lost rows that their messages are
        intact is worse than saying nothing, so a report that names any
        non-FTS finding falls through to the generic hint no matter how
        many FTS findings accompany it.
        """
        assert update_cmd._state_db_damage_is_fts_only(message) is False

    def test_a_truncated_report_is_not_claimed_as_fts_only(self):
        """A quoted-prefix report cannot be classified honestly.

        ``verify_sqlite_integrity`` quotes the first
        ``INTEGRITY_CHECK_MAX_REPORTED_ERRORS`` findings and discloses the
        rest as a count.  The finding that did not fit is exactly the one
        that could contradict the ones that did, so the reassuring claim is
        refused even though every *visible* finding is FTS.
        """
        message = (
            "integrity check failed: "
            + "; ".join(
                "malformed inverted index for FTS5 table main.messages_fts"
                for _ in range(backup.INTEGRITY_CHECK_MAX_REPORTED_ERRORS)
            )
            + f"; (3 {backup.INTEGRITY_CHECK_OMITTED_SUFFIX})"
        )

        assert update_cmd._state_db_damage_is_fts_only(message) is False

    def test_a_complete_all_fts_report_is_still_recognised(self):
        """The tightening must not cost the case the hint exists for.

        Several FTS indexes damaged together is the common shape of #88252
        (trigram, cjk and the base index are separate FTS5 tables), and it
        is still fully recoverable by a rebuild.
        """
        message = (
            "integrity check failed: malformed inverted index for FTS5 table "
            "main.messages_fts; malformed inverted index for FTS5 table "
            "main.messages_fts_trigram; malformed inverted index for FTS5 "
            "table main.messages_fts_cjk"
        )

        assert update_cmd._state_db_damage_is_fts_only(message) is True

    def test_generic_malformed_image_is_not_treated_as_fts(self):
        """The load-bearing negative case.

        ``database disk image is malformed`` is what a corrupt FTS shadow
        table raises on *older* SQLite builds, which is why
        ``is_malformed_db_error`` accepts it — but it is equally what real
        page damage raises, and integrity_check offers no way to tell them
        apart from the string alone.  Reassuring a user with genuine page
        damage that their messages are fine is a worse failure than saying
        nothing, so this class deliberately falls through to the generic
        hint.
        """
        assert (
            update_cmd._state_db_damage_is_fts_only("database disk image is malformed")
            is False
        )


class TestIntegrityReportTruncationIsDisclosed:
    """The producer must keep telling the classifier when it held back.

    ``_state_db_damage_is_fts_only`` refuses to reassure on a report that
    discloses omitted findings.  That guard is only worth anything while
    ``verify_sqlite_integrity`` actually discloses them -- a silent
    ``[:5]`` would hand the classifier a message that looks complete and
    quietly restore the over-claim.
    """

    def _report(self, errors):
        rows = [(e,) for e in errors]

        class _Cursor:
            def fetchall(self_inner):
                return rows

        class _Conn:
            def execute(self_inner, _sql):
                return _Cursor()

            def close(self_inner):
                pass

        return _Conn()

    def test_a_short_report_discloses_the_count(self, tmp_path, monkeypatch):
        db = tmp_path / "state.db"
        db.write_bytes(backup._SQLITE_HEADER + b"\x00" * 4096)
        errors = [f"row {n} missing from index idx_messages" for n in range(9)]
        monkeypatch.setattr(
            backup.sqlite3, "connect", lambda *a, **k: self._report(errors)
        )

        message = backup.verify_sqlite_integrity(db)["message"]

        omitted = len(errors) - backup.INTEGRITY_CHECK_MAX_REPORTED_ERRORS
        assert f"({omitted} {backup.INTEGRITY_CHECK_OMITTED_SUFFIX})" in message
        assert update_cmd._state_db_damage_is_fts_only(message) is False

    def test_a_complete_report_says_nothing_about_omissions(
        self, tmp_path, monkeypatch
    ):
        db = tmp_path / "state.db"
        db.write_bytes(backup._SQLITE_HEADER + b"\x00" * 4096)
        errors = [
            "malformed inverted index for FTS5 table main.messages_fts",
            "malformed inverted index for FTS5 table main.messages_fts_cjk",
        ]
        monkeypatch.setattr(
            backup.sqlite3, "connect", lambda *a, **k: self._report(errors)
        )

        message = backup.verify_sqlite_integrity(db)["message"]

        assert backup.INTEGRITY_CHECK_OMITTED_SUFFIX not in message
        # And the round trip still reaches the reassuring hint.
        assert update_cmd._state_db_damage_is_fts_only(message) is True


class TestHintOutput:
    """What the user is actually told once auto-restore is not an option."""

    def test_fts_hint_names_the_command_and_reassures(self, capsys):
        update_cmd._print_state_db_repair_hint(REPORTED_MESSAGE)
        out = capsys.readouterr().out

        assert "hermes sessions repair" in out
        assert "intact" in out
        assert "FTS5" in out

    def test_generic_hint_names_the_command_without_reassuring(self, capsys):
        update_cmd._print_state_db_repair_hint(
            "row 12 missing from index sqlite_autoindex_sessions_1"
        )
        out = capsys.readouterr().out

        assert "hermes sessions repair" in out
        # No promise about the data: this class can genuinely have lost rows.
        assert "intact" not in out

    @pytest.mark.parametrize(
        "message",
        [REPORTED_MESSAGE, "row 12 missing from index sqlite_autoindex_sessions_1"],
    )
    def test_hint_never_claims_to_have_done_anything(self, capsys, message):
        """A hint, never a receipt.

        An update must not acquire a write lock on a database the user has
        not asked it to rewrite, so nothing here repairs anything.  If a
        later change makes it act, this test should fail and force the
        wording — and the reasoning — to be revisited.
        """
        update_cmd._print_state_db_repair_hint(message)
        out = capsys.readouterr().out.lower()

        for claim in ("repaired", "rebuilt", "restored", "fixed"):
            assert claim not in out


class TestBothUpdatePathsAreWired:
    """The two post-update checks must both reach the hint.

    ``_cmd_update_impl`` and ``_update_via_zip`` are the git and zip update
    flows; each verifies state.db afterwards and each could reach the dead
    end.  Neither is callable in a test — they drive a whole update — so
    this asserts on their source, the same technique used across
    ``tests/hermes_cli`` for logic buried inside long command functions.
    """

    @pytest.mark.parametrize(
        "func",
        [update_cmd._cmd_update_impl, update_cmd._update_via_zip],
        ids=["git-path", "zip-path"],
    )
    def test_path_prints_the_hint_when_nothing_was_restored(self, func):
        source = inspect.getsource(func)

        assert "_print_state_db_repair_hint(" in source, (
            f"{func.__name__} reports state.db corruption but never names a repair"
        )
        assert "if not _state_restored:" in source, (
            f"{func.__name__} must gate the hint on the restore having failed"
        )

    @pytest.mark.parametrize(
        "func",
        [update_cmd._cmd_update_impl, update_cmd._update_via_zip],
        ids=["git-path", "zip-path"],
    )
    def test_path_records_a_successful_restore(self, func):
        """Otherwise a repaired-by-restore database still gets the hint."""
        source = inspect.getsource(func)

        assert "_state_restored = False" in source
        assert "_restore_state_db_from_snapshot(" in source, (
            f"{func.__name__} must go through the shared restore helper"
        )
        assert "_state_restored = _outcome" in source, (
            f"{func.__name__} ignores the restore outcome, so a successful "
            "restore would still print the repair hint"
        )


def _make_db(path):
    """A small but genuinely valid SQLite file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO messages (body) VALUES ('kept')")
        conn.commit()
    finally:
        conn.close()
    return path


class TestRestoreFromSnapshot:
    """The half both update flows share, tested on real files.

    The flows disagree about which snapshot to trust, but once either has a
    candidate they do the same four things, and the ordering of those four
    is load-bearing: verifying the copy *after* writing it is what stops a
    corrupt snapshot from replacing a corrupt database with another one.
    """

    def test_a_good_snapshot_is_copied_and_confirmed(self, tmp_path, capsys):
        state = tmp_path / "state.db"
        state.write_bytes(b"not a database at all")
        snap = _make_db(tmp_path / "snap" / "state.db")

        outcome = update_cmd._restore_state_db_from_snapshot(
            snap, state, "snapshot 20260822-0100"
        )

        assert outcome is True
        assert state.read_bytes() == snap.read_bytes()
        assert "snapshot 20260822-0100" in capsys.readouterr().out

    def test_a_corrupt_snapshot_is_left_alone_and_reported_as_no_attempt(
        self, tmp_path, capsys
    ):
        """None, not False: the ZIP path keeps walking back on this."""
        state = _make_db(tmp_path / "state.db")
        original = state.read_bytes()
        snap = tmp_path / "snap" / "state.db"
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(b"garbage" * 64)

        outcome = update_cmd._restore_state_db_from_snapshot(
            snap, state, "snapshot 20260822-0100"
        )

        assert outcome is None
        assert state.read_bytes() == original
        # Nothing was attempted, so the caller owns the explanation.
        assert capsys.readouterr().out == ""

    def test_a_failed_copy_is_an_attempt_that_did_not_work(
        self, tmp_path, capsys, monkeypatch
    ):
        state = tmp_path / "state.db"
        state.write_bytes(b"not a database at all")
        snap = _make_db(tmp_path / "snap" / "state.db")

        def _boom(*_args, **_kwargs):
            raise OSError("Errno 13 Permission denied")

        monkeypatch.setattr(shutil, "copy2", _boom)

        outcome = update_cmd._restore_state_db_from_snapshot(
            snap, state, "snapshot 20260822-0100"
        )

        assert outcome is False
        assert "Permission denied" in capsys.readouterr().out

    def test_a_copy_that_lands_corrupt_is_not_reported_as_restored(
        self, tmp_path, capsys, monkeypatch
    ):
        """The re-verify is the point of this helper.

        A snapshot can pass its own check and still not survive the copy
        (a truncated write, a full disk).  Announcing a restore that did
        not happen is the same false reassurance the repair hint exists to
        avoid, one layer down.
        """
        state = tmp_path / "state.db"
        state.write_bytes(b"not a database at all")
        snap = _make_db(tmp_path / "snap" / "state.db")

        def _truncating_copy(_src, dst):
            Path(dst).write_bytes(b"SQLite format 3\x00" + b"\x00" * 16)

        monkeypatch.setattr(shutil, "copy2", _truncating_copy)

        outcome = update_cmd._restore_state_db_from_snapshot(
            snap, state, "snapshot 20260822-0100"
        )

        assert outcome is False
        out = capsys.readouterr().out
        assert "Auto-restore FAILED" in out
        assert "Auto-restored" not in out
