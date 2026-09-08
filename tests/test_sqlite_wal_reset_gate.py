"""SQLite WAL-reset vulnerability gate (issue #69784).

Hermes must not *enable* multi-process WAL on SQLite builds that still contain
the upstream WAL-reset corruption bug:
https://sqlite.org/wal.html#walresetbug

Existing on-disk WAL databases are left alone (no live downgrade).
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import hermes_state
import hermes_state_wal
from hermes_state_wal import apply_wal_with_fallback, is_sqlite_wal_reset_vulnerable, sqlite_source_id


@pytest.fixture(autouse=True)
def _reset_wal_reset_bug_warnings():
    hermes_state_wal._wal_reset_bug_warned_paths.clear()
    yield
    hermes_state_wal._wal_reset_bug_warned_paths.clear()


class TestIsSqliteWalResetVulnerable:
    @pytest.mark.parametrize(
        "version_info,expected",
        [
            ((3, 6, 23), False),  # pre-WAL
            ((3, 7, 0), True),
            ((3, 44, 5), True),
            ((3, 44, 6), False),  # backport
            ((3, 44, 9), False),
            ((3, 45, 0), True),
            ((3, 46, 1), True),
            ((3, 50, 4), True),
            ((3, 50, 6), True),
            ((3, 50, 7), False),  # backport
            ((3, 50, 99), False),
            ((3, 51, 0), True),
            ((3, 51, 2), True),
            ((3, 51, 3), False),  # fixed line
            ((3, 52, 0), False),
        ],
    )
    def test_version_matrix(self, version_info, expected):
        assert is_sqlite_wal_reset_vulnerable(version_info) is expected



class TestApplyWalWalResetGate:
    def test_fresh_db_uses_delete_when_vulnerable(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        conn = sqlite3.connect(str(tmp_path / "fresh.db"))
        with caplog.at_level("WARNING", logger="hermes_state"):
            mode = apply_wal_with_fallback(conn, db_label="fresh.db")
        assert mode == "delete"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        assert any("instead of enabling WAL" in r.getMessage() for r in caplog.records)
        assert any(sys.executable in r.getMessage() for r in caplog.records)
        conn.close()

    def test_existing_wal_left_alone_when_vulnerable(
        self, tmp_path, monkeypatch, caplog
    ):
        """Already-WAL DBs must not be live-downgraded under concurrent openers."""
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        path = tmp_path / "prior_wal.db"
        seed = sqlite3.connect(str(path))
        try:
            seed.execute("PRAGMA journal_mode=WAL")
            seed.execute("CREATE TABLE t (x INTEGER)")
            seed.execute("INSERT INTO t VALUES (42)")
            seed.commit()
            assert seed.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            seed.close()

        conn = sqlite3.connect(str(path), timeout=30.0)
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                mode = apply_wal_with_fallback(conn, db_label="prior_wal.db")
            assert mode == "wal"
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            assert conn.execute("SELECT x FROM t").fetchone()[0] == 42
            assert any("already in WAL mode" in r.getMessage() for r in caplog.records)
            # Must not attempt a live journal_mode flip.
            assert not any(
                "instead of enabling WAL" in r.getMessage() for r in caplog.records
            )
        finally:
            conn.close()



    def test_warning_deduped_per_label(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        with caplog.at_level("WARNING", logger="hermes_state"):
            for name in ("a.db", "a.db", "b.db"):
                conn = sqlite3.connect(str(tmp_path / name))
                apply_wal_with_fallback(conn, db_label=name)
                conn.close()
        warnings = [r for r in caplog.records if "WAL-reset" in r.getMessage()]
        assert len(warnings) == 2


_HOLDER_SCRIPT = """
import sqlite3, sys, time, os
db, ready, done = sys.argv[1], sys.argv[2], sys.argv[3]
conn = sqlite3.connect(db, timeout=30.0)
conn.execute("INSERT INTO t VALUES (777)")
conn.commit()  # committed but (in WAL) very likely un-checkpointed
with open(ready, "w") as fh:
    fh.write("ready")
deadline = time.time() + 60
while not os.path.exists(done) and time.time() < deadline:
    time.sleep(0.05)
conn.close()
"""


class TestNoDowngradeUnderConcurrentOpeners:
    """The Aug 2026 state.db incident class: a vulnerable-SQLite process must
    never flip journal modes on a database it does not exclusively own.

    A concurrent WAL writer's committed-but-uncheckpointed transactions are
    destroyed by a live WAL→DELETE flip (observed: disk rows went 10 → 0
    while the writer's memory held 185)."""

    def test_second_process_opener_keeps_wal_when_vulnerable(
        self, tmp_path, monkeypatch, caplog
    ):
        """A REAL second process holds the WAL DB open while the vulnerable
        gate runs — WAL must be left in place and its committed rows survive.

        All blocked-state assertions run WHILE the holder owns the DB."""
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        db = tmp_path / "live_wal.db"
        seed = sqlite3.connect(str(db))
        try:
            seed.execute("PRAGMA journal_mode=WAL")
            seed.execute("CREATE TABLE t (x INTEGER)")
            seed.execute("INSERT INTO t VALUES (1)")
            seed.commit()
        finally:
            seed.close()

        ready = tmp_path / "holder.ready"
        done = tmp_path / "holder.done"
        holder = subprocess.Popen(
            [sys.executable, "-c", _HOLDER_SCRIPT, str(db), str(ready), str(done)]
        )
        try:
            deadline = time.time() + 30
            while not ready.exists() and time.time() < deadline:
                time.sleep(0.02)
            assert ready.exists(), "holder subprocess never became ready"

            conn = sqlite3.connect(str(db), timeout=30.0)
            try:
                with caplog.at_level("WARNING", logger="hermes_state"):
                    mode = apply_wal_with_fallback(conn, db_label="live_wal.db")
                # Asserted while the second opener still holds the DB:
                assert holder.poll() is None, "holder must still be alive here"
                assert mode == "wal"
                assert (
                    conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
                    == "wal"
                )
                # The concurrent opener's committed row must have survived.
                rows = {r[0] for r in conn.execute("SELECT x FROM t")}
                assert rows == {1, 777}
                assert not any(
                    "instead of enabling WAL" in r.getMessage()
                    for r in caplog.records
                )
            finally:
                conn.close()
        finally:
            done.write_text("done")
            holder.wait(timeout=30)

        check = sqlite3.connect(str(db))
        try:
            assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            assert {r[0] for r in check.execute("SELECT x FROM t")} == {1, 777}
        finally:
            check.close()

    def test_unreadable_mode_keeps_journal_mode_when_vulnerable(
        self, tmp_path, monkeypatch, caplog
    ):
        """An exclusive-locking holder blocks even the journal-mode read.

        Ownership is then not provably exclusive: the gate must leave the
        journal mode untouched instead of treating 'could not read the mode'
        as 'not WAL' and flipping anyway (the incident's exact confusion).
        Assertions run WHILE the holder's exclusive lock is live."""
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        db = tmp_path / "locked_wal.db"
        seed = sqlite3.connect(str(db))
        try:
            seed.execute("PRAGMA journal_mode=WAL")
            seed.execute("CREATE TABLE t (x INTEGER)")
            seed.execute("INSERT INTO t VALUES (1)")
            seed.commit()
        finally:
            seed.close()

        holder = sqlite3.connect(str(db))
        try:
            holder.execute("PRAGMA locking_mode=EXCLUSIVE")
            holder.execute("BEGIN IMMEDIATE")
            holder.execute("INSERT INTO t VALUES (2)")

            conn = sqlite3.connect(str(db), timeout=0.2)
            try:
                # Sanity: the probe really is blocked right now.
                with pytest.raises(sqlite3.OperationalError):
                    conn.execute("PRAGMA journal_mode").fetchone()
                with caplog.at_level("WARNING", logger="hermes_state"):
                    mode = apply_wal_with_fallback(conn, db_label="locked_wal.db")
                assert mode == "wal"
                assert any(
                    "concurrent openers" in r.getMessage() for r in caplog.records
                )
                assert not any(
                    "instead of enabling WAL" in r.getMessage()
                    for r in caplog.records
                )
            finally:
                conn.close()
        finally:
            holder.rollback()
            holder.close()

        check = sqlite3.connect(str(db))
        try:
            assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            check.close()

    def test_exclusively_owned_fresh_db_still_downgrades(
        self, tmp_path, monkeypatch, caplog
    ):
        """No concurrent openers → the vulnerable-SQLite DELETE gate still
        applies exactly as before."""
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        conn = sqlite3.connect(str(tmp_path / "exclusive.db"))
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                mode = apply_wal_with_fallback(conn, db_label="exclusive.db")
            assert mode == "delete"
            assert (
                conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
            )
            assert any(
                "instead of enabling WAL" in r.getMessage() for r in caplog.records
            )
        finally:
            conn.close()

    def test_flip_lock_conflict_leaves_mode_alone(self, tmp_path, caplog, monkeypatch):
        """If the DELETE flip itself hits another opener's lock (opener arrived
        between probe and flip), the gate returns the observed mode instead of
        raising or waiting the lock out."""
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )

        class _FlipLockedConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):  # type: ignore[override]
                if "journal_mode=delete" in sql.lower().replace(" ", ""):
                    raise sqlite3.OperationalError("database is locked")
                return super().execute(sql, *args, **kwargs)

        conn = sqlite3.connect(
            str(tmp_path / "race.db"), factory=_FlipLockedConnection
        )
        try:
            with caplog.at_level("WARNING", logger="hermes_state"):
                mode = apply_wal_with_fallback(conn, db_label="race.db")
            assert mode == "delete"  # observed pre-flip mode, not a forced flip
            assert any(
                "concurrent openers" in r.getMessage() for r in caplog.records
            )
        finally:
            conn.close()

    def test_configured_delete_refuses_when_probe_blocked(
        self, tmp_path, monkeypatch
    ):
        """Operator-configured DELETE on a non-vulnerable build must also
        refuse to downgrade when the mode probe is blocked by a concurrent
        opener's exclusive lock — raise, never flip blind."""
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable",
            lambda version_info=None: False,
        )
        monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "delete")
        db = tmp_path / "cfg_delete.db"
        seed = sqlite3.connect(str(db))
        try:
            seed.execute("PRAGMA journal_mode=WAL")
            seed.execute("CREATE TABLE t (x INTEGER)")
            seed.commit()
        finally:
            seed.close()

        holder = sqlite3.connect(str(db))
        try:
            holder.execute("PRAGMA locking_mode=EXCLUSIVE")
            holder.execute("BEGIN IMMEDIATE")
            holder.execute("INSERT INTO t VALUES (1)")

            conn = sqlite3.connect(str(db), timeout=0.2)
            try:
                with pytest.raises(
                    sqlite3.OperationalError, match="refusing to downgrade"
                ):
                    apply_wal_with_fallback(conn, db_label="cfg_delete.db")
            finally:
                conn.close()
        finally:
            holder.rollback()
            holder.close()

        check = sqlite3.connect(str(db))
        try:
            assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            check.close()

    def test_nfs_fallback_reraises_when_mode_unreadable(self, tmp_path, monkeypatch):
        """The filesystem-incompat fallback must not downgrade when the on-disk
        mode cannot be verified (possible concurrent openers)."""
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable",
            lambda version_info=None: False,
        )
        hermes_state_wal._wal_fallback_warned_paths.clear()

        class _LockedProbeConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):  # type: ignore[override]
                normalized = sql.lower().replace(" ", "")
                if "journal_mode=wal" in normalized:
                    raise sqlite3.OperationalError("locking protocol")
                if normalized == "pragmajournal_mode":
                    raise sqlite3.OperationalError("database is locked")
                return super().execute(sql, *args, **kwargs)

        conn = sqlite3.connect(
            str(tmp_path / "nfs.db"), factory=_LockedProbeConnection
        )
        try:
            with pytest.raises(sqlite3.OperationalError, match="locking protocol"):
                apply_wal_with_fallback(conn, db_label="nfs.db")
        finally:
            conn.close()




def test_doctor_warns_without_adding_issues(monkeypatch, tmp_path, capsys):
    """Vulnerable SQLite is warn-only in doctor — not a blocking issues[] entry."""
    from hermes_cli.doctor import run_doctor

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr(
        hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
    )
    monkeypatch.setattr(hermes_state_wal, "sqlite_source_id", lambda: "testid-abc")
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.50.4", raising=False)

    args = SimpleNamespace(fix=False, ack=None)
    try:
        run_doctor(args)
    except SystemExit:
        pass

    out = capsys.readouterr().out
    assert "SQLite" in out
    assert "3.50.4" in out
    assert "WAL-reset" in out
    assert "hermes update" in out
    # No longer appended to the blocking issues summary.
    assert "Linked SQLite is vulnerable" not in out
