"""Tests for the WAL→DELETE journal-mode fallback on NFS / SMB / FUSE.

When ``PRAGMA journal_mode=WAL`` raises ``OperationalError("locking protocol")``
(SQLITE_PROTOCOL — typical on NFS/SMB), Hermes must fall back to
``journal_mode=DELETE`` so ``state.db`` / ``kanban.db`` remain usable.

Without this fallback, users on NFS-mounted ``HERMES_HOME`` silently lose
``/resume``, ``/title``, ``/history``, ``/branch``, session search, and the
kanban dispatcher — because ``SessionDB()`` init propagates the error and
every caller swallows it, leaving ``_session_db = None``.

See: https://www.sqlite.org/wal.html — "WAL does not work over a network
filesystem".
"""

import sqlite3
from unittest.mock import patch

import pytest

import hermes_state
from hermes_state import (
    SessionDB,
    apply_wal_with_fallback,
    format_session_db_unavailable,
    get_last_init_error,
)


# ``sqlite3.Connection.execute`` is a C-level slot and can't be monkeypatched
# directly (``'sqlite3.Connection' object attribute 'execute' is read-only``).
# A factory-built subclass lets us intercept journal_mode=WAL per-test with
# its own mutable counter, avoiding the xdist-parallel class-state race.
def _make_blocking_factory(reason: str, attempt_counter: list):
    """Return a sqlite3.Connection subclass that raises on PRAGMA journal_mode=WAL."""

    class _WalBlockingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                attempt_counter[0] += 1
                raise sqlite3.OperationalError(reason)
            return super().execute(sql, *args, **kwargs)

    return _WalBlockingConnection


def _open_blocking(path, reason="locking protocol", **kwargs):
    """Open a connection whose WAL pragma raises ``reason``.

    Returns ``(conn, attempt_counter_list)`` so callers can assert how many
    times WAL was attempted.
    """
    attempts = [0]
    factory = _make_blocking_factory(reason, attempts)
    return sqlite3.connect(str(path), factory=factory, **kwargs), attempts


@pytest.fixture(autouse=True)
def _reset_last_init_error():
    """Reset the module-global last-error before and after each test."""
    hermes_state._set_last_init_error(None)
    yield
    hermes_state._set_last_init_error(None)


@pytest.fixture(autouse=True)
def _reset_wal_fallback_warned_paths():
    """Reset the WAL-fallback warned-paths set so dedup doesn't leak between tests."""
    hermes_state._wal_fallback_warned_paths.clear()
    yield
    hermes_state._wal_fallback_warned_paths.clear()


class TestApplyWalWithFallback:
    def test_succeeds_on_local_fs(self, tmp_path):
        """Happy path: WAL works on a normal filesystem."""
        conn = sqlite3.connect(str(tmp_path / "ok.db"), isolation_level=None)
        mode = apply_wal_with_fallback(conn)
        assert mode == "wal"
        cur = conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0].lower() == "wal"
        conn.close()

    def test_falls_back_to_delete_on_locking_protocol(self, tmp_path, caplog):
        """NFS-style ``locking protocol`` error → DELETE mode + one WARNING."""
        conn, _ = _open_blocking(tmp_path / "nfs.db", isolation_level=None)
        with caplog.at_level("WARNING", logger="hermes_state"):
            mode = apply_wal_with_fallback(conn, db_label="test.db")

        assert mode == "delete"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "test.db" in msg
        assert "journal_mode=DELETE" in msg
        assert "locking protocol" in msg

        # Post-fallback the DB is still usable for real writes
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        assert list(conn.execute("SELECT x FROM t"))[0][0] == 1
        conn.close()

    def test_falls_back_on_not_authorized(self, tmp_path):
        """Some FUSE mounts block WAL pragma outright ('not authorized')."""
        conn, _ = _open_blocking(
            tmp_path / "fuse.db", reason="not authorized", isolation_level=None
        )
        mode = apply_wal_with_fallback(conn)
        assert mode == "delete"
        conn.close()

    def test_falls_back_on_disk_io_error(self, tmp_path):
        """Flaky network FS → disk I/O error → still fall back."""
        conn, _ = _open_blocking(
            tmp_path / "flaky.db", reason="disk I/O error", isolation_level=None
        )
        mode = apply_wal_with_fallback(conn)
        assert mode == "delete"
        conn.close()

    def test_reraises_unrelated_operational_error(self, tmp_path):
        """Non-WAL-compat errors must NOT be silently swallowed by the fallback."""
        conn, _ = _open_blocking(
            tmp_path / "other.db",
            reason="no such table: nope",
            isolation_level=None,
        )
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            apply_wal_with_fallback(conn)
        conn.close()

    def test_warning_deduplicated_per_db_label(self, tmp_path, caplog):
        """Repeated calls with the same db_label log exactly ONE warning.

        Prevents log spam when NFS users run kanban (which opens a fresh
        connection on every operation — see hermes_cli/kanban_db.py).
        Regression guard: the fix for #22032 ran apply_wal_with_fallback()
        on every kb.connect() call; without dedup, errors.log fills with
        hundreds of identical warnings per hour.
        """
        with caplog.at_level("WARNING", logger="hermes_state"):
            # Three separate connections to "the same DB" via the same label
            for i in range(3):
                conn, _ = _open_blocking(
                    tmp_path / f"dup-{i}.db", isolation_level=None
                )
                mode = apply_wal_with_fallback(conn, db_label="shared.db")
                assert mode == "delete"
                conn.close()

        # Exactly one warning across all three calls
        warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "shared.db" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            f"Expected 1 deduplicated warning, got {len(warnings)}: "
            f"{[r.getMessage() for r in warnings]}"
        )

    def test_warning_fires_independently_per_db_label(self, tmp_path, caplog):
        """Different db_labels each get their own one warning (not globally dedup'd)."""
        with caplog.at_level("WARNING", logger="hermes_state"):
            conn1, _ = _open_blocking(tmp_path / "a.db", isolation_level=None)
            apply_wal_with_fallback(conn1, db_label="state.db")
            conn1.close()

            conn2, _ = _open_blocking(tmp_path / "b.db", isolation_level=None)
            apply_wal_with_fallback(conn2, db_label="kanban.db")
            conn2.close()

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        labels_warned = {
            lbl for r in warnings for lbl in ("state.db", "kanban.db")
            if lbl in r.getMessage()
        }
        assert labels_warned == {"state.db", "kanban.db"}, (
            f"Each db_label should warn once; got {labels_warned}"
        )


class TestGetLastInitError:
    def test_none_on_successful_init(self, tmp_path):
        """Happy-path SessionDB init does NOT clear a stale error from a prior thread.

        We deliberately don't clear on success so that in multi-threaded
        callers (gateway / web_server per-request SessionDB()), a concurrent
        successful open racing past a different thread's failure won't
        erase the cause string the failing thread's /resume is about to
        format.  The caller or test fixture is responsible for explicitly
        calling _set_last_init_error(None) to reset.
        """
        # Autouse fixture starts at None — success-path leaves it None
        db = SessionDB(db_path=tmp_path / "ok.db")
        try:
            assert get_last_init_error() is None
        finally:
            db.close()

    def test_success_does_not_clear_prior_error(self, tmp_path):
        """Thread-safety guard: a successful init must not erase a pre-existing error.

        Simulates the multi-threaded race: thread A fails, records cause;
        thread B succeeds concurrently.  thread A's /resume handler must
        still see A's cause — not B's None.
        """
        hermes_state._set_last_init_error("OperationalError: locking protocol")
        # Now a "successful" init happens on another path — must NOT clear
        db = SessionDB(db_path=tmp_path / "ok2.db")
        try:
            assert get_last_init_error() == "OperationalError: locking protocol"
        finally:
            db.close()

    def test_captures_cause_on_failed_init(self, tmp_path):
        """When SessionDB() raises, the cause is preserved for slash commands.

        Simulates a filesystem where BOTH WAL and DELETE journal modes fail —
        e.g. a read-only mount where no ``PRAGMA journal_mode=X`` works.  The
        fallback tries DELETE and also gets rejected; the exception bubbles
        out of ``SessionDB.__init__`` and the cause is captured.
        """
        target = tmp_path / "broken.db"
        real_connect = sqlite3.connect

        class _BothPragmasFailConnection(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):  # type: ignore[override]
                if "journal_mode" in sql.lower():
                    raise sqlite3.OperationalError(
                        "locking protocol: read-only filesystem"
                    )
                return super().execute(sql, *args, **kwargs)

        def gated_connect(*args, **kwargs):
            return real_connect(str(target), factory=_BothPragmasFailConnection, **kwargs)

        with patch("hermes_state.sqlite3.connect", side_effect=gated_connect):
            with pytest.raises(sqlite3.OperationalError):
                SessionDB(db_path=target)

        cause = get_last_init_error()
        assert cause is not None
        assert "OperationalError" in cause
        assert "locking protocol" in cause


class TestFormatSessionDbUnavailable:
    def test_bare_message_when_no_cause(self):
        """No init error recorded → generic message."""
        hermes_state._set_last_init_error(None)
        assert format_session_db_unavailable() == "Session database not available."

    def test_includes_cause(self):
        """Cause is surfaced for slash-command error strings."""
        hermes_state._set_last_init_error("OperationalError: generic SQLite error")
        msg = format_session_db_unavailable()
        assert "generic SQLite error" in msg
        assert msg.startswith("Session database not available:")
        assert msg.endswith(".")

    def test_adds_nfs_hint_for_locking_protocol(self):
        """Locking-protocol cause gets an NFS/SMB pointer for the user."""
        hermes_state._set_last_init_error("OperationalError: locking protocol")
        msg = format_session_db_unavailable()
        assert "locking protocol" in msg
        assert "NFS/SMB" in msg
        assert "sqlite.org/wal.html" in msg

    def test_custom_prefix(self):
        """Callers can customize the prefix for context-specific messages."""
        hermes_state._set_last_init_error("OperationalError: locking protocol")
        msg = format_session_db_unavailable(prefix="Cannot /resume")
        assert msg.startswith("Cannot /resume:")


class TestSessionDbUsesWalFallback:
    def test_sessiondb_works_when_wal_unavailable(self, tmp_path):
        """E2E: SessionDB initializes and performs a write on a WAL-blocked FS."""
        target = tmp_path / "nfs_style.db"

        real_connect = sqlite3.connect
        attempts = [0]
        factory = _make_blocking_factory("locking protocol", attempts)

        def gated_connect(*args, **kwargs):
            return real_connect(str(target), factory=factory, **kwargs)

        with patch("hermes_state.sqlite3.connect", side_effect=gated_connect):
            db = SessionDB(db_path=target)

        try:
            # WAL was attempted and rejected — fallback kicked in
            assert attempts[0] >= 1, (
                "WAL pragma was never executed — check the patch target"
            )
            # SessionDB is usable end-to-end: create a session, read it back
            db.create_session(session_id="s1", source="cli", model="test")
            sess = db.get_session("s1")
            assert sess is not None
            assert sess["source"] == "cli"
            # No init error was recorded since init succeeded via the fallback
            assert get_last_init_error() is None
        finally:
            db.close()
