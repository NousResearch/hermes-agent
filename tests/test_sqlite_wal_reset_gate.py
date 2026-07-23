"""SQLite WAL-reset vulnerability gate (issue #69784).

Hermes must not *enable* multi-process WAL on SQLite builds that still contain
the upstream WAL-reset corruption bug:
https://sqlite.org/wal.html#walresetbug

Existing on-disk WAL databases are left alone (no live downgrade).
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import (
    apply_wal_with_fallback,
    is_sqlite_wal_reset_vulnerable,
    sqlite_source_id,
)


@pytest.fixture(autouse=True)
def _reset_wal_reset_bug_warnings():
    hermes_state._wal_reset_bug_warned_paths.clear()
    yield
    hermes_state._wal_reset_bug_warned_paths.clear()


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

    def test_defaults_to_linked_library(self):
        assert isinstance(is_sqlite_wal_reset_vulnerable(), bool)


class TestApplyWalWalResetGate:
    def test_fresh_db_uses_delete_when_vulnerable(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(
            hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        conn = sqlite3.connect(str(tmp_path / "fresh.db"))
        with caplog.at_level("WARNING", logger="hermes_state"):
            mode = apply_wal_with_fallback(conn, db_label="fresh.db")
        assert mode == "delete"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        assert any("instead of enabling WAL" in r.getMessage() for r in caplog.records)
        conn.close()

    def test_existing_wal_left_alone_when_vulnerable(
        self, tmp_path, monkeypatch, caplog
    ):
        """Already-WAL DBs must not be live-downgraded under concurrent openers."""
        monkeypatch.setattr(
            hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
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

    def test_existing_wal_does_not_run_checkpoint_or_delete(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):  # type: ignore[override]
                self.executed.append(sql)
                return super().execute(sql, params)

        path = tmp_path / "trace_wal.db"
        with sqlite3.connect(str(path)) as seed:
            seed.execute("PRAGMA journal_mode=WAL")

        conn = _TracingConn(str(path))
        try:
            assert apply_wal_with_fallback(conn, db_label="trace_wal.db") == "wal"
        finally:
            conn.close()

        joined_lower = "\n".join(conn.executed).lower().replace(" ", "")
        assert "wal_checkpoint" not in joined_lower
        assert "journal_mode=delete" not in joined_lower

    def test_fixed_sqlite_still_enables_wal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
        )
        conn = sqlite3.connect(str(tmp_path / "fixed.db"))
        mode = apply_wal_with_fallback(conn, db_label="fixed.db")
        assert mode == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        conn.close()

    def test_warning_deduped_per_label(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(
            hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
        )
        with caplog.at_level("WARNING", logger="hermes_state"):
            for name in ("a.db", "a.db", "b.db"):
                conn = sqlite3.connect(str(tmp_path / name))
                apply_wal_with_fallback(conn, db_label=name)
                conn.close()
        warnings = [r for r in caplog.records if "WAL-reset" in r.getMessage()]
        assert len(warnings) == 2


def test_sqlite_source_id_non_empty_string():
    src = sqlite_source_id()
    assert isinstance(src, str)
    assert src


def test_doctor_warns_without_adding_issues(monkeypatch, tmp_path, capsys):
    """Vulnerable SQLite is warn-only in doctor — not a blocking issues[] entry."""
    from hermes_cli.doctor import run_doctor

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: True
    )
    monkeypatch.setattr(hermes_state, "sqlite_source_id", lambda: "testid-abc")
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
    # No longer appended to the blocking issues summary.
    assert "Linked SQLite is vulnerable" not in out
