"""``database.synchronous`` is configurable, validated, and floored on macOS.

Before this, `apply_database_pragmas()` accepted five sizing pragmas and no
durability one, and `_enforce_macos_synchronous_full()` returned early off
Darwin. So on Linux and Windows nothing in the process ever executed
`PRAGMA synchronous` against state.db, and the effective level was whichever
`SQLITE_DEFAULT_WAL_SYNCHRONOUS` the interpreter's SQLite happened to be
compiled with -- invisible from config, unpinnable, and different between
builds. See #90837, where three weeks of corruption forensics were carried out
under the stated assumption of `synchronous=FULL` on Ubuntu.
"""

import sqlite3
import sys

import pytest

import hermes_state
from hermes_state import (
    apply_database_pragmas,
    resolve_synchronous_level,
)


def _wal_conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "state.db"), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _level(conn):
    return conn.execute("PRAGMA synchronous").fetchone()[0]


def _config(monkeypatch, database_section):
    """Point apply_database_pragmas at an in-memory config.

    It imports `hermes_cli.config` lazily inside the function body, so the
    patch has to land on that module rather than on a name in this one.
    """
    import hermes_cli.config as config_mod

    cfg = {"database": database_section}
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda *a, **k: cfg)
    return cfg


class TestResolveSynchronousLevel:
    """The parser, in isolation -- a wrong answer here is a silent downgrade."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("OFF", 0),
            ("NORMAL", 1),
            ("FULL", 2),
            ("EXTRA", 3),
            ("full", 2),
            ("Full", 2),
            ("  FULL  ", 2),
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            ("2", 2),
            (" 2 ", 2),
        ],
    )
    def test_accepts_documented_spellings(self, raw, expected):
        assert resolve_synchronous_level(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "PARANOID",
            "",
            "   ",
            4,
            -1,
            "4",
            None,
            [],
            {},
            2.5,
            "2.0",
        ],
    )
    def test_rejects_everything_else(self, raw):
        assert resolve_synchronous_level(raw) is None

    def test_yaml_off_is_a_level_but_yaml_on_is_not(self):
        """`synchronous: off` is bool False in YAML and means OFF.

        `synchronous: on` is bool True and means nothing -- there is no
        durability level it could map to, so it must be rejected rather than
        coerced to 1 by int(True).
        """
        assert resolve_synchronous_level(False) == 0
        assert resolve_synchronous_level(True) is None


class TestAppliedFromConfig:
    def test_configured_level_reaches_the_connection(self, tmp_path, monkeypatch):
        _config(monkeypatch, {"synchronous": "NORMAL"})
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "linux")
            apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 1
        finally:
            conn.close()

    def test_full_reaches_the_connection(self, tmp_path, monkeypatch):
        """The #90837 case: an operator asking for FULL on Linux gets FULL.

        Fails without the patch -- the pragma was never executed at all, so the
        connection kept whatever the build's default was.
        """
        _config(monkeypatch, {"synchronous": "FULL"})
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "linux")
            conn.execute("PRAGMA synchronous=0")
            assert _level(conn) == 0
            apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 2
        finally:
            conn.close()

    def test_unset_leaves_the_level_alone(self, tmp_path, monkeypatch):
        """No config key must mean no write -- this is the default path."""
        _config(monkeypatch, {"wal_autocheckpoint": 1000})
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "linux")
            conn.execute("PRAGMA synchronous=0")
            apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 0
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1000
        finally:
            conn.close()

    def test_garbage_warns_and_changes_nothing(self, tmp_path, monkeypatch, caplog):
        """A typo must not fall through to a different durability level."""
        _config(monkeypatch, {"synchronous": "PARANOID"})
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "linux")
            conn.execute("PRAGMA synchronous=2")
            with caplog.at_level("WARNING"):
                apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 2
            assert "PARANOID" in caplog.text
        finally:
            conn.close()

    def test_the_other_pragmas_still_apply(self, tmp_path, monkeypatch):
        """Guardrail for #77630's five keys -- they share the same function."""
        _config(
            monkeypatch,
            {"synchronous": "FULL", "wal_autocheckpoint": 250, "mmap_size": 0},
        )
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "linux")
            apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 2
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 250
            assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 0
        finally:
            conn.close()


class TestMacOSFloor:
    """#64355 enforced FULL on Darwin. Config must not be able to undo it.

    `_enforce_macos_synchronous_full()` runs inside `apply_wal_with_fallback()`,
    which is earlier than `apply_database_pragmas()`, so without an explicit
    floor the config value would simply win by running last.
    """

    def test_lowering_below_full_is_refused_on_darwin(
        self, tmp_path, monkeypatch, caplog
    ):
        _config(monkeypatch, {"synchronous": "NORMAL"})
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "darwin")
            hermes_state._enforce_macos_synchronous_full(conn)
            assert _level(conn) == 2
            with caplog.at_level("WARNING"):
                apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 2, "config lowered the macOS btree protection"
            assert "macOS" in caplog.text
        finally:
            conn.close()

    def test_raising_above_full_is_allowed_on_darwin(self, tmp_path, monkeypatch):
        """The floor is a floor, not a pin -- EXTRA is strictly safer."""
        _config(monkeypatch, {"synchronous": "EXTRA"})
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "darwin")
            apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 3
        finally:
            conn.close()

    def test_the_floor_does_not_apply_off_darwin(self, tmp_path, monkeypatch):
        """Linux operators may deliberately choose NORMAL for write volume."""
        _config(monkeypatch, {"synchronous": "NORMAL"})
        conn = _wal_conn(tmp_path)
        try:
            monkeypatch.setattr(sys, "platform", "linux")
            conn.execute("PRAGMA synchronous=2")
            apply_database_pragmas(conn, db_label="state.db")
            assert _level(conn) == 1
        finally:
            conn.close()


def test_example_config_documents_the_key():
    """An unpinnable durability level is the bug; docs are half the fix."""
    from pathlib import Path

    text = Path(__file__).resolve().parents[1] / "cli-config.yaml.example"
    body = text.read_text(encoding="utf-8")
    assert "synchronous: FULL" in body
