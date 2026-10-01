"""Regression tests for raw lifecycle-guard reads of live SQLite files."""

from __future__ import annotations

from pathlib import Path

import cron.lifecycle_guard as lifecycle_guard
from hermes_cli.sqlite_safe_read import connect_tracked


def test_referenced_script_read_refuses_live_sqlite_connection(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    conn = connect_tracked(db)
    try:
        def must_not_open(*_args, **_kwargs):
            raise AssertionError("raw-opened a live SQLite database")

        monkeypatch.setattr(lifecycle_guard.os, "open", must_not_open)

        text, unsafe = lifecycle_guard._read_referenced_script(db)

        assert text is None
        assert unsafe is True
    finally:
        conn.close()


def test_referenced_script_read_preserves_normal_script_behavior(tmp_path):
    script = tmp_path / "script.sh"
    script.write_text("echo ok\n", encoding="utf-8")

    text, unsafe = lifecycle_guard._read_referenced_script(script)

    assert text == "echo ok\n"
    assert unsafe is False


def test_invalid_referenced_script_path_is_still_tolerated():
    text, unsafe = lifecycle_guard._read_referenced_script(
        Path("/tmp/hermes\x00binary")
    )

    assert text is None
    assert unsafe is False
