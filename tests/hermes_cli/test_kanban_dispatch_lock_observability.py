"""Board tick contention must be observable on the very first skipped tick."""
import argparse
import json
import logging
import os
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from tests.hermes_cli.test_kanban_dispatch_lock import kanban_home, conn  # noqa: F401


def test_first_contended_tick_reports_holder(conn, monkeypatch, caplog):
    db_path = kb.kanban_db_path()
    monkeypatch.setattr(kbc.time, "monotonic", lambda: 100.0)
    with kbc._dispatch_tick_lock(db_path) as held:
        assert held
        monkeypatch.setattr(kbc.time, "monotonic", lambda: 107.0)
        with caplog.at_level(logging.WARNING):
            result = kbd.dispatch_once(conn, dry_run=True)
        assert result.skipped_locked
        assert result.lock_holder["pid"] == os.getpid()
        assert result.lock_holder["age_seconds"] == 7.0
        assert "_dispatch_tick_lock" in result.lock_holder["acquire_site"]
        assert str(os.getpid()) in caplog.text
        assert "7.0s" in caplog.text
        assert "_dispatch_tick_lock" in caplog.text
    assert kbc._read_dispatch_lock_holder(db_path) == {}


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_reports_real_contention(conn, capsys, json_output):
    from hermes_cli.kanban_ops import _cmd_dispatch

    with kbc._dispatch_tick_lock(kb.kanban_db_path()) as held:
        assert held
        assert _cmd_dispatch(argparse.Namespace(dry_run=True, json=json_output)) == 0
    output = capsys.readouterr().out
    if json_output:
        data = json.loads(output)
        assert data["skipped_locked"] is True
        assert data["lock_holder"]["pid"] == os.getpid()
    else:
        assert "skipped: board dispatcher lock" in output
        assert str(os.getpid()) in output
        assert "_dispatch_tick_lock" in output


@pytest.mark.parametrize("payload", [b"", b" {", b" []", b' {"pid": "bad"}',
    b" " + json.dumps({"pid": 1, "monotonic": 10**1000, "acquire_site": "test"}).encode(),
    b' {"pid": 1, "monotonic": NaN, "acquire_site": "test"}'])
def test_legacy_or_malformed_stamp_does_not_hide_skip(conn, payload):
    path = kb.kanban_db_path()
    with kbc._dispatch_tick_lock(path) as held:
        assert held
        with path.with_name(path.name + ".dispatch.lock").open("r+b") as stamp:
            stamp.seek(1)  # Do not write the Windows locked byte.
            stamp.write(payload[1:])
            stamp.truncate()
        result = kbd.dispatch_once(conn, dry_run=True)
        assert result.skipped_locked
        assert result.lock_holder == {}


def test_other_process_reports_parent_holder_in_one_tick(conn):
    code = """
import json
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
with kbc.connect() as conn:
    result = kbd.dispatch_once(conn, dry_run=True)
print(json.dumps({'skipped': result.skipped_locked, 'holder': result.lock_holder}))
"""
    with kbc._dispatch_tick_lock(kb.kanban_db_path()) as held:
        assert held
        child = subprocess.run(
            [sys.executable, "-c", code], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30, check=True,
        )
    result = json.loads(child.stdout)
    assert result["skipped"] is True
    assert result["holder"]["pid"] == os.getpid()
    assert result["holder"]["age_seconds"] >= 0
    assert "_dispatch_tick_lock" in result["holder"]["acquire_site"]


def test_unknown_holder_is_explicit():
    assert kbc.format_dispatch_lock_skip({}) == (
        "skipped: board dispatcher lock held by pid unknown for unknown; acquire site=unknown"
    )
