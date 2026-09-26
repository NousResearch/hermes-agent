"""Synthetic psutil attribute contracts, not native quiescence qualification."""

import os
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_state_holders as holders


def _watched(db_path):
    real = os.path.realpath(str(db_path))
    return {holders.canonical_sqlite_path(real + sidecar) for sidecar in ("", "-wal", "-shm")}


@pytest.mark.parametrize("attributes", [{}, {"open_files": None}], ids=["missing", "none"])
@pytest.mark.parametrize("known_position", [None, "before", "after"])
@pytest.mark.parametrize(
    ("identity", "suspected"),
    [
        ({"cmdline": ["/usr/sbin/syslogd"], "name": "syslogd"}, False),
        ({"cmdline": None, "name": "syslogd"}, False),
        ({"cmdline": [], "name": "syslogd"}, False),
        ({"cmdline": ["/usr/bin/logger", "hermes", "run_agent.py"]}, False),
        ({"cmdline": ["python3", "-c", "print('hermes')"], "name": "python3"}, False),
        ({"cmdline": ["python3", "worker.py", "run_agent.py"]}, False),
        ({"cmdline": ["hermes", "gateway", "run"]}, True),
        ({"cmdline": ["python3", "-X", "dev", "-m", "hermes_cli.main"]}, True),
        ({"cmdline": ["python3", "/opt/app/run_agent.py"]}, True),
        ({"cmdline": None, "name": "hermes"}, True),
        ({"cmdline": None, "name": "python3.14"}, True),
        ({"cmdline": [], "name": "pypy3"}, True),
        ({"cmdline": None, "name": None}, True),
        ({}, True),
    ],
    ids=[
        "daemon", "daemon-no-argv", "daemon-empty-argv", "argv-mention",
        "python-code", "python-other-script", "hermes", "python-module",
        "python-script", "hermes-no-argv", "python-no-argv", "pypy-empty-argv",
        "opaque", "missing-identity",
    ],
)
def test_unavailable_open_files_uses_identity_without_losing_known_holders(
    tmp_path, monkeypatch, attributes, known_position, identity, suspected
):
    db_path = tmp_path / "state.db"
    watched = _watched(db_path)
    known = SimpleNamespace(info={
        "pid": 4242, "name": "unrelated", "cmdline": ["unrelated"],
        "open_files": [SimpleNamespace(path=str(db_path))],
    })
    unknown = SimpleNamespace(info={"pid": 4343, **attributes, **identity})
    processes = [unknown]
    if known_position is not None:
        processes.insert(0 if known_position == "before" else 1, known)
    process_iter = Mock(return_value=processes)
    monkeypatch.setattr(holders, "psutil", SimpleNamespace(process_iter=process_iter))
    monkeypatch.setattr(holders.os, "getpid", lambda: 9999)

    found = holders._psutil_foreign_holders(watched)

    assert ((4242, str(db_path)) in found) is (known_position is not None)
    assert any(pid == 4343 and path.startswith("uninspectable holder:") for pid, path in found) is suspected
    # Feed the adapter output into real admission guards and a real SQLite probe;
    # this seam does not pretend Linux verification is a native macOS scan.
    monkeypatch.setattr(
        holders, "foreign_state_db_holders", lambda _path: holders._psutil_foreign_holders(watched)
    )
    probe = Mock(wraps=sqlite3.connect)
    blocked = suspected or known_position is not None
    assert holders.live_writer_holds_db(db_path, connect_repair_durable=probe) is blocked
    assert (holders.held_store_refusal(db_path, command="vacuum") is not None) is blocked
    if blocked:
        probe.assert_not_called()
    else:
        probe.assert_called_once_with(db_path, timeout=0.0)
        # The successful probe must have released its exclusive connection.
        with sqlite3.connect(db_path, timeout=0.0) as connection:
            connection.execute("CREATE TABLE probe_released (value)")


@pytest.mark.parametrize(
    ("suffix", "expected_holder"),
    [
        (None, False),
        (".unrelated", False),
        ("", True),
        ("-wal", True),
        ("-shm", True),
        ("-wal (deleted)", True),
    ],
    ids=["empty", "unrelated", "db", "wal", "shm", "deleted-wal"],
)
def test_successful_open_files_preserves_paths_and_probe_policy(
    tmp_path, monkeypatch, suffix, expected_holder
):
    db_path = tmp_path / "state.db"
    watched = _watched(db_path)
    path = str(db_path) + suffix if suffix is not None else None
    open_files = [SimpleNamespace(path=path)] if path is not None else []
    process_iter = Mock(return_value=[
        SimpleNamespace(info={
            "pid": 4242, "open_files": open_files,
            "cmdline": ["/usr/bin/unrelated"], "name": "unrelated",
        }),
        # Unavailable descriptors of this process remain excluded as before.
        SimpleNamespace(info={"pid": 9999, "open_files": None}),
    ])
    monkeypatch.setattr(holders, "psutil", SimpleNamespace(process_iter=process_iter))
    monkeypatch.setattr(holders.os, "getpid", lambda: 9999)

    found = holders._psutil_foreign_holders(watched)

    assert found == ([(4242, path)] if expected_holder else [])
    monkeypatch.setattr(
        holders, "foreign_state_db_holders", lambda _path: holders._psutil_foreign_holders(watched)
    )
    probe = Mock()
    # Any foreign holder is authoritative (#103339); only a clean scan reaches the lock probe.
    assert holders.live_writer_holds_db(db_path, connect_repair_durable=probe) is expected_holder
    if expected_holder:
        probe.assert_not_called()
    else:
        probe.assert_called_once_with(db_path, timeout=0.0)
        probe.return_value.execute.assert_any_call("BEGIN IMMEDIATE")
        probe.return_value.close.assert_called_once_with()
