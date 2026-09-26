"""Synthetic psutil attribute contracts, not native quiescence qualification."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_state_holders as holders


def _watched(db_path):
    real = os.path.realpath(str(db_path))
    return {holders.canonical_sqlite_path(real + sidecar) for sidecar in ("", "-wal", "-shm")}


@pytest.mark.parametrize("attributes", [{}, {"open_files": None}], ids=["missing", "none"])
@pytest.mark.parametrize("unknown_first", [False, True])
def test_unavailable_open_files_blocks_probe_even_with_known_holder(
    tmp_path, monkeypatch, attributes, unknown_first
):
    db_path = tmp_path / "state.db"
    watched = _watched(db_path)
    known = SimpleNamespace(info={"pid": 4242, "open_files": [SimpleNamespace(path=str(db_path))]})
    unknown = SimpleNamespace(info={"pid": 4343, **attributes})
    processes = [unknown, known] if unknown_first else [known, unknown]
    process_iter = Mock(return_value=processes)
    monkeypatch.setattr(holders, "psutil", SimpleNamespace(process_iter=process_iter))
    monkeypatch.setattr(holders.os, "getpid", lambda: 9999)

    found = holders._psutil_foreign_holders(watched)

    assert (4242, str(db_path)) in found
    assert any(pid == 4343 and path.startswith("uninspectable holder:") for pid, path in found)
    process_iter.assert_called_once_with(["pid", "open_files"])
    # Feed the adapter output into the real admission guard; the helper seam avoids
    # pretending a Linux verifier is a native macOS process scan.
    monkeypatch.setattr(
        holders, "foreign_state_db_holders", lambda _path: holders._psutil_foreign_holders(watched)
    )
    probe = Mock()
    assert holders.live_writer_holds_db(db_path, connect_repair_durable=probe) is True
    probe.assert_not_called()


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
        SimpleNamespace(info={"pid": 4242, "open_files": open_files}),
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
