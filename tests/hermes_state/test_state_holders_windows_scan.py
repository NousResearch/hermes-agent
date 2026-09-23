"""foreign_state_db_holders() on Windows: the psutil open-files scan (#120205).

The Windows lane used to short-circuit to ``[]`` — an all-clear every write-guard
consumer (``sessions optimize-storage``/``prune`` admission, repair, set-journal-mode)
then trusted, so structural maintenance ran under live writers with no refusal
(see also the stale-stopgap note this scan retires in sessions_cmd_journal_mode).

These tests fake the host as win32 — branch selection reads ``sys.platform`` live, not
an import-time constant — and inject a fake psutil, so every CI lane exercises the
Windows code path deterministically. The fake records the attributes process_iter was
asked for, so a resurrected ``return []`` bypass cannot pass as a clean scan.
"""

import os
import sys

import hermes_state_holders
import pytest

# Above typical Linux pid ceilings and never a real pid under test: describe_holder_pid's
# /proc fallback must miss so the fake psutil's cmdline is what the refusal shows.
_HOLDER_PID = 4_190_000


class _FakeOpenFile:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeProcess:
    def __init__(self, pid: int, open_files) -> None:
        self.info = {"pid": pid, "open_files": list(open_files)}


class _FakeProcessHandle:
    def cmdline(self):
        return ["hermes", "gateway", "run"]


class _FakePsutil:
    def __init__(self, processes) -> None:
        self._processes = list(processes)
        self.requested_attrs = None

    def process_iter(self, attrs):
        self.requested_attrs = list(attrs)
        return list(self._processes)

    def Process(self, pid):  # noqa: N802 - mirrors the psutil API name
        return _FakeProcessHandle()


def _fake_windows(monkeypatch, psutil):
    """Fake the host as win32 for BOTH platform reads: the live ``sys.platform``
    selector and an import-time ``_IS_WINDOWS`` constant (raising=False, so a
    resurrected bypass keyed on either form trips these tests instead of
    passing as a clean scan)."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(hermes_state_holders, "psutil", psutil)
    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)


def test_windows_scan_flags_foreign_holder_of_the_db(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00")
    _fake_windows(
        monkeypatch,
        _FakePsutil([
            _FakeProcess(111, [_FakeOpenFile(str(tmp_path / "unrelated.log"))]),
            _FakeProcess(_HOLDER_PID, [_FakeOpenFile(str(db))]),
        ]),
    )

    holders = hermes_state_holders.foreign_state_db_holders(db)

    assert holders == [(_HOLDER_PID, str(db))]


def test_windows_scan_flags_a_wal_sidecar_holder(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00")
    _fake_windows(
        monkeypatch,
        _FakePsutil([
            _FakeProcess(_HOLDER_PID, [_FakeOpenFile(str(db) + "-wal")]),
        ]),
    )

    holders = hermes_state_holders.foreign_state_db_holders(db)

    assert [pid for pid, _ in holders] == [_HOLDER_PID]


def test_windows_quiet_store_is_a_scanned_all_clear_not_a_bypass(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00")
    fake = _FakePsutil([
        _FakeProcess(_HOLDER_PID, [_FakeOpenFile(str(tmp_path / "other.db"))])
    ])
    _fake_windows(monkeypatch, fake)

    holders = hermes_state_holders.foreign_state_db_holders(db)

    assert holders == []
    # The scan must have actually run: a reintroduced `if _IS_WINDOWS: return []` leaves
    # this None and fails here instead of passing as a clean store.
    assert fake.requested_attrs == ["pid", "open_files"]


def test_windows_missing_psutil_fails_closed_as_unknown_holder(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00")
    _fake_windows(monkeypatch, None)

    holders = hermes_state_holders.foreign_state_db_holders(db)

    assert holders == [(-1, "open-file scan unavailable")]


def test_windows_scan_failure_fails_closed_as_unknown_holder(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00")

    class _ExplodingPsutil(_FakePsutil):
        def process_iter(self, attrs):
            raise PermissionError("handle table locked")

    _fake_windows(monkeypatch, _ExplodingPsutil([]))

    holders = hermes_state_holders.foreign_state_db_holders(db)

    assert len(holders) == 1
    pid, target = holders[0]
    assert pid == -1
    assert "open-file scan failed" in target


def test_windows_holder_refuses_structural_maintenance(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00")
    _fake_windows(
        monkeypatch,
        _FakePsutil([
            _FakeProcess(_HOLDER_PID, [_FakeOpenFile(str(db))]),
        ]),
    )

    refusal = hermes_state_holders.held_store_refusal(db, command="optimize-storage")

    # Fail-closed end to end: a same-user foreign holder on Windows now produces the
    # operator-facing refusal instead of the silent all-clear the [] bypass gave.
    assert refusal is not None
    assert "Refusing" in refusal
    assert str(_HOLDER_PID) in refusal


def test_windows_scan_result_matches_the_posix_contract(tmp_path, monkeypatch):
    """The Windows all-clear must be indistinguishable from a scanned POSIX one:
    ``held_store_refusal`` returns None (proceed) — never a bypassed maybe."""
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite format 3\x00")
    _fake_windows(
        monkeypatch,
        _FakePsutil([
            _FakeProcess(_HOLDER_PID, [_FakeOpenFile(str(tmp_path / "elsewhere.db"))]),
        ]),
    )

    assert os.path.realpath(str(db))  # tmp_path resolves cleanly on every lane
    assert (
        hermes_state_holders.held_store_refusal(db, command="optimize-storage") is None
    )
