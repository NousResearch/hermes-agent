"""Behavioral tests for the state-holder and repair-admission authority."""

import os

import pytest

import hermes_state_holders


@pytest.mark.linux_only
def test_foreign_holder_accepts_same_inode_reached_through_an_alias(
    tmp_path, monkeypatch
):
    """Descriptor identity is authoritative even when /proc spells another path."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    alias_path = tmp_path / "namespace-alias" / "state.db"

    proc_root = tmp_path / "proc"
    for pid in (111, 222):
        (proc_root / str(pid) / "fd").mkdir(parents=True)
    os.symlink(db_path, proc_root / "222" / "fd" / "3")

    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)
    real_listdir = os.listdir

    def _listdir(path):
        if isinstance(path, str):
            path = path.replace("/proc", str(proc_root))
        return real_listdir(path)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)

    def _readlink(path):
        if path == "/proc/222/fd/3":
            return str(alias_path)
        return os.readlink(path.replace("/proc", str(proc_root)))

    monkeypatch.setattr(hermes_state_holders.os, "readlink", _readlink)
    real_stat = os.stat

    def _stat(path, *args, **kwargs):
        path = str(path).replace("/proc", str(proc_root))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

    assert hermes_state_holders.foreign_state_db_holders(db_path) == [
        (222, str(alias_path))
    ]


class _Opened:
    def __init__(self, path: str) -> None:
        self.path = path


def test_windows_holder_scan_reports_psutil_open_files(tmp_path, monkeypatch):
    """A Windows scan must list foreign open files, not pretend the store is quiet."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(hermes_state_holders.sys, "platform", "win32")

    class _Proc:
        info = {"pid": 4242, "open_files": [_Opened(os.path.realpath(db_path))]}

    monkeypatch.setattr(
        hermes_state_holders.psutil, "process_iter", lambda _attrs: [_Proc()]
    )
    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)

    assert [
        pid for pid, _path in hermes_state_holders.foreign_state_db_holders(db_path)
    ] == [4242]


def test_windows_holder_scan_fail_closed_when_psutil_is_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db_path.touch()
    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(hermes_state_holders.sys, "platform", "win32")
    monkeypatch.setattr(hermes_state_holders, "psutil", None)

    assert hermes_state_holders.foreign_state_db_holders(db_path) == [
        (-1, "open-file scan unavailable")
    ]
