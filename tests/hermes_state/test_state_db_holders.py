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


def test_windows_restart_manager_scan_sizes_then_excludes_self(monkeypatch, tmp_path):
    """The rstrtmgr lane sizes on ERROR_MORE_DATA, drops our own pid, and fails closed."""
    import ctypes

    db_path = tmp_path / "state.db"
    db_path.touch()
    (tmp_path / "state.db-wal").touch()
    calls = []

    class _Fn:
        def __init__(self, impl):
            self.impl = impl

        def __call__(self, *args):
            return self.impl(*args)

    def start(session, _flags, _key):
        session._obj.value = 7
        return 0

    def register(_session, count, names, *_rest):
        calls.append(("register", count, sorted(names)))
        return 0

    def get_list(_session, needed, count, apps, _reasons):
        needed._obj.value = 2
        if apps is None:
            return 234  # ERROR_MORE_DATA: sizing call
        apps[0].process.pid = os.getpid()
        apps[1].process.pid = 4242
        count._obj.value = 2
        return 0

    class _Api:
        RmStartSession = _Fn(start)
        RmRegisterResources = _Fn(register)
        RmGetList = _Fn(get_list)
        RmEndSession = _Fn(lambda _session: calls.append("end") or 0)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: _Api(), raising=False)

    holders = hermes_state_holders._windows_restart_manager_holders(db_path)

    assert [pid for pid, _ in holders] == [4242]
    assert calls[0] == ("register", 2, sorted(str(p) for p in (db_path, tmp_path / "state.db-wal")))
    assert calls[-1] == "end"

    _Api.RmStartSession = _Fn(lambda *_args: 5)
    with pytest.raises(OSError):
        hermes_state_holders._windows_restart_manager_holders(db_path)
