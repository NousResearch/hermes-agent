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


def test_psutil_branch_matches_a_holder_by_file_identity_through_an_alias(tmp_path, monkeypatch):
    """Non-Linux scanner (review on 5c82961ab3): psutil reports the PHYSICAL path, the caller scans
    through a symlinked HERMES_HOME (or macOS ``/var`` -> ``/private/var``). A spelling compare
    alone returned ``[]`` and doctor called the file quiet while a process held it. Identity wins."""
    import types

    real = tmp_path / "real"
    real.mkdir()
    db_path = real / "state.db"
    db_path.write_bytes(b"")
    alias = tmp_path / "alias"
    alias.symlink_to(real)

    class _Opened:
        def __init__(self, path):
            self.path = path

    class _Proc:
        def __init__(self, pid, paths):
            self.info = {"pid": pid, "open_files": [_Opened(p) for p in paths]}

    fake_psutil = types.SimpleNamespace(process_iter=lambda attrs: [
        _Proc(4242, [str(db_path)]),                 # holds the PHYSICAL path
        _Proc(4343, [str(tmp_path / "other.db")]),   # unrelated file
    ])
    monkeypatch.setattr(hermes_state_holders, "psutil", fake_psutil)
    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", False)
    monkeypatch.setattr(hermes_state_holders.sys, "platform", "darwin")
    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 1)

    # Scanned through the ALIAS spelling: only file identity can connect it to the physical path.
    assert hermes_state_holders.foreign_state_db_holders(alias / "state.db") == [(4242, str(db_path))]
    # And the unrelated holder never appears.
    assert hermes_state_holders.foreign_state_db_holders(tmp_path / "nothing.db") == []

