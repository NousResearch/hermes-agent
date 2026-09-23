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



def test_windows_holder_scan_delegates_to_restart_manager(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    db_path.touch()
    seen = []

    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        hermes_state_holders,
        "_windows_restart_manager_holders",
        lambda path: seen.append(path) or [(4242, str(path))],
    )

    assert hermes_state_holders.foreign_state_db_holders(db_path) == [(4242, str(db_path))]
    assert seen == [db_path]


def test_windows_holder_scan_failure_fails_closed(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    db_path.touch()

    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", True)

    def fail(_path):
        raise OSError("Restart Manager unavailable")

    monkeypatch.setattr(hermes_state_holders, "_windows_restart_manager_holders", fail)

    holders = hermes_state_holders.foreign_state_db_holders(db_path)
    assert holders[0][0] == -1
    assert "Restart Manager scan failed" in holders[0][1]
    refusal = hermes_state_holders.held_store_refusal(db_path, command="optimize-storage")
    assert refusal is not None and "cannot prove the database is quiet" in refusal
