"""Behavioral tests for the state-holder and repair-admission authority."""

import os
from types import SimpleNamespace

import pytest

import hermes_state_holders


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("identity", ["alias", "different-inode", "different-device"])
def test_foreign_holder_uses_device_and_inode_not_path(
    tmp_path, monkeypatch, identity
):
    """Descriptor identity is authoritative even when /proc spells another path."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    alias_path = tmp_path / "namespace-alias" / "state.db" if identity == "alias" else db_path

    proc_root = tmp_path / "proc"
    for pid in (111, 222):
        (proc_root / str(pid) / "fd").mkdir(parents=True)
    os.symlink(db_path, proc_root / "222" / "fd" / "3")

    # Keep the /proc projection local: patching os globally also intercepts
    # pytest's home guard and pathlib while they inspect these same symlinks.
    projected_os = SimpleNamespace(**vars(os))
    projected_os.getpid = lambda: 111
    monkeypatch.setattr(hermes_state_holders, "os", projected_os)
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
        mapped = str(path).replace("/proc", str(proc_root))
        result = real_stat(mapped, *args, **kwargs)
        if str(path) == "/proc/222/fd/3" and identity != "alias":
            values = list(result)
            values[1 if identity == "different-inode" else 2] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

    assert hermes_state_holders.foreign_state_db_holders(db_path) == (
        [(222, str(alias_path))] if identity == "alias" else []
    )
