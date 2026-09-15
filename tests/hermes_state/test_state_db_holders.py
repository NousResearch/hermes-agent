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


def test_psutil_branch_resolves_a_holder_reported_under_an_alias_path(tmp_path, monkeypatch):
    """Non-Linux scanner: psutil reports the kernel-resolved pathname, which need not be spelled the
    way the caller spells the database — a symlinked profile home, or macOS ``/var`` vs
    ``/private/var``. Both sides are realpath'd (`9b419a2d3c`), so the compare must survive the two
    spellings disagreeing.

    Deliberately makes the PSUTIL side the aliased one: realpath'ing only the watched side (which
    `foreign_state_db_holders` does when it resolves ``db_path``) would still match if the fake
    reported the physical path, so this pins the compare-side resolve specifically.
    """
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
        _Proc(4242, [str(alias / "state.db")]),      # holder reported under the ALIAS spelling
        _Proc(4343, [str(tmp_path / "other.db")]),   # unrelated file
    ])
    monkeypatch.setattr(hermes_state_holders, "psutil", fake_psutil)
    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", False)
    monkeypatch.setattr(hermes_state_holders.sys, "platform", "darwin")
    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 1)

    # Scanned by its PHYSICAL path; only resolving the reported path connects the two.
    assert hermes_state_holders.foreign_state_db_holders(db_path) == [(4242, str(alias / "state.db"))]
    # And the unrelated holder never appears.
    assert hermes_state_holders.foreign_state_db_holders(tmp_path / "nothing.db") == []


def test_uninspectable_process_is_a_scan_gap_not_zero_open_files(tmp_path, monkeypatch):
    """psutil ``process_iter(attrs)`` uses ``ad_value=None``: on AccessDenied / ZombieProcess the
    attribute is None, not an empty list. ``or ()`` counted that as an inspected process with zero
    files, so a system-owned holder the caller cannot inspect produced a confident empty scan
    (review on e0c8ee3bf5). Measured on macOS: 359 of 1093 processes, and only 2% of those expose a
    readable cmdline, so the Linux argv gating cannot classify them.

    The gap row is OPT-IN. Default off keeps ``live_writer_holds_db`` — which fails closed on any
    negative pid — from refusing structural maintenance on every macOS scan; it has the EXCLUSIVE
    lock probe as a second gate. ``hermes doctor``'s report has none, so it opts in.
    """
    import types

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")

    class _Proc:
        def __init__(self, pid, open_files):
            self.info = {"pid": pid, "open_files": open_files}

    fake = types.SimpleNamespace(process_iter=lambda attrs: [
        _Proc(4242, None),   # AccessDenied -> None
        _Proc(4343, []),     # genuinely inspected, holds nothing
    ])
    monkeypatch.setattr(hermes_state_holders, "psutil", fake)
    monkeypatch.setattr(hermes_state_holders, "_IS_WINDOWS", False)
    monkeypatch.setattr(hermes_state_holders.sys, "platform", "darwin")
    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 1)

    # Default: unchanged for every existing caller.
    assert hermes_state_holders.foreign_state_db_holders(db_path) == []

    # Opt in: the uninspectable process becomes one aggregate scan-gap row.
    gaps = hermes_state_holders.foreign_state_db_holders(db_path, include_scan_gaps=True)
    assert len(gaps) == 1
    pid, detail = gaps[0]
    assert pid == -1
    assert "scan incomplete" in detail and "1 process(es)" in detail

    # A fully inspectable scan reports no gap even when asked.
    fake.process_iter = lambda attrs: [_Proc(4343, [])]
    assert hermes_state_holders.foreign_state_db_holders(db_path, include_scan_gaps=True) == []

