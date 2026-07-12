import os

import psutil

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_true_when_parent_create_time_mismatch():
    # Same ppid but a different starttime means the PID was reused by an
    # unrelated process. On Linux the /proc starttime won't be 0.0, so this
    # exercises the mismatch branch directly.
    me = psutil.Process()
    assert slash_worker._is_orphaned(me.pid, 0.0, getppid=lambda: me.pid) is True


def test_is_orphaned_false_when_parent_alive_and_matches():
    # Snapshot the starttime with the SAME helper the implementation compares
    # against, so the units match (ticks on Linux, epoch via psutil otherwise).
    me = psutil.Process()
    starttime = slash_worker._read_proc_starttime(me.pid)
    if starttime is not None:
        snapshot = float(starttime)
    else:
        snapshot = me.create_time()
    assert (
        slash_worker._is_orphaned(me.pid, snapshot, getppid=lambda: me.pid) is False
    )


def test_read_proc_starttime_handles_comm_with_spaces_and_parens():
    # /proc/<pid>/stat field 2 (comm) can contain spaces and parens — e.g.
    # "chrome (helper)". A naive split() over the whole record corrupts every
    # field index after it; the helper must split on the LAST ')'.
    me = psutil.Process()
    # comm is our own process name; force a synthetic record with a tricky
    # comm to prove the index is correct regardless of content.
    fake_pid = me.pid
    real_stat = open(f"/proc/{fake_pid}/stat").read()
    # Replace comm with something containing spaces+parens.
    real_tail = real_stat.rsplit(")", 1)[1]
    fake_stat = f"{fake_pid} (my proc (with) parens){real_tail}"
    import tui_gateway.slash_worker as sw
    import unittest.mock as mock

    m = mock.mock_open(read_data=fake_stat)
    with mock.patch("builtins.open", m):
        result = sw._read_proc_starttime(fake_pid)
    # The real starttime from the tail must parse correctly despite the
    # spaces/parens in the synthetic comm.
    expected = int(real_tail.split()[19])
    assert result == expected


def test_is_orphaned_fallback_preserves_pid_reuse_guard(monkeypatch):
    # On non-Linux (or when /proc is unavailable) the fallback must NOT
    # degrade to a bare pid_exists() — it must compare the psutil
    # create_time() fingerprint so a recycled PID is detected as different.
    me = psutil.Process()
    real_create_time = me.create_time()

    # Force the /proc fast path off so the fallback runs.
    monkeypatch.setattr(slash_worker, "_read_proc_starttime", lambda pid: None)
    # Mismatched create_time snapshot → must report orphaned (PID reuse).
    assert (
        slash_worker._is_orphaned(
            me.pid, real_create_time + 9999, getppid=lambda: me.pid
        )
        is True
    )
    # Matching snapshot → must report NOT orphaned (parent alive, same proc).
    assert (
        slash_worker._is_orphaned(
            me.pid, real_create_time, getppid=lambda: me.pid
        )
        is False
    )
