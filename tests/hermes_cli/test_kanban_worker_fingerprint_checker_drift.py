"""A worker's identity must not depend on which process inspects it.

On macOS, ``psutil`` adjusts ``Process.create_time()`` by the difference between the boot time it
read when it was imported and the boot time it reads now (``psutil._psosx.adjust_proc_create_time``).
Two long-lived checkers that imported ``psutil`` at different moments therefore report different
creation times for the same live process, and exact comparison declares that process dead. The
2026-10-19 board outage came from this: one dispatcher closed another process's live runs as
``pid not alive``.

The identity must also keep real PID reuse distinct: a process that started well after the recorded
one, under the same PID, is a stranger and must not match.
"""

import subprocess
import time

import psutil
import pytest

from hermes_cli import kanban_db_dispatch as kbd


def _boot_clock():
    fn = getattr(psutil, "_psosx", None) and getattr(psutil._psosx, "adjust_proc_create_time", None)
    return fn, (fn.__globals__ if fn else None)


@pytest.fixture
def live_child():
    child = subprocess.Popen(["/bin/sleep", "120"])
    time.sleep(0.3)
    try:
        yield child
    finally:
        child.kill()
        child.wait()


def _with_boot_clock(globals_, boot_time, fn):
    saved = globals_["INIT_BOOT_TIME"]
    globals_["INIT_BOOT_TIME"] = boot_time
    try:
        return fn()
    finally:
        globals_["INIT_BOOT_TIME"] = saved


def test_same_live_process_matches_across_checker_boot_clocks(live_child):
    """Two checkers whose psutil boot clocks differ by 2 s must still agree the child is the worker
    recorded by the first of them. This is the misjudgment that closed live runs."""
    adjust, globals_ = _boot_clock()
    if adjust is None:
        pytest.skip("boot-clock adjustment is macOS psutil behavior")
    pid = live_child.pid
    base = psutil.boot_time()

    recorded = _with_boot_clock(globals_, base, lambda: kbd._process_fingerprint(pid))
    reread = _with_boot_clock(globals_, base + 2.0, lambda: kbd._process_fingerprint(pid))

    assert recorded is not None and "|" in recorded
    assert reread == recorded
    assert _with_boot_clock(globals_, base + 2.0, lambda: kbd._worker_alive(pid, recorded)) is True
    assert _with_boot_clock(globals_, base, lambda: kbd._worker_alive(pid, recorded)) is True


def test_pid_reuse_stays_distinct_across_checker_boot_clocks(live_child):
    """A process that started 30 s later and inherited the PID is a stranger, from every checker's
    boot clock — drift tolerance must not swallow a real recycle."""
    adjust, globals_ = _boot_clock()
    if adjust is None:
        pytest.skip("boot-clock adjustment is macOS psutil behavior")
    pid = live_child.pid
    base = psutil.boot_time()
    recorded = _with_boot_clock(globals_, base, lambda: kbd._process_fingerprint(pid))
    epoch, start = recorded.split("|", 1)
    recycled = f"{epoch}|{int(start) + 3000}"

    for offset in (0.0, 2.0):
        assert _with_boot_clock(globals_, base + offset, lambda: kbd._worker_alive(pid, recycled)) is False


def test_exited_process_is_not_alive_from_any_checker(live_child):
    adjust, globals_ = _boot_clock()
    if adjust is None:
        pytest.skip("boot-clock adjustment is macOS psutil behavior")
    pid = live_child.pid
    base = psutil.boot_time()
    recorded = _with_boot_clock(globals_, base, lambda: kbd._process_fingerprint(pid))
    live_child.kill()
    live_child.wait()

    for offset in (0.0, 2.0):
        assert _with_boot_clock(globals_, base + offset, lambda: kbd._worker_alive(pid, recorded)) is False
