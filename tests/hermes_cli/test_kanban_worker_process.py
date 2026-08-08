"""Regression tests for the worker_process cluster (s4-w1b extraction).

Covers the reap/classify/liveness helpers moved verbatim from
``hermes_cli.kanban_db`` (cluster c5 / worker_process) into
``hermes_cli.worker_process``, including the module-level
``_recent_worker_exits`` registry that moved with the cluster.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli.worker_process import (
    _classify_worker_exit,
    _pid_alive,
    _record_worker_exit,
    reap_worker_zombies,
)


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


# ---------------------------------------------------------------------------
# Re-export parity
# ---------------------------------------------------------------------------


def test_moved_names_reexported_on_kanban_db_module():
    for name in ("_record_worker_exit", "_classify_worker_exit",
                 "reap_worker_zombies", "_pid_alive"):
        assert getattr(kb, name) is globals()[name], name


def test_direct_module_import_works():
    import hermes_cli.worker_process as wp
    assert wp._pid_alive is _pid_alive


# ---------------------------------------------------------------------------
# Exit registry: record -> classify roundtrip
#
# NOTE: ``os.WIFEXITED``/``os.WIFSIGNALED`` do not exist on Windows, so the
# classification branch degrades to ``("unknown", None)`` there — identical
# to the behavior of the same code before extraction (verified against the
# live repo). The classification-semantics tests are POSIX-only; the registry
# sharing tests run everywhere.
# ---------------------------------------------------------------------------

_is_posix = os.name != "nt"


def test_record_then_classify_clean_exit():
    if not _is_posix:
        pytest.skip("os.WIFEXITED unavailable on Windows")
    pid = 424242
    _record_worker_exit(pid, _exited_status(0))
    assert _classify_worker_exit(pid) == ("clean_exit", 0)


def test_record_then_classify_rate_limited():
    if not _is_posix:
        pytest.skip("os.WIFEXITED unavailable on Windows")
    pid = 424243
    _record_worker_exit(pid, _exited_status(kb.KANBAN_RATE_LIMIT_EXIT_CODE))
    assert _classify_worker_exit(pid) == ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE)


def test_record_then_classify_nonzero_exit():
    if not _is_posix:
        pytest.skip("os.WIFEXITED unavailable on Windows")
    pid = 424244
    _record_worker_exit(pid, _exited_status(7))
    assert _classify_worker_exit(pid) == ("nonzero_exit", 7)


def test_classify_unknown_pid():
    assert _classify_worker_exit(99999999) == ("unknown", None)


def test_record_worker_exit_ignores_falsy_pid():
    import hermes_cli.worker_process as wp
    _record_worker_exit(0, _exited_status(1))
    _record_worker_exit(-5, _exited_status(1))
    # No crash, and nothing recorded for pid 0.
    assert 0 not in wp._recent_worker_exits
    assert -5 not in wp._recent_worker_exits


def test_duplicate_pid_overwrites_latest_wins():
    import hermes_cli.worker_process as wp
    pid = 424245
    _record_worker_exit(pid, _exited_status(1))
    _record_worker_exit(pid, _exited_status(0))
    # The registry holds the raw status; the latest write wins.
    assert wp._recent_worker_exits[pid][0] == _exited_status(0)
    if _is_posix:
        assert _classify_worker_exit(pid) == ("clean_exit", 0)


def test_registry_shared_with_kanban_db_namespace():
    """The registry must be one object shared by both import surfaces."""
    import hermes_cli.worker_process as wp
    pid = 424246
    _record_worker_exit(pid, _exited_status(3))
    assert pid in wp._recent_worker_exits
    # Both surfaces observe the same registry (same function objects).
    assert kb._record_worker_exit is _record_worker_exit
    assert kb._classify_worker_exit is _classify_worker_exit
    assert _classify_worker_exit(pid) == kb._classify_worker_exit(pid)


# ---------------------------------------------------------------------------
# reap_worker_zombies
# ---------------------------------------------------------------------------


def test_reap_worker_zombies_noop_when_no_children():
    reaped = reap_worker_zombies()
    assert isinstance(reaped, list)


@pytest.mark.skipif(os.name == "nt", reason="fork-based reap test is POSIX-only")
def test_reap_worker_zombies_reaps_and_records():
    """POSIX: spawn a child that exits; reap it and see it in the registry."""
    # fork() so the child becomes a real zombie: subprocess.wait() would reap
    # the child itself (waitpid consumes it), leaving nothing for
    # reap_worker_zombies() to find. The child may not have reached os._exit
    # by the time the parent runs waitpid(WNOHANG), so poll with a bounded
    # deadline instead of a single reap call.
    pid = os.fork()
    if pid == 0:
        os._exit(0)  # child exits immediately; parent does not wait -> zombie
    reaped = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        reaped = reap_worker_zombies()
        if pid in reaped:
            break
        time.sleep(0.01)
    assert pid in reaped
    kind, code = _classify_worker_exit(pid)
    assert kind == "clean_exit"
    assert code == 0


# ---------------------------------------------------------------------------
# _pid_alive
# ---------------------------------------------------------------------------


def test_pid_alive_falsy_and_negative():
    assert _pid_alive(None) is False
    assert _pid_alive(0) is False
    assert _pid_alive(-1) is False


def test_pid_alive_own_process():
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_dead_pid_false():
    # A pid that can never be alive (Windows PIDs stay < 2^32; this is far
    # above the range and no process table entry can exist for it).
    assert _pid_alive(2**31 + 99999) is False


# ---------------------------------------------------------------------------
# TTL / size-cap trimming (module state behavior)
# ---------------------------------------------------------------------------


def test_record_worker_exit_trims_by_age(monkeypatch):
    import hermes_cli.worker_process as wp
    pid = 424247
    fake_now = [1_000_000.0]

    monkeypatch.setattr(wp.time, "time", lambda: fake_now[0])
    _record_worker_exit(pid, _exited_status(0))
    fake_now[0] += wp._RECENT_WORKER_EXIT_TTL_SECONDS + 1
    # Trigger a trim by pushing the registry past half its cap: since the
    # entry is older than the TTL, it must be dropped.
    for i in range(wp._RECENT_WORKER_EXITS_MAX // 2 + 1):
        _record_worker_exit(90000 + i, _exited_status(0))
    assert _classify_worker_exit(pid) == ("unknown", None)


def test_record_worker_exit_size_cap_evicts_oldest(monkeypatch):
    import hermes_cli.worker_process as wp
    fake_now = [1_000_000.0]
    monkeypatch.setattr(wp.time, "time", lambda: fake_now[0])

    # Fill the registry past the hard cap; the oldest half must be dropped.
    for i in range(wp._RECENT_WORKER_EXITS_MAX + 20):
        _record_worker_exit(100000 + i, _exited_status(0))
        fake_now[0] += 0.001  # each entry strictly newer than the last

    assert len(wp._recent_worker_exits) <= wp._RECENT_WORKER_EXITS_MAX
    # The very first entry (oldest) must be gone.
    assert _classify_worker_exit(100000) == ("unknown", None)
