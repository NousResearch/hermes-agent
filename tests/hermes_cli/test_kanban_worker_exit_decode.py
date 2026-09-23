"""Worker exit classification must not depend on POSIX-only ``os.WIF*`` helpers.

``_classify_worker_exit`` feeds the dead-worker reclaim: a ``rate_limited``
verdict requeues the card without counting a failure, ``unknown`` counts as a
crash. Windows has neither ``os.WIFEXITED`` nor ``waitpid(-1)``, so both the
decode and the reaper's exit capture have a Windows-independent path.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


def _spawn_exit(code: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])  # noqa: S603
    proc.wait()
    return proc




@pytest.mark.windows_only
def test_native_windows_reaper_and_decode(monkeypatch):
    """Native Windows, nothing patched: ``_IS_WINDOWS`` selects the Popen-poll
    reaper and the decode runs where ``os.WIFEXITED`` does not exist, so the
    rate-limit sentinel exit is a requeue, not a crash."""
    monkeypatch.setattr(kbd, "_live_worker_procs", {})
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})
    assert not hasattr(os, "WIFEXITED")
    proc = _spawn_exit(kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    kbd._live_worker_procs[proc.pid] = proc
    assert kbd.reap_worker_zombies() == [proc.pid]
    assert kbd._classify_worker_exit(proc.pid) == ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE)
