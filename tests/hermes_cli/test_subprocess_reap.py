"""Fire-and-forget children are reaped: a long-lived parent must not collect zombies.

A detached helper nobody waits for stays ``<defunct>`` under its parent. CPython only polls abandoned
``Popen`` objects when the same process starts another child, so a plugin host or an idle dashboard
that spawns one helper (the hourly scratch prune) kept it as a zombie until it was restarted. A handle
the dashboard keeps for its status route is never abandoned, so CPython never polls it at all.
"""

import subprocess
import sys

import psutil
import pytest

from hermes_cli._subprocess_compat import reap_in_background
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")  # zombies are a POSIX process state


def test_reap_in_background_collects_a_child_nobody_waits_for():
    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    pid = proc.pid
    reap_in_background(proc)
    del proc

    # psutil.pid_exists is True for a zombie, so this only passes once the child has been waited for.
    wait_until(lambda: not psutil.pid_exists(pid), timeout=30, what="the finished child to be reaped")


def test_dashboard_action_child_is_reaped_without_a_status_poll(monkeypatch, tmp_path):
    import hermes_cli._launchers as launchers
    from hermes_cli import web_server_gateway as gateway

    monkeypatch.setattr(launchers, "runtime_command", lambda root, argv, **kwargs: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(gateway, "_ACTION_LOG_DIR", tmp_path)
    for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS", "_ACTION_LOG_STARTS"):
        monkeypatch.setattr(gateway, registry, {})

    proc = gateway._spawn_hermes_action(["doctor"], "doctor")

    # The registry keeps the handle for /api/actions/doctor/status; nobody polls it here, as when the UI
    # that fired the action stops asking.
    assert gateway._ACTION_PROCS["doctor"] is proc
    wait_until(lambda: not psutil.pid_exists(proc.pid), timeout=30, what="the finished action to be reaped")
    assert proc.poll() == 0, "the status route must still read the exit code"
