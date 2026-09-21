"""Process-level repro of the 2026-09-20 orphan (#93091 follow-up, card item 4).

RED on the un-fixed tree: a spawner starts a delivery child, the spawner is killed mid-turn, and
the child keeps running — reparented, holding the target profile's ``state.db`` and its MCP
children, still spawning deliveries.

These tests drive REAL processes. The child below mirrors the real one-shot contract exactly
(``cli._run_quiet_single_query``): the armed watchdog fires on a watcher thread and SETS A FLAG —
it does not exit the process from that thread — and the main thread's turn observes the flag,
ends, and leaves through its own exit code. SIGKILL on the spawner is deliberate: it is the
fleet's own way of killing a stuck delivery, and nothing catchable in the spawner could clean up
after it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hermes_cli import quiet_single_query as qsq

REPO_ROOT = Path(__file__).resolve().parents[2]
LINUX_ONLY = pytest.mark.linux_only

# The child's "turn" is an idle wait: nothing in it knows about the requester, so only the armed
# policy can end it — which is exactly the property under test. The callback sets a flag (never
# ``sys.exit`` from the watcher thread, which would only end that thread) and the turn observes it.
CHILD_CODE = textwrap.dedent(
    f"""
    import sys, time
    sys.path.insert(0, {str(REPO_ROOT)!r})
    from hermes_cli import quiet_single_query as qsq

    gone = {{"flag": False}}
    qsq.arm_requester_watchdog(qsq.peek_requester_pid(), lambda: gone.__setitem__("flag", True))
    deadline = time.monotonic() + 60
    while not gone["flag"] and time.monotonic() < deadline:
        time.sleep(0.05)          # the in-flight turn, observing its own cancellation
    sys.exit(qsq.REQUESTER_DEATH_EXIT_CODE if gone["flag"] else 0)
    """
)

# The spawner arms the policy on its child and then blocks forever, so the test can kill it
# mid-"turn". It prints the child's pid so the test can watch that child specifically.
SPAWNER_CODE = textwrap.dedent(
    f"""
    import os, subprocess, sys, time
    sys.path.insert(0, {str(REPO_ROOT)!r})
    from hermes_cli import quiet_single_query as qsq
    env = {{**os.environ, **qsq.requester_pid_env()}}
    child = subprocess.Popen([sys.executable, "-c", {CHILD_CODE!r}], env=env)
    print(child.pid, flush=True)
    while True:
        time.sleep(0.05)
    """
)


def _alive(pid: int) -> bool:
    """Whether ``pid`` is a RUNNING process (a zombie counts as gone — it has already exited)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            state = fh.read().rsplit(b")", 1)[1].split()[0]
    except OSError:
        return False
    return state != b"Z"


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_child(code: str, timeout: float = 15.0) -> int:
    child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    finally:
        _kill(child.pid)


@LINUX_ONLY
def test_killing_the_requester_ends_the_delivery_child():
    """The card's repro: kill the requester mid-turn -> the spawned session ends by itself."""
    spawner = subprocess.Popen(
        [sys.executable, "-c", SPAWNER_CODE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    assert spawner.stdout is not None
    child_pid = None
    try:
        line = spawner.stdout.readline().strip()
        assert line.isdigit(), f"spawner did not report its delivery child: {line!r}"
        child_pid = int(line)
        assert _alive(child_pid), "the delivery child is running before the requester dies"

        # The fleet kills the requester. SIGKILL: the spawner cannot clean up, so only the
        # child's own policy can end it.
        _kill(spawner.pid)
        spawner.wait(timeout=10)

        assert _wait_gone(child_pid, 20.0), (
            "the delivery child must end itself when its requester dies — otherwise it holds the "
            "target profile's state.db and its MCP children with nobody left to read the answer "
            "(the 1h46m orphan of 2026-09-20)"
        )
    finally:
        _kill(spawner.pid)
        if child_pid:
            _kill(child_pid)
        for stream in (spawner.stdout, spawner.stderr):
            if stream is not None:
                stream.close()


@LINUX_ONLY
def test_a_requester_already_dead_at_start_is_caught_immediately():
    """A requester killed in the spawn window is caught on the FIRST check, not a cadence later.

    That window is real for the delivery lanes (the fleet can reap a stuck dispatcher at any
    moment) and it is the same defect: nobody is left to read the child's answer.
    """
    probe = subprocess.Popen([sys.executable, "-c", "pass"])
    probe.wait(timeout=10)
    dead_pid = probe.pid

    code = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from hermes_cli import quiet_single_query as qsq
        gone = {{"flag": False}}
        qsq.arm_requester_watchdog({dead_pid}, lambda: gone.__setitem__("flag", True))
        # A cadence-delayed policy would still be parked in its first wait here; the check must
        # come BEFORE that wait, so this loop is out in well under one cadence.
        deadline = time.monotonic() + 2.0
        while not gone["flag"] and time.monotonic() < deadline:
            time.sleep(0.01)
        sys.exit(qsq.REQUESTER_DEATH_EXIT_CODE if gone["flag"] else 0)
        """
    )
    try:
        rc = _run_child(code, timeout=10.0)
    except subprocess.TimeoutExpired:
        pytest.fail("the child waited out a poll cadence despite its requester already being gone")
    assert rc == qsq.REQUESTER_DEATH_EXIT_CODE


@LINUX_ONLY
def test_an_unarmed_child_runs_to_its_own_end():
    """No spawner armed it (a person's ``hermes chat -Q``): the child is never ended by this."""
    code = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from hermes_cli import quiet_single_query as qsq
        gone = {{"flag": False}}
        qsq.arm_requester_watchdog(None, lambda: gone.__setitem__("flag", True))
        time.sleep(1.5)
        sys.exit(qsq.REQUESTER_DEATH_EXIT_CODE if gone["flag"] else 0)
        """
    )
    assert _run_child(code) == 0
