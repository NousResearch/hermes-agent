"""process_identity.release_process_group_once_empty: release only an empty group, waiting for reaps."""

import os
import signal
import subprocess
import threading

import pytest

from hermes_cli import process_identity

@pytest.fixture
def released(monkeypatch):
    calls: list = []
    monkeypatch.setattr(process_identity, "release_process_groups", lambda pgids: calls.append(list(pgids)))
    return calls


def _leader(script: str) -> subprocess.Popen:
    return subprocess.Popen(["sh", "-c", script], stdin=subprocess.DEVNULL,
                            start_new_session=True)  # detached-spawn: ok — test stand-in, killed in finally


@pytest.mark.platforms("posix")
def test_an_empty_group_is_released_even_with_an_unreaped_leader(released):
    leader = _leader("exit 0")
    # Not wait()ed: the zombie leader keeps killpg(pgid, 0) succeeding until proc.poll() reaps it.
    assert process_identity.release_process_group_once_empty(leader.pid, proc=leader, timeout=5)
    assert released == [[leader.pid]]


@pytest.mark.platforms("posix")
def test_a_group_with_a_live_member_stays_covered(released):
    leader = _leader("trap '' TERM; exec sleep 60")
    try:
        assert not process_identity.release_process_group_once_empty(leader.pid, proc=leader, timeout=0.2)
        assert released == []
    finally:
        os.killpg(leader.pid, signal.SIGKILL)  # windows-footgun: ok — posix-marked test
        leader.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_it_waits_for_a_member_that_exits_late(released):
    leader = _leader("exec sleep 60")
    threading.Timer(0.3, lambda: os.killpg(leader.pid, signal.SIGKILL)).start()  # windows-footgun: ok — posix-marked test
    try:
        assert process_identity.release_process_group_once_empty(leader.pid, proc=leader, timeout=10)
        assert released == [[leader.pid]]
    finally:
        if leader.poll() is None:
            leader.kill()
        leader.wait(timeout=10)


@pytest.mark.platforms("windows")
def test_windows_never_releases(released):
    """Windows children die through the job object; there is no supervised group to release."""
    assert not process_identity.release_process_group_once_empty(os.getpid(), timeout=0)
    assert released == []
