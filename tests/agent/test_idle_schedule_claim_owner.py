"""The idle-job run claim is taken over as soon as its owner is provably dead, never from a live owner.

The curator and every plugin idle task hold this claim for one run at a time across processes. A
gateway that crashed (or was SIGKILLed) mid-run used to keep its claim for an hour, so the task
stayed suppressed after the restart. The claim records its owner's pid, process start and machine:
a dead owner, or a pid that now names another process, loses the claim at once; an owner that
cannot be checked (a bare pid written by an older Hermes) keeps the old hour-old rule.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent.idle_schedule import CLAIM_STALE_SECONDS, claim_run, release_run

pytestmark = pytest.mark.platforms("linux")  # pid start + machine identity come from /proc

_ROOT = Path(__file__).resolve().parents[2]
_HOLDER = """
import sys
from pathlib import Path
from agent.idle_schedule import claim_run
print(claim_run(Path(sys.argv[1])), flush=True)
sys.stdin.read()  # hold the claim until the test closes stdin
"""


def _holder(lock: Path) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    return subprocess.Popen([sys.executable, "-c", _HOLDER, str(lock)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, env=env, cwd=_ROOT)


def _claimed(proc: subprocess.Popen) -> bool:
    return proc.stdout.readline().strip() == "True"


def _age(lock: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(lock, (old, old))


def _end(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait(10)


def test_a_claim_of_a_killed_owner_is_taken_over_at_once(tmp_path):
    lock = tmp_path / ".task.lock"
    owner = _holder(lock)
    try:
        assert _claimed(owner)
        assert claim_run(lock) is False  # the owner is alive
    finally:
        _end(owner)  # SIGKILL: no release ran
    assert lock.exists()
    assert claim_run(lock) is True
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    release_run(lock)
    assert not lock.exists()


def test_a_killed_owners_claim_rewritten_with_a_bom_is_still_taken_over_at_once(tmp_path):
    lock = tmp_path / ".task.lock"
    owner = _holder(lock)
    try:
        assert _claimed(owner)
    finally:
        _end(owner)
    lock.write_bytes(b"\xef\xbb\xbf" + lock.read_bytes())  # Windows tooling BOMs files it touches
    assert claim_run(lock) is True


def test_a_live_owner_keeps_its_claim_however_old_it_is(tmp_path):
    lock = tmp_path / ".task.lock"
    owner = _holder(lock)
    try:
        assert _claimed(owner)
        _age(lock, CLAIM_STALE_SECONDS + 60)
        assert claim_run(lock) is False
        release_run(lock)  # not this process's claim
        assert lock.exists()
    finally:
        _end(owner)


def test_a_pid_reused_by_another_process_does_not_keep_the_claim(tmp_path):
    lock = tmp_path / ".task.lock"
    other = _holder(tmp_path / ".other.lock")  # a live process now holding the recorded pid
    try:
        assert _claimed(other)
        record = json.loads((tmp_path / ".other.lock").read_text())
        record["start"] -= 1  # the owner that wrote this claim started earlier: it was another process
        lock.write_text(json.dumps(record))
        assert claim_run(lock) is True
    finally:
        _end(other)


def test_a_bare_pid_claim_keeps_the_hour_rule(tmp_path):
    lock = tmp_path / ".task.lock"
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(10)
    lock.write_text(str(gone.pid))  # an older Hermes's claim: no start time to check the owner by
    assert claim_run(lock) is False
    _age(lock, CLAIM_STALE_SECONDS + 1)
    assert claim_run(lock) is True


def test_a_claim_from_another_machine_keeps_the_hour_rule(tmp_path):
    lock = tmp_path / ".task.lock"
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(10)
    lock.write_text(json.dumps({"pid": gone.pid, "start": 1, "machine": "another-boot:1"}))
    assert claim_run(lock) is False  # its pid says nothing about the other machine's processes
    _age(lock, CLAIM_STALE_SECONDS + 1)
    assert claim_run(lock) is True


def test_racing_claimants_of_a_dead_owners_claim_have_one_winner(tmp_path):
    lock = tmp_path / ".task.lock"
    owner = _holder(lock)
    assert _claimed(owner)
    _end(owner)
    racers = [_holder(lock) for _ in range(6)]
    try:
        results = [_claimed(r) for r in racers]  # each keeps running, so a winner stays a live owner
    finally:
        for r in racers:
            _end(r)
    assert results.count(True) == 1, results


def test_without_proc_a_killed_owner_is_still_taken_over_at_once(tmp_path):
    """macOS and Windows have no /proc boot id: the claim names the host and its boot time instead."""
    lock = tmp_path / ".task.lock"
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    holder = (
        "import sys\n"
        "from pathlib import Path\n"
        "import gateway.drain_control as d\n"
        "d.current_instantiation_epoch = lambda: ''\n"
        "from agent.idle_schedule import claim_run\n"
        "print(claim_run(Path(sys.argv[1])), flush=True)\n"
        "sys.stdin.read()\n"
    )
    owner = subprocess.Popen([sys.executable, "-c", holder, str(lock)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True, env=env, cwd=_ROOT)
    try:
        assert _claimed(owner)
        record = json.loads(lock.read_text(encoding="utf-8"))
        assert str(record["machine"]).startswith("host:")
    finally:
        _end(owner)
    from gateway import drain_control

    original = drain_control.current_instantiation_epoch
    drain_control.current_instantiation_epoch = lambda: ""
    try:
        assert claim_run(lock) is True
    finally:
        drain_control.current_instantiation_epoch = original
        release_run(lock)
