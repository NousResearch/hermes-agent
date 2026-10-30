"""A stalled ``docker exec`` is one failed poll attempt, never the poll's verdict.

Driven without Docker: ``docker_exec_sh`` is replaced by a scripted stand-in and the poll's clock
is fake, so the timeout branches run on every host (DOCKER_DAEMON_REQUIRED opts out of the skip).
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from tests.docker import conftest as docker_conftest
from tests.docker import test_dashboard

DOCKER_DAEMON_REQUIRED = False


class _Clock:
    """``time`` stand-in: each probe attempt and each sleep advance the fake clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _scripted_exec(clock: _Clock, outcomes: list):
    def fake(container, command, *, user="hermes", timeout=30):
        clock.now += timeout if outcomes[0] == "stall" else 0.1
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if outcome == "stall":
            raise subprocess.TimeoutExpired(["docker", "exec"], timeout)
        return outcome
    return fake


@pytest.mark.parametrize("module", [docker_conftest, test_dashboard], ids=["poll_container", "_http_probe"])
def test_stalled_exec_is_retried_until_the_deadline(monkeypatch, module):
    clock = _Clock()
    monkeypatch.setattr(module, "time", clock)
    ok = SimpleNamespace(returncode=0, stdout="200\n{}", stderr="")
    monkeypatch.setattr(module, "docker_exec_sh", _scripted_exec(clock, ["stall", ok]))
    if module is docker_conftest:
        assert module.poll_container("c", "probe", deadline_s=30.0) == (True, "200\n{}")
    else:
        assert module._http_probe("c", "/api/status", deadline_s=30.0) == (200, "{}")

    clock.now = 0.0
    monkeypatch.setattr(module, "docker_exec_sh", _scripted_exec(clock, ["stall"]))
    if module is docker_conftest:
        with pytest.raises(AssertionError, match="docker exec stalled"):
            module.poll_container("c", "probe", deadline_s=12.0)
    else:
        with pytest.raises(AssertionError, match="docker exec stalled"):
            module._http_probe("c", "/api/status", deadline_s=12.0)
    assert clock.now <= 12.0  # each attempt gets only the time left, so the deadline holds


def test_fail_closed_dashboard_guard_needs_a_completed_probe(monkeypatch):
    """Every exec stalling proves nothing about the dashboard, so the negative guard must not pass."""
    clock = _Clock()
    monkeypatch.setattr(docker_conftest, "time", clock)
    monkeypatch.setattr(docker_conftest, "docker_exec_sh", _scripted_exec(clock, ["stall"]))
    monkeypatch.setattr(test_dashboard, "start_container", lambda *args, **kwargs: "c")
    with pytest.raises(AssertionError, match="stalled"):
        test_dashboard.test_dashboard_insecure_env_var_no_longer_bypasses_gate("image", "c")


@pytest.mark.parametrize(
    ("returncode", "stdout", "failure"),
    [
        (7, "rc=7\n", None),  # connection refused: the real fail-closed signal
        (0, "rc=0\n", "must NOT serve"),
        (1, "", "refusal"),  # docker exec into a container that already exited
        (127, "rc=127\n", "refusal"),  # curl missing
    ],
    ids=["refused", "served", "container-gone", "curl-missing"],
)
def test_fail_closed_dashboard_guard_needs_an_observed_refusal(monkeypatch, returncode, stdout, failure):
    """A completed probe that never reached the port proves nothing either: only a refusal passes."""
    clock = _Clock()
    monkeypatch.setattr(docker_conftest, "time", clock)
    result = SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(docker_conftest, "docker_exec_sh", _scripted_exec(clock, [result]))
    monkeypatch.setattr(test_dashboard, "start_container", lambda *args, **kwargs: "c")
    if failure is None:
        test_dashboard.test_dashboard_insecure_env_var_no_longer_bypasses_gate("image", "c")
    else:
        with pytest.raises(AssertionError, match=failure):
            test_dashboard.test_dashboard_insecure_env_var_no_longer_bypasses_gate("image", "c")


def test_stall_after_a_completed_probe_keeps_its_output(monkeypatch):
    """A late stall must not replace the last real probe output that positive callers print."""
    clock = _Clock()
    monkeypatch.setattr(docker_conftest, "time", clock)
    failed = SimpleNamespace(returncode=1, stdout="HOME=/root", stderr="")
    monkeypatch.setattr(docker_conftest, "docker_exec_sh", _scripted_exec(clock, [failed, "stall"]))
    assert docker_conftest.poll_container("c", "probe", deadline_s=12.0) == (False, "HOME=/root")
