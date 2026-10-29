"""A stalled ``docker exec`` is one failed poll attempt, never the poll's verdict.

Driven without Docker: ``docker_exec_sh`` is replaced by a scripted stand-in and the poll's clock
is fake, so the timeout branches run on every host.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from tests.docker import conftest as docker_conftest
from tests.docker import test_dashboard


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
        assert module.poll_container("c", "probe", deadline_s=30.0) == (False, "")
    else:
        with pytest.raises(AssertionError, match="docker exec stalled"):
            module._http_probe("c", "/api/status", deadline_s=30.0)
