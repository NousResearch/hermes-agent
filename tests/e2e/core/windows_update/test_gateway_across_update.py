"""A running gateway across ``hermes update`` on native Windows.

Failure class: gateway hand-off. A user (or the Desktop / login item) hosts
``hermes gateway run`` through the published ``hermes.exe``; ``hermes update`` must stop
it, update the checkout, relaunch it on the new commit, and leave it where every later
command can find it: ``hermes gateway status``, ``hermes gateway stop`` and the next
``hermes update``'s own pause step.
"""

from __future__ import annotations

import re

import psutil
import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.windows._helpers import wait_until
from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    Journey,
    fail_with,
    new_machine,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

_INVISIBLE = ("gated on #124318: the gateway the updater relaunches holds its lock without gateway.pid "
              "metadata, so `hermes gateway status`/`stop` and the next update's pause cannot see it")
_RESTART_FAILURE = re.compile(
    r"^.*(recovery failed|restart could not be verified|restart incomplete|not verified alive).*$", re.M)
_STATUS_LINE = re.compile(r"^.*[✓✗].*[Gg]ateway.*$", re.M)


def _status_line(text: str) -> str:
    found = _STATUS_LINE.findall(text)
    return found[0].strip() if found else "<no status line>"


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _exits(pid: int, timeout: float) -> bool:
    try:
        wait_until(lambda: not _alive(pid), timeout, f"gateway pid {pid} to exit", interval=0.5)
    except AssertionError:
        return False
    return True


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("gw"), srv.base_url, label="gw", system_git=True)
        j = Journey(machine)
        try:
            install = j.step("install", machine.install)
            if j.ok("install"):
                j.step("install_ok", lambda: j.require("install", install.returncode == 0,
                                                        f"install.ps1 exited {install.returncode}", install))
            if j.ok("install_ok"):
                with machine.gateway_phase():
                    j.step("spawn", machine.spawn_gateway)
                    before = j.step("state_before", machine.wait_gateway_running)
                    j.step("status_before", lambda: machine.hermes("gateway", "status", label="status-before"))
                    machine.advance()
                    j.step("update", machine.update)
                    old_pid = int(before.get("pid") or 0) if isinstance(before, dict) else None
                    j.step("state_after", lambda: machine.wait_gateway_running(not_pid=old_pid))
                    j.step("status_after", lambda: machine.hermes("gateway", "status", label="status-after"))
                    j.step("pidfile_after", lambda: (machine.hermes_home / "gateway.pid").exists())
                    j.step("update_again", lambda: machine.update(label="update-again"))
                    last = j.step("state_before_stop", machine.gateway_state)
                    j.step("stop", lambda: machine.hermes("gateway", "stop", label="stop"))
                    last_pid = int(last.get("pid") or 0) if isinstance(last, dict) else 0
                    j.step("stopped", lambda: _exits(last_pid, 90))
                    machine.kill_owned()  # nothing of this machine outlives its gateway phase
            yield j
        finally:
            machine.teardown()


def test_launcher_started_gateway_is_visible_before_update(journey: Journey) -> None:
    m, state, status = journey.machine, journey["state_before"], journey["status_before"]
    line = _status_line(status.stdout)
    assert status.returncode == 0 and line.startswith("✓") and str(state.get("pid")) in line, fail_with(
        m, f"the launcher-started gateway (pid {state.get('pid')}) is invisible to `hermes gateway status` "
           f"before any update: {_status_line(status.stdout)!r}", status)


def test_update_with_running_gateway_succeeds(journey: Journey) -> None:
    m, run = journey.machine, journey["update"]
    journey["state_before"]  # the precondition: a gateway was running when the update began
    failure = _RESTART_FAILURE.search(run.stdout)
    with known_failure(r"^hermes update with a running gateway reported a failed gateway restart "
                       r"\(rc=\d+\): ⚠ Windows gateway restart could not be verified",
                       _INVISIBLE):
        assert run.returncode == 0 and failure is None, fail_with(
            m, f"hermes update with a running gateway reported a failed gateway restart "
               f"(rc={run.returncode}): {failure.group(0).strip() if failure else '<no restart message>'}", run)


def test_gateway_serves_next_after_update(journey: Journey) -> None:
    m, state = journey.machine, journey["state_after"]
    sha = str(state.get("code_sha") or "")  # state_after only resolves for a live, running pid
    assert len(sha) >= 7 and m.next.startswith(sha), fail_with(
        m, f"the relaunched gateway serves code_sha={sha!r}, expected NEXT {m.next}")


def test_gateway_discoverable_after_update(journey: Journey) -> None:
    m, state, status = journey.machine, journey["state_after"], journey["status_after"]
    pid, pidfile = int(state.get("pid") or 0), journey["pidfile_after"]
    line = _status_line(status.stdout)
    with known_failure(r"^after update the serving gateway \(pid \d+\) is invisible",
                       _INVISIBLE):
        assert status.returncode == 0 and str(pid) in line and pidfile, fail_with(
            m, f"after update the serving gateway (pid {pid}) is invisible: `hermes gateway status` says "
               f"{line!r}; gateway.pid present={pidfile}", status)


def test_next_update_is_not_blocked(journey: Journey) -> None:
    m, run = journey.machine, journey["update_again"]
    journey["state_after"]  # a gateway was running (the relaunched one)
    blocked = "Could not map Windows gateway PIDs" in run.stdout
    with known_failure(r"^the next hermes update is blocked at the gateway pause step \(rc=1, "
                       r"'Could not map Windows gateway PIDs' printed=True\)",
                       _INVISIBLE):
        assert run.returncode == 0 and not blocked, fail_with(
            m, f"the next hermes update is blocked at the gateway pause step (rc={run.returncode}, "
               f"'Could not map Windows gateway PIDs' printed={blocked})", run)


def test_gateway_stop_after_update(journey: Journey) -> None:
    m, stop, state = journey.machine, journey["stop"], journey["state_before_stop"]
    pid = int(state.get("pid") or 0)
    assert pid, fail_with(m, f"no gateway recorded before stop: {state}")
    with known_failure(r"^after update `hermes gateway stop` left the serving gateway \(pid \d+\) running",
                       _INVISIBLE):
        assert journey["stopped"], fail_with(
            m, f"after update `hermes gateway stop` left the serving gateway (pid {pid}) running", stop)
    assert stop.returncode == 0, fail_with(m, f"hermes gateway stop exited {stop.returncode}", stop)
