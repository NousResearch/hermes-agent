"""Two Hermes installs on one Windows host: one install's ``hermes update`` leaves the other's
gateway alone (#124659).

Failure class: update blast radius. The updater's gateway discovery scans the whole process
table; a gateway of another install (its own ``%LOCALAPPDATA%\\hermes``, its own checkout and
venv) holds none of the updating install's files, so the update must neither stop it nor try to
replay it. ``neighbour`` updates while ``host`` serves a gateway; both are real installs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest

from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    UPDATE_TIMEOUT,
    Journey,
    fail_with,
    new_machine,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

_FOREIGN_STOP = ("without profile mapping", "Force-stopped")


def _alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        host = new_machine(tmp_path_factory.mktemp("host"), srv.base_url, label="host", system_git=True)
        neighbour = new_machine(tmp_path_factory.mktemp("nbr"), srv.base_url, label="nbr", system_git=True)
        j = Journey(host)
        j.results["neighbour"] = neighbour
        try:
            with ThreadPoolExecutor(2) as pool:
                installs = list(pool.map(lambda m: m.install(), (host, neighbour)))
            j.step("installs_ok", lambda: [j.require(m.profile_name, run.returncode == 0,
                                                     f"install.ps1 exited {run.returncode}", run)
                                           for m, run in zip((host, neighbour), installs)])
            if j.ok("installs_ok"):
                # One job-wide gateway phase for both machines; ``neighbour.hermes`` (not ``.update``)
                # because the phase lock is per machine and this module holds it through ``host``.
                with host.gateway_phase():
                    j.step("spawn", host.spawn_gateway)
                    j.step("host_before", host.wait_gateway_running)
                    neighbour.advance()
                    j.step("neighbour_update", lambda: neighbour.hermes(
                        "update", "--yes", label="update", timeout=UPDATE_TIMEOUT))
                    # Liveness is read here, before ``kill_owned`` tears the gateway down.
                    j.step("host_after", lambda: {**host.gateway_state(),
                                                  "alive": _alive(int(j["host_before"].get("pid") or 0))})
                    host.kill_owned()
                    neighbour.kill_owned()
            yield j
        finally:
            neighbour.teardown()
            host.teardown()


def test_neighbour_update_leaves_this_installs_gateway_running(journey: Journey) -> None:
    m, before, run = journey.machine, journey["host_before"], journey["neighbour_update"]
    pid = int(before.get("pid") or 0)
    after = journey["host_after"]
    assert after["alive"] and after.get("pid") == pid and after.get("gateway_state") == "running", fail_with(
        m, f"another install's `hermes update` stopped this install's gateway (pid {pid}); "
           f"alive={after['alive']}, gateway_state after: {after.get('gateway_state')!r} pid {after.get('pid')}", run)


def test_neighbour_update_stops_no_foreign_gateway(journey: Journey) -> None:
    n, run = journey["neighbour"], journey["neighbour_update"]
    journey["host_before"]  # a foreign gateway was running during the update
    stopped = [line.strip() for line in run.stdout.splitlines() if any(s in line for s in _FOREIGN_STOP)]
    assert run.returncode == 0 and not stopped, fail_with(
        n, f"`hermes update` without a gateway of its own reported stopping gateways "
           f"(rc={run.returncode}): {stopped}", run)
