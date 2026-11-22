"""A checkout whose ``.git`` git rejects, on native Windows.

Failure class: a damaged checkout strands the fleet. A killed update or a sync tool can leave
``.git`` in place with ``HEAD`` gone; git then says "not a git repository" to every command.
``hermes update`` pauses the running gateway before it fetches, so the fetch failure used to
leave the gateway stopped with a promise that "the next `hermes` command restarts them", which no
command kept while ``HEAD`` stayed unreadable. The user's way out is rerunning the installer, which
moves the damaged checkout aside and clones fresh.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    Journey,
    fail_with,
    new_machine,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("dmg"), srv.base_url, label="dmg", system_git=True)
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
                    (machine.install_dir / ".git" / "HEAD").unlink()
                    j.step("update", machine.update)
                    old_pid = int(before.get("pid") or 0) if isinstance(before, dict) else None
                    j.step("state_after", lambda: machine.wait_gateway_running(not_pid=old_pid))
                    machine.kill_owned()
                j.step("reinstall", machine.install)
                j.step("head_after_reinstall", machine.installed_head)
                j.step("update_after_reinstall", lambda: machine.update(label="update-after-reinstall"))
            yield j
        finally:
            machine.teardown()


def test_update_names_the_damaged_checkout_and_its_repair(journey: Journey) -> None:
    m, run = journey.machine, journey["update"]
    assert run.returncode != 0 and "damaged" in run.stdout and "install.ps1" in run.stdout, fail_with(
        m, f"hermes update on a checkout git rejects did not name the damage and the installer repair "
           f"(rc={run.returncode})", run)


def test_paused_gateway_comes_back_when_the_update_cannot_fetch(journey: Journey) -> None:
    m, run = journey.machine, journey["update"]
    journey["state_before"]  # the precondition: a gateway was running when the update began
    assert isinstance(journey.results.get("state_after"), dict), fail_with(
        m, "the gateway the update paused stayed stopped after the fetch failed on a damaged checkout", run)


def test_rerunning_the_installer_repairs_the_checkout(journey: Journey) -> None:
    m, reinstall = journey.machine, journey["reinstall"]
    parked = list(m.hermes_home.glob("hermes-agent.broken-*"))
    head, update = journey["head_after_reinstall"], journey["update_after_reinstall"]
    assert reinstall.returncode == 0 and parked and len(head) == 40, fail_with(
        m, f"rerunning install.ps1 over the damaged checkout did not repair it (rc={reinstall.returncode}, "
           f"parked={[p.name for p in parked]}, head={head!r})", reinstall)
    assert update.returncode == 0, fail_with(
        m, f"hermes update after the installer repair exited {update.returncode}", update)
