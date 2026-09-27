"""Git and process-holder edges of ``hermes update`` on native Windows.

Failure class: git on Windows.

* ``install.ps1`` clones with ``--filter=tree:0`` using the Git it pins. ``hermes update``
  must then fetch into that partial clone and land on the new commit. The fetch goes
  through the same pinned Git that has hit ``BUG: builtin/pack-objects.c`` (#124323).
* ``hermes update --list-venv-holders`` is the read-only twin of the Windows venv-holder
  guard. It must list every live process running from the install's venv and exit 3,
  so automation can stop exactly those PIDs, and print ``[]``/exit 0 once they are gone
  (#123050).
"""

from __future__ import annotations

import json
import subprocess

import psutil
import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.windows._helpers import Run, wait_until
from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    Journey,
    Machine,
    fail_with,
    failure_line,
    new_machine,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

VENV_HOLDERS_EXIT = 3


def _venv_pythons(machine: Machine) -> list:
    return sorted(machine.hermes_home.glob("installs/*/environments/*/venv/Scripts/python.exe"))


def _spawn_holders(machine: Machine) -> list[int]:
    """What keeps a venv busy on a real box: long-lived processes on its interpreter."""
    pythons = _venv_pythons(machine)
    assert pythons, f"harness: no venv\\Scripts\\python.exe under {machine.hermes_home / 'installs'}"
    pids = []
    for exe in pythons:
        proc = subprocess.Popen([str(exe), "-c", "import time; time.sleep(900)"], cwd=machine.profile,
                                env=machine.env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        machine._spawned.append(proc)
        pids.append(proc.pid)
    wait_until(lambda: all(psutil.pid_exists(p) for p in pids), 30, "venv holders to start")
    return pids


def _stop_holders(pids: list[int]) -> None:
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=60)
    wait_until(lambda: not any(psutil.pid_exists(p) for p in pids), 60, "venv holders to exit")


def _reported(run: Run) -> list[int] | None:
    text = run.stdout
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        return sorted(int(h["pid"]) for h in json.loads(text[start:end + 1]))
    except (ValueError, TypeError, KeyError):
        return None


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("git"), srv.base_url, label="git")
        j = Journey(machine)
        try:
            install = j.step("install", machine.install)
            if j.ok("install"):
                j.step("install_ok", lambda: j.require("install", install.returncode == 0,
                                                        f"install.ps1 exited {install.returncode}", install))
            if j.ok("install_ok"):
                j.step("partial_filter", lambda: machine.git_config("remote.origin.partialclonefilter"))
                machine.advance()
                j.step("update", machine.update)
                pids = j.step("holders", lambda: _spawn_holders(machine))
                j.step("list_live", lambda: machine.hermes("update", "--list-venv-holders", label="list-holders"))
                if j.ok("holders"):
                    j.step("holders_stopped", lambda: _stop_holders(pids))
                j.step("list_none", lambda: machine.hermes("update", "--list-venv-holders", label="list-none"))
            yield j
        finally:
            machine.teardown()


def test_update_fetches_into_the_installers_partial_clone(journey: Journey) -> None:
    m, run, flt = journey.machine, journey["update"], journey["partial_filter"]
    assert flt, fail_with(m, "harness: the installer's clone is not partial (no remote.origin.partialclonefilter); "
                             "serve.git should allow filters")
    fetch_bug = next((ln.strip() for ln in run.stdout.splitlines() if "BUG:" in ln or "fatal:" in ln),
                     failure_line(run))
    assert run.returncode == 0 and m.installed_head() == m.next, fail_with(
        m, f"hermes update over the installer's partial clone failed: rc={run.returncode}, checkout at "
           f"{m.installed_head()} (NEXT {m.next}); {fetch_bug or 'no git error printed'}", run)


def test_list_venv_holders_reports_live_holders(journey: Journey) -> None:
    m, pids, run = journey.machine, journey["holders"], journey["list_live"]
    reported = _reported(run)
    assert reported is not None, fail_with(m, "--list-venv-holders printed no JSON list", run)
    with known_failure(r"^`hermes update --list-venv-holders` did not report the live venv holders "
                       r".*\(rc=0, reported pids \[\]\)",
                       "gated on #123050: --list-venv-holders reads the retired hermes_cli.main stub "
                       "and always prints []"):
        assert run.returncode == VENV_HOLDERS_EXIT and set(pids) <= set(reported), fail_with(
            m, f"`hermes update --list-venv-holders` did not report the live venv holders {pids} "
               f"(rc={run.returncode}, reported pids {reported})", run)


def test_list_venv_holders_empty_once_they_exit(journey: Journey) -> None:
    m, run = journey.machine, journey["list_none"]
    journey["holders_stopped"]
    reported = _reported(run)
    assert run.returncode == 0 and reported == [], fail_with(
        m, f"with no venv holders left, --list-venv-holders exited {run.returncode} reporting {reported}", run)
