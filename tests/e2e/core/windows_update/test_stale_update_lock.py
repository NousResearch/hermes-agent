"""An update killed while it pauses the gateway, then the next ``hermes update``, on native Windows.

Failure class (#149520): an update interrupted after it paused the gateway leaves
``.hermes-update-in-progress.lock`` (0 bytes) and the marker of a dead process behind. The next
``hermes update`` must not wait on that lock: its owner is gone, and the kernel freed the lock with
it. The update reclaims the dead claim, lands, and the gateway comes back. A lock that a LIVE process
still holds is different: that update refuses at once (exit 2) and leaves the running gateway alone,
because the gateway is paused only after the update holds the lock.

One install, a gateway running through the published launcher, then:

(a) ``hermes update`` killed (``taskkill /T /F``) once it has asked the gateway to pause;
    the dead update's 0-byte lock file is planted again in case the kill raced its creation;
(b) the next ``hermes update`` finishes within the bound, lands the target, leaves no
    marker, and a gateway runs again;
(c) a live process holds ``.hermes-update-in-progress.lock`` with no sharing (the lock every marker
    mutation takes) over a dead marker: ``hermes update`` exits 2 within the bound, the
    checkout stays at HEAD, and the gateway that was running keeps running (same pid);
(d) once that holder exits, ``hermes update`` lands the target.
"""

from __future__ import annotations

import subprocess
import sys
import time

import psutil
import pytest

from tests.e2e.core.windows._helpers import wait_until
from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    Journey,
    fail_with,
    new_machine,
    taskkill_tree,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

MARKER = ".hermes-update-in-progress"
SIDECAR = MARKER + ".lock"
# The reporter's update sat for minutes; a working one here takes well under this.
RECOVERY_BOUND = 900.0
REFUSAL_BOUND = 120.0
PAUSE_ASKED = "Stopping Windows gateway process(es)"

# A wedged peer: the sidecar opened with NO sharing (FileShare.None in marker.ps1, the Python mutex).
_HOLDER = r"""
import ctypes, sys, time
from ctypes import wintypes
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                          wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k.CreateFileW.restype = wintypes.HANDLE
h = k.CreateFileW(sys.argv[1], 0xC0000000, 0, None, 4, 0x80, None)
assert h and h != ctypes.c_void_p(-1).value, ctypes.get_last_error()
print("held", flush=True)
time.sleep(900)
"""


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).is_running()
    except psutil.Error:
        return False


def _timed(fn):
    started = time.monotonic()
    run = fn()
    return run, time.monotonic() - started


def _transcript(proc) -> str:
    try:  # the pump thread creates the file a moment after the spawn
        return proc.transcript.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _kill_once_pause_asked(machine) -> dict:
    proc = machine.spawn_logged([str(machine.hermes_exe), "update", "--yes"], "update-killed")
    try:
        wait_until(lambda: PAUSE_ASKED in _transcript(proc) or proc.poll() is not None, 900,
                   "the update to ask the gateway to pause", interval=0.2)
    finally:
        asked = PAUSE_ASKED in _transcript(proc)
        exited = proc.poll()
        killed = taskkill_tree(proc.pid)
        wait_until(lambda: proc.poll() is not None, 60, "the killed update to exit")
    # The dead update's leftovers as the reporter found them: a 0-byte lock beside its marker.
    sidecar = machine.hermes_home / SIDECAR
    if not sidecar.exists():
        sidecar.write_bytes(b"")
    return {"asked": asked, "exited_before_kill": exited, "taskkill": killed.returncode,
            "marker_left": (machine.hermes_home / MARKER).exists()}


def _gateway_pid(machine) -> int:
    return int(machine.gateway_state().get("pid") or 0)


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("stl"), srv.base_url, label="stl", system_git=True)
        j = Journey(machine)
        try:
            install = j.step("install", machine.install)
            if j.ok("install"):
                j.step("install_ok", lambda: j.require("install", install.returncode == 0,
                                                        f"install.ps1 exited {install.returncode}", install))
            if j.ok("install_ok"):
                with machine.gateway_phase():
                    machine.spawn_gateway()
                    before = j.step("gateway_before", machine.wait_gateway_running)
                    machine.advance()
                    j.step("killed", lambda: _kill_once_pause_asked(machine))
                    j.step("after_kill", lambda: _timed(lambda: machine.update(label="update-after-kill")))
                    j.step("head_after_kill", machine.installed_head)
                    j.step("marker_after_kill", lambda: (machine.hermes_home / MARKER).exists())
                    old = int(before.get("pid") or 0) if isinstance(before, dict) else None
                    if not machine.gateway_state() or not _alive(_gateway_pid(machine)):
                        machine.spawn_gateway()  # (c) needs a running gateway either way
                    j.step("gateway_after_kill", lambda: machine.wait_gateway_running(not_pid=old))

                    third = machine.mint(machine.next, "held")
                    machine.publish(third)
                    j.results["third"] = third
                    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(machine.hermes_home / SIDECAR)],
                                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                    machine._spawned.append(holder)
                    j.step("holder", lambda: j.require("holder", holder.stdout.readline().strip() == b"held",
                                                       "harness: the sidecar holder did not take the lock"))
                    (machine.hermes_home / MARKER).write_text(f"999999\n{int(time.time())}\nct:1.000\n",
                                                              encoding="utf-8")
                    gw = _gateway_pid(machine)
                    j.results["gateway_pid_held"] = gw
                    j.step("held", lambda: _timed(lambda: machine.update(label="update-while-held")))
                    j.step("head_held", machine.installed_head)
                    j.results["gateway_alive_held"] = _alive(gw) and _gateway_pid(machine) == gw
                    holder.kill()
                    holder.wait(timeout=60)
                    j.step("after_holder", lambda: machine.update(label="update-after-holder"))
                    j.step("head_after_holder", machine.installed_head)
                    machine.kill_owned()
            yield j
        finally:
            machine.teardown()


def test_next_update_reclaims_the_killed_updates_lock(journey: Journey) -> None:
    m, killed = journey.machine, journey["killed"]
    assert killed["asked"] and killed["exited_before_kill"] is None, fail_with(
        m, f"premise: the first update was not killed while it paused the gateway ({killed})")
    run, took = journey["after_kill"]
    assert run.returncode == 0 and took < RECOVERY_BOUND and journey["head_after_kill"] == m.next, fail_with(
        m, f"after an update killed mid-pause, the next `hermes update` did not land within "
           f"{RECOVERY_BOUND:.0f}s (rc={run.returncode}, {took:.0f}s, checkout {journey['head_after_kill']}, "
           f"NEXT {m.next})", run)
    assert not journey["marker_after_kill"], fail_with(m, "the update left its marker behind", run)


def test_gateway_runs_after_the_recovered_update(journey: Journey) -> None:
    assert isinstance(journey.results.get("gateway_after_kill"), dict), fail_with(
        journey.machine, "no gateway ran after the update that followed the killed one", journey["after_kill"][0])


def test_live_lock_holder_refuses_fast_without_pausing_the_gateway(journey: Journey) -> None:
    m = journey.machine
    run, took = journey["held"]
    assert run.returncode == 2 and took < REFUSAL_BOUND and journey["head_held"] == m.next, fail_with(
        m, f"with the update lock held by a live process, `hermes update` did not refuse within "
           f"{REFUSAL_BOUND:.0f}s and leave the checkout (rc={run.returncode}, {took:.0f}s, "
           f"checkout {journey['head_held']})", run)
    assert journey.results["gateway_alive_held"] and "Paused gateway" not in run.stdout, fail_with(
        m, f"the refused update paused or replaced the running gateway (pid {journey.results['gateway_pid_held']})", run)


def test_update_lands_once_the_holder_exits(journey: Journey) -> None:
    m, run = journey.machine, journey["after_holder"]
    assert run.returncode == 0 and journey["head_after_holder"] == journey.results["third"], fail_with(
        m, f"after the lock holder exited, `hermes update` did not land (rc={run.returncode})", run)
