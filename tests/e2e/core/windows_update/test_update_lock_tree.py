"""One ``hermes update`` owns an installed checkout on native Windows, and its whole tree.

Failure class: concurrent updates of one install (contract C1.7). The update lock lives on the
install's git dir (an msvcrt byte lock on ``<install>/.git/hermes-update.lock``), not in HERMES_HOME, so a
second ``hermes update`` from ANOTHER home cannot mutate the checkout while the first runs. On
Windows a child cannot inherit that lock, so the owner binds every update-tree child into a
kill-on-close job: ``taskkill /F`` of the owner kills the child too, and only then is the lock free.

One install (install.ps1), then, with NEXT published:
(a) an update owner holds the lock from another HERMES_HOME with a bound child running; the real
    ``hermes update`` of the installed checkout must exit 2 and leave the checkout at HEAD;
(b) ``taskkill /F`` of the owner (not ``/T``) must take the bound child down with it;
(c) with the tree gone the lock is free: the real ``hermes update`` lands NEXT.

A second install proves the other direction of the job: a gateway the update RESTARTS is started
with ``CREATE_BREAKAWAY_FROM_JOB`` and must leave the job, so it outlives a successful update.
"""

from __future__ import annotations

import subprocess
import time

import psutil
import pytest

from tests.e2e.core.windows._helpers import wait_until
from tests.e2e.core.windows_update._machine import (
    REQUIRES_OPT_IN,
    fail_with,
    harness_git,
    new_machine,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

REFUSAL = "Another Hermes update is already running"
SURVIVAL_SECONDS = 30

# Runs on the INSTALLED venv against the installed checkout's own update_lock. Degrades to the
# marker-only API where the checkout has no install-root lock, so an older tree fails on behaviour.
_OWNER = r"""
import inspect, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_lock as ul
kw = {"install_root": sys.argv[1]} if "install_root" in inspect.signature(ul.UpdateLock).parameters else {}
lock = ul.UpdateLock(**kw)
assert lock.acquire(), "owner could not take the update lock"
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(900)"],
                         stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
getattr(ul, "bind_child_to_update_tree", lambda proc: None)(child)
print(f"{child.pid} {bool(kw)}", flush=True)
time.sleep(900)
"""


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).is_running()
    except psutil.Error:
        return False


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    out: dict = {}
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("lk"), srv.base_url, label="lk")
        out["machine"] = machine
        try:
            install = machine.install()
            assert install.returncode == 0, fail_with(machine, f"install.ps1 exited {install.returncode}", install)
            pythons = sorted(machine.hermes_home.glob("installs/*/environments/*/venv/Scripts/python.exe"))
            assert pythons, fail_with(machine, "harness: the install has no venv python")
            other_home = machine.root / "other-home"
            other_home.mkdir()
            machine.advance()
            owner = subprocess.Popen(
                [str(pythons[0]), "-c", _OWNER, str(machine.install_dir)], cwd=machine.profile,
                env=machine.env({"HERMES_HOME": str(other_home)}), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            machine._spawned.append(owner)
            assert owner.stdout is not None
            first = owner.stdout.readline().decode("utf-8", "replace")
            line = first.split()
            if len(line) != 2 or not line[0].isdigit():
                owner.kill()
                rest = owner.communicate(timeout=60)[0].decode("utf-8", "replace")
                raise AssertionError(fail_with(machine, f"owner failed to start:\n{first}{rest}"))
            child, out["root_lock"] = int(line[0]), line[1] == "True"
            out["child_alive"] = _alive(child)
            out["held_status"] = harness_git("-C", str(machine.install_dir), "status", "--porcelain",
                                             "--untracked-files=all")
            # (a) a real update of the same checkout from the machine's own home
            out["a_update"] = machine.update(label="update-while-held")
            out["a_head"] = machine.installed_head()
            # (b) kill the owner only; the job must take its child down
            subprocess.run(["taskkill", "/F", "/PID", str(owner.pid)], capture_output=True, timeout=60, check=False)
            owner.wait(timeout=60)
            try:
                wait_until(lambda: not _alive(child), 20, "the bound child to die with its owner")
                out["b_child_dead"] = True
            except AssertionError:
                out["b_child_dead"] = False
                subprocess.run(["taskkill", "/F", "/PID", str(child)], capture_output=True, timeout=60, check=False)
            # (c) the tree is gone: the lock is free
            out["c_update"] = machine.update(label="update-after-tree")
            out["c_head"] = machine.installed_head()
            yield out
        finally:
            machine.teardown()


def test_second_home_update_is_refused_while_the_checkout_is_held(journey) -> None:
    m, run = journey["machine"], journey["a_update"]
    assert journey["child_alive"], fail_with(m, "premise: the owner's update-tree child never ran")
    assert journey["held_status"] == "", fail_with(
        m, f"the held update lock shows in `git status` (autostash would take it): {journey['held_status']!r}")
    assert run.returncode == 2 and REFUSAL in run.stdout and journey["a_head"] == m.head, fail_with(
        m, f"a second `hermes update` of one checkout ran while another home's update held it "
           f"(rc={run.returncode}, checkout {journey['a_head']} vs HEAD {m.head}, "
           f"install-root lock in this tree: {journey['root_lock']})", run)


def test_killed_owner_takes_its_update_tree_down(journey) -> None:
    assert journey["b_child_dead"], fail_with(
        journey["machine"], "`taskkill /F` of the update owner left its update-tree child running")


def test_free_lock_lets_the_next_update_land(journey) -> None:
    m, run = journey["machine"], journey["c_update"]
    assert run.returncode == 0 and journey["c_head"] == m.next, fail_with(
        m, f"after the whole update tree exited, `hermes update` did not land NEXT "
           f"(rc={run.returncode}, checkout {journey['c_head']}, NEXT {m.next})", run)


@pytest.fixture(scope="module")
def gateway_journey(tmp_path_factory):
    out: dict = {}
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("lg"), srv.base_url, label="lg", system_git=True)
        out["machine"] = machine
        try:
            install = machine.install()
            assert install.returncode == 0, fail_with(machine, f"install.ps1 exited {install.returncode}", install)
            with machine.gateway_phase():
                machine.spawn_gateway()
                old_pid = int(machine.wait_gateway_running().get("pid") or 0)
                machine.advance()
                out["update"] = machine.update(label="update-with-gateway")
                exited = time.monotonic()
                out["relaunched"] = int(machine.wait_gateway_running(not_pid=old_pid).get("pid") or 0)
                time.sleep(max(0.0, SURVIVAL_SECONDS - (time.monotonic() - exited)))
                out["alive_after"] = _alive(out["relaunched"])
                out["status"] = machine.hermes("gateway", "status", label="status-after-30s")
                machine.kill_owned()
            yield out
        finally:
            machine.teardown()


def test_restarted_gateway_outlives_the_update(gateway_journey) -> None:
    m, run, pid = gateway_journey["machine"], gateway_journey["update"], gateway_journey["relaunched"]
    assert run.returncode == 0, fail_with(m, f"`hermes update` with a running gateway exited {run.returncode}", run)
    status = gateway_journey["status"]
    assert gateway_journey["alive_after"] and str(pid) in status.stdout, fail_with(
        m, f"the gateway the update restarted (pid {pid}) died with the update's job: alive "
           f"{SURVIVAL_SECONDS}s after `hermes update` exited = {gateway_journey['alive_after']}", status)
