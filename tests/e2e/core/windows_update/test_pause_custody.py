"""Restart debt survives every custody transfer of a Windows update pause (review R7).

Failure class: the durable pause record changes hands — a recovering launch claims it, an update
folds an orphaned set into its own record — and a crash or a rival at the transfer boundary must
neither lose nor double a paused gateway. Each cell plants the exact schedule with the INSTALLED
code: a driver (the install's venv python) runs the real pause-record module and stops itself on
the named source line; every recovery is a real ``hermes gateway status`` launch through the
published ``hermes.exe`` that really restarts the real gateway.

* claim race: a launch is paused inside its claim, between taking the set and naming itself in it
  ("after rename, before identity write" on the old design); a second launch runs meanwhile. The
  paused gateway is restarted exactly once and no record survives.
* publish crash: an update dies after publishing the merged record, before retiring the claim it
  absorbed. The next launch restarts the gateway once; nothing is left behind.
* draining: the record says the update asked the gateway to stop and the gateway is still running
  (draining). A launch must keep that debt while it runs, and the first launch after it exited
  must start it again.
* readiness: a launch owes a gateway that comes back, one that never becomes ready, and an SCM
  service that cannot start. The ready one runs; only the other two stay owed.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import psutil
import pytest

from tests.e2e.core.windows_update._machine import REQUIRES_OPT_IN, fail_with, new_machine
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration,
              pytest.mark.live_system_guard_bypass, REQUIRES_OPT_IN]

_STEM = ".hermes-update-paused-gateways"
_RESTARTING = "Restarting gateway(s) paused by an interrupted"
_MISSING_SERVICE = "hermes-e2e-no-such-gateway-service"

# Every driver: the installed module, and a hook that stops on one line of it (kill or park).
_DRIVER = """
import inspect, json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli import update_pause_record as r

def stop_at(fn, texts, action, flag=None):
    lines, start = inspect.getsourcelines(fn)
    line = next(start + i for text in texts for i, t in enumerate(lines) if text in t)
    def trace(frame, event, arg):
        if event == "line" and frame.f_code is fn.__code__ and frame.f_lineno == line:
            if action == "kill":
                os._exit(71)
            print("paused", flush=True)
            while not Path(flag).exists():
                time.sleep(0.05)
        return trace
    sys.settrace(trace)

def orphan(profiles, **extra):
    # The record of an updater that died after recording (owner unowned = dead).
    r.write(r.stamp_tree({"resume_needed": True, "profiles": profiles, **extra}), owner=r.UNOWNED)
"""
# Claim boundary: the new design publishes the claim (identity inside) then retires the source;
# the old one renamed the source first and wrote its identity after.
_CLAIM_BOUNDARY = '("src.unlink()", \'body["claimer"] = identity()\')'


def _python(machine) -> Path:
    found = sorted(machine.hermes_home.glob("installs/*/environments/*/venv/Scripts/python.exe"))
    assert found, fail_with(machine, "harness: the install has no venv python")
    return found[0]


def _driver(machine, body: str, *args: str) -> subprocess.Popen:
    proc = subprocess.Popen([str(_python(machine)), "-c", _DRIVER + body, str(machine.install_dir), *args],
                            cwd=machine.profile, env=machine.env(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    machine._spawned.append(proc)
    return proc


def _run_driver(machine, body: str, *args: str, timeout: float = 300) -> tuple[int, str]:
    proc = _driver(machine, body, *args)
    out, _ = proc.communicate(timeout=timeout)
    return proc.returncode, out.decode("utf-8", "replace")


def _records(machine) -> list[Path]:
    return sorted(p for p in machine.hermes_home.glob(_STEM + "*") if p.suffix in (".json", ".claim"))


def _owed(machine) -> list[dict]:
    out = []
    for path in _records(machine):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8-sig"))["token"])
        except (OSError, ValueError, KeyError):
            continue
    return out


def _clear(machine) -> None:
    machine.kill_owned()
    for path in _records(machine):
        path.unlink(missing_ok=True)


def _running(machine, not_pid: int, timeout: float) -> int | None:
    try:
        return int(machine.wait_gateway_running(not_pid=not_pid, timeout=timeout).get("pid") or 0) or None
    except AssertionError:
        return None


def _paused_gateway(machine) -> int:
    """A real gateway, then stopped hard: the state a pause left it in. Its (dead) pid."""
    machine.spawn_gateway()
    pid = int(machine.wait_gateway_running().get("pid") or 0)
    machine.kill_owned()
    return pid


def _launch(machine, label: str):
    return machine.hermes("gateway", "status", label=label, timeout=600)


def _claim_race(machine, out: dict) -> None:
    dead = _paused_gateway(machine)
    # A claim handed back unowned (a launch that could not finish the resume): any launch may take it.
    out["race_seed"] = _run_driver(machine, f"""
orphan({{"default": {dead}}})
won = r.claim(r.record_path())
r._atomic_write(won[0], {{**won[1], "claimer": r.UNOWNED}})
""")
    flag = machine.root / "claim-race-go"
    first = _driver(machine, f"""
stop_at(r.claim, {_CLAIM_BOUNDARY}, "park", sys.argv[2])
r.recover(["status"])
""", str(flag))
    assert first.stdout is not None
    out["race_parked"] = first.stdout.readline().decode("utf-8", "replace").strip()
    out["race_second"] = _launch(machine, "claim-race-second")
    flag.touch()
    rest, _ = first.communicate(timeout=600)
    out["race_first"] = out["race_parked"] + "\n" + rest.decode("utf-8", "replace")
    out["race_running"] = _running(machine, dead, 120)
    out["race_owed"] = _owed(machine)
    _clear(machine)


def _publish_crash(machine, out: dict) -> None:
    dead = _paused_gateway(machine)
    out["publish_seed"] = _run_driver(machine, f'orphan({{"default": {dead}}})')
    out["publish_kill"] = _run_driver(machine, """
adopted, claims = r.adopt_orphans()
stop_at(r.record_pause, ("release_claims(claims)",), "kill")
r.record_pause({"resume_needed": True, "profiles": {}}, adopted, claims)
""")
    out["publish_files"] = [p.name for p in _records(machine)]
    out["publish_launch"] = _launch(machine, "after-publish-crash")
    out["publish_running"] = _running(machine, dead, 120)
    out["publish_again"] = _launch(machine, "after-publish-crash-2")
    out["publish_owed"] = _owed(machine)
    _clear(machine)


def _draining(machine, out: dict) -> None:
    # The real gateway keeps running: it was asked to stop and has not exited yet (draining).
    machine.spawn_gateway()
    live = int(machine.wait_gateway_running().get("pid") or 0)
    updater = _driver(machine, f"""
pid = {live}
token = {{"resume_needed": True, "profiles": {{"default": pid}}, "identities": {{str(pid): r.identity(pid)["ct"]}}}}
if hasattr(r, "mark_stop_requested"):
    r.mark_stop_requested(r.record_pause(token, None, []), [pid])
else:  # the old design has no stop record: write the same fact it would have needed
    r.write(r.stamp_tree({{**token, "stop_requested": [str(pid)]}}), owner=r.identity())
print("asked", flush=True)
time.sleep(600)
""")
    assert updater.stdout is not None
    out["drain_asked"] = updater.stdout.readline().decode("utf-8", "replace").strip()
    subprocess.run(["taskkill", "/PID", str(updater.pid), "/T", "/F"], capture_output=True, timeout=60)
    updater.wait(timeout=60)
    out["drain_live_before"] = psutil.pid_exists(live)
    out["drain_launch_while"] = _launch(machine, "while-draining")
    out["drain_owed_while"] = _owed(machine)
    subprocess.run(["taskkill", "/PID", str(live), "/T", "/F"], capture_output=True, timeout=60)
    out["drain_launch_after"] = _launch(machine, "after-drained")
    out["drain_running_after"] = _running(machine, live, 120)
    out["drain_owed_after"] = _owed(machine)
    _clear(machine)


def _readiness(machine, out: dict) -> None:
    dead = _paused_gateway(machine)
    out["ready_seed"] = _run_driver(machine, f"""
orphan({{"default": {dead}, "ghost": {dead}}}, services=["{_MISSING_SERVICE}"],
       expected_services=["{_MISSING_SERVICE}"], restarted_services=[],
       service_profiles={{"{_MISSING_SERVICE}": "svc"}})
""")
    out["ready_launch"] = _launch(machine, "readiness")
    out["ready_running"] = _running(machine, dead, 120)
    out["ready_owed"] = _owed(machine)
    _clear(machine)


@pytest.fixture(scope="module")
def journey(tmp_path_factory):
    out: dict = {}
    with FakeLLMServer() as srv:
        machine = new_machine(tmp_path_factory.mktemp("pc"), srv.base_url, label="pc", system_git=True)
        out["machine"] = machine
        try:
            install = machine.install()
            assert install.returncode == 0, fail_with(machine, f"install.ps1 exited {install.returncode}", install)
            with machine.gateway_phase():
                for cell in (_claim_race, _publish_crash, _draining, _readiness):
                    started = time.monotonic()
                    try:
                        cell(machine, out)
                    except Exception as exc:  # one broken cell must not hide the others' verdicts
                        out[f"{cell.__name__}_error"] = f"{type(exc).__name__}: {exc}"
                        _clear(machine)
                    machine.timings.append((cell.__name__, round(time.monotonic() - started, 1)))
            yield out
        finally:
            machine.teardown()


def _no_error(journey, cell: str) -> None:
    error = journey.get(f"_{cell}_error")
    assert error is None, fail_with(journey["machine"], f"harness: cell {cell} broke: {error}")


def _restarts(text: str) -> int:
    return text.count(_RESTARTING)


def test_a_claim_in_transfer_is_restarted_exactly_once(journey) -> None:
    _no_error(journey, "claim_race")
    m, second = journey["machine"], journey["race_second"]
    assert journey["race_parked"] == "paused", fail_with(m, f"premise: the first launch never reached the boundary: {journey['race_first']}")
    restarts = _restarts(journey["race_first"]) + _restarts(second.stdout)
    assert restarts == 1, fail_with(
        m, f"a set claimed while a second launch ran was restarted {restarts} times\n--- first ---\n"
           f"{journey['race_first']}", second)
    assert journey["race_running"], fail_with(m, "the paused gateway did not come back", second)
    assert journey["race_owed"] == [], fail_with(m, f"a restarted set is still on disk: {journey['race_owed']}", second)


def test_an_update_killed_after_publishing_restarts_the_set_once(journey) -> None:
    _no_error(journey, "publish_crash")
    m, run = journey["machine"], journey["publish_launch"]
    rc, text = journey["publish_kill"]
    assert rc == 71 and len(journey["publish_files"]) == 2, fail_with(
        m, f"premise: the update was not killed between publish and retire (rc={rc}, files={journey['publish_files']})\n{text}")
    assert _restarts(run.stdout) == 1, fail_with(
        m, f"one paused set was restarted {_restarts(run.stdout)} times from two copies", run)
    assert journey["publish_running"], fail_with(m, "the paused gateway did not come back", run)
    assert _restarts(journey["publish_again"].stdout) == 0 and journey["publish_owed"] == [], fail_with(
        m, f"a restarted set was owed again: {journey['publish_owed']}", journey["publish_again"])


def test_a_draining_gateway_keeps_its_restart_debt_until_it_exits(journey) -> None:
    _no_error(journey, "draining")
    m, during, after = journey["machine"], journey["drain_launch_while"], journey["drain_launch_after"]
    assert journey["drain_asked"] == "asked" and journey["drain_live_before"], fail_with(
        m, f"premise: no live gateway with a recorded stop request ({journey['drain_asked']!r})")
    owed = [sorted(t.get("profiles") or {}) for t in journey["drain_owed_while"]]
    assert owed == [["default"]], fail_with(
        m, f"a gateway asked to stop and still running lost its restart debt (owed while draining: {owed})", during)
    assert journey["drain_running_after"], fail_with(
        m, "after the draining gateway exited, the next launch did not start it again", after)
    assert journey["drain_owed_after"] == [], fail_with(m, f"debt left after the restart: {journey['drain_owed_after']}", after)


def test_each_runtime_is_retired_only_on_its_own_readiness(journey) -> None:
    _no_error(journey, "readiness")
    m, run = journey["machine"], journey["ready_launch"]
    assert journey["ready_running"], fail_with(
        m, "a gateway that could come back stayed stopped behind a failed service/profile", run)
    owed = journey["ready_owed"]
    assert len(owed) == 1, fail_with(m, f"the unready runtimes are not owed any more: {owed}", run)
    assert sorted(owed[0].get("profiles") or {}) == ["ghost"], fail_with(
        m, f"retired on another target's readiness (owed profiles {owed[0].get('profiles')})", run)
    assert owed[0].get("services") == [_MISSING_SERVICE], fail_with(
        m, f"the service that did not start is not owed (services {owed[0].get('services')})", run)
