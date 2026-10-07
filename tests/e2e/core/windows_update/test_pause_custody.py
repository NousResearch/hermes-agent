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
* never asked: the REAL pause dies after recording its intent to stop the gateway, before asking it.
  The gateway keeps serving and owes nothing; when the user stops it later, no launch restarts it.
* asked: the REAL pause dies right after issuing the request (planned-stop marker on disk), before
  the socket call. The gateway drains and exits on its own; the next launch restarts it once.
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

# Every driver: the installed module behind the same bootstrap a launch runs (a resume imports
# hermes_cli.main, whose bootstrap would otherwise re-enter this script mid-resume), and a hook that
# stops on one line of it (kill or park).
_DRIVER = """
import inspect, json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hermes_bootstrap  # as every launch does: re-enters the managed interpreter with its dependencies
from hermes_cli import update_pause_record as r

def stop_at(fn, texts, action, flag=None):
    lines, start = inspect.getsourcelines(fn)
    line = next(start + i for text in texts for i, t in enumerate(lines) if text in t)
    def trace(frame, event, arg):  # traces only fn's own frames: the rest runs at full speed
        if frame.f_code is not fn.__code__:
            return None
        if event == "line" and frame.f_lineno == line:
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


def _driver(machine, name: str, body: str, *args: str) -> tuple[subprocess.Popen, Path]:
    """The install's python running *body*; its output goes to a transcript (as a launch's would)."""
    machine._seq += 1
    log = machine.logs / f"{machine._seq:02d}-driver-{name}.log"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with log.open("wb") as fh:
        proc = subprocess.Popen([str(_python(machine)), "-u", "-c", _DRIVER + body, str(machine.install_dir), *args],
                                cwd=machine.profile, env=machine.env(), stdin=subprocess.DEVNULL, stdout=fh,
                                stderr=subprocess.STDOUT, creationflags=flags)
    machine._spawned.append(proc)
    return proc, log


def _text(log: Path) -> str:
    try:
        return log.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def _await_line(proc: subprocess.Popen, log: Path, line: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if line in _text(log).splitlines():
            return True
        if proc.poll() is not None:
            return line in _text(log).splitlines()
        time.sleep(0.2)
    return False


def _run_driver(machine, name: str, body: str, *args: str, timeout: float = 300) -> tuple[int, str]:
    proc, log = _driver(machine, name, body, *args)
    proc.wait(timeout=timeout)
    return proc.returncode, _text(log)


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
    deadline = time.monotonic() + 60
    while machine.owned_processes() and time.monotonic() < deadline:
        time.sleep(0.5)
    for path in _records(machine):
        path.unlink(missing_ok=True)
    # A hard-killed gateway leaves its last ``running`` state behind: the next cell's gateway must
    # be the one it reads, never a dying pid from this one.
    (machine.hermes_home / "gateway_state.json").unlink(missing_ok=True)


def _running(machine, not_pid: int, timeout: float) -> int | None:
    try:
        return int(machine.wait_gateway_running(not_pid=not_pid, timeout=timeout).get("pid") or 0) or None
    except AssertionError:
        return None


def _fresh_gateway(machine) -> int:
    """A real gateway started now (``hermes gateway run``); its pid once it reports running."""
    (machine.hermes_home / "gateway_state.json").unlink(missing_ok=True)
    proc = machine.spawn_gateway()
    pid = int(machine.wait_gateway_running().get("pid") or 0)
    assert pid and psutil.pid_exists(pid) and proc.poll() is None, fail_with(machine, f"harness: gateway {pid} not up")
    return pid


def _paused_gateway(machine) -> int:
    """A real gateway, stopped the way a pause stops it (a clean ``hermes gateway stop``, not a
    hard kill whose stale lock state slows the next ``--replace``). Its (now dead) pid."""
    pid = _fresh_gateway(machine)
    machine.hermes("gateway", "stop", label="pause-stop", timeout=180)
    deadline = time.monotonic() + 60
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    _clear(machine)
    return pid


def _launch(machine, label: str):
    return machine.hermes("gateway", "status", label=label, timeout=600)


def _claim_race(machine, out: dict) -> None:
    dead = _paused_gateway(machine)
    # A claim handed back unowned (a launch that could not finish the resume): any launch may take it.
    out["race_seed"] = _run_driver(machine, "race-seed", f"""
orphan({{"default": {dead}}})
won = r.claim(r.record_path())
r._atomic_write(won[0], {{**won[1], "claimer": r.UNOWNED}})
""")
    flag = machine.root / "claim-race-go"
    first, log = _driver(machine, "race-first", f"""
import psutil
print("paused pid {dead} alive before recovery:", psutil.pid_exists({dead}), flush=True)
stop_at(r.claim, {_CLAIM_BOUNDARY}, "park", sys.argv[2])
r.recover(["status"])
sys.settrace(None)
print("left on disk:", sorted(p.name for p in r.record_path().parent.glob(r.RECORD_STEM + "*")), flush=True)
""", str(flag))
    out["race_parked"] = _await_line(first, log, "paused")
    out["race_second"] = _launch(machine, "claim-race-second")
    flag.touch()
    first.wait(timeout=600)
    out["race_first"] = _text(log)
    out["race_running"] = _running(machine, dead, 150)
    out["race_owed"] = _owed(machine)
    _clear(machine)


def _publish_crash(machine, out: dict) -> None:
    dead = _paused_gateway(machine)
    out["publish_seed"] = _run_driver(machine, "publish-seed", f'orphan({{"default": {dead}}})')
    out["publish_kill"] = _run_driver(machine, "publish-kill", """
adopted, claims = r.adopt_orphans()
stop_at(r.record_pause, ("release_claims(claims)",), "kill")
r.record_pause({"resume_needed": True, "profiles": {}}, adopted, claims)
""")
    out["publish_files"] = [p.name for p in _records(machine)]
    out["publish_launch"] = _launch(machine, "after-publish-crash")
    out["publish_running"] = _running(machine, dead, 150)
    out["publish_again"] = _launch(machine, "after-publish-crash-2")
    out["publish_owed"] = _owed(machine)
    _clear(machine)


def _draining(machine, out: dict) -> None:
    # The real gateway keeps running: it was asked to stop and has not exited yet (draining).
    live = _fresh_gateway(machine)
    updater, log = _driver(machine, "drain-updater", f"""
pid = {live}
token = {{"resume_needed": True, "profiles": {{"default": pid}}, "identities": {{str(pid): r.identity(pid)["ct"]}}}}
if hasattr(r, "mark_stop_requested"):
    token = r.record_pause(token, None, [])
    r.mark_stop_requested(token, [pid])
    if hasattr(r, "mark_stop_sent"):  # the request went out; the gateway has not exited yet
        r.mark_stop_sent(token, pid)
else:  # the old design has no stop record: write the same fact it would have needed
    r.write(r.stamp_tree({{**token, "stop_requested": [str(pid)]}}), owner=r.identity())
print("asked", flush=True)
time.sleep(600)
""")
    out["drain_asked"] = _await_line(updater, log, "asked")
    # The updater dies with the stop request on disk (its venv launcher and the interpreter under it).
    subprocess.run(["taskkill", "/PID", str(updater.pid), "/T", "/F"], capture_output=True, timeout=60)
    updater.wait(timeout=60)
    out["drain_updater"] = _text(log)
    out["drain_live_before"] = psutil.pid_exists(live)
    out["drain_launch_while"] = _launch(machine, "while-draining")
    out["drain_live_after_launch"] = psutil.pid_exists(live)
    out["drain_owed_while"] = _owed(machine)
    subprocess.run(["taskkill", "/PID", str(live), "/F"], capture_output=True, timeout=60)
    deadline = time.monotonic() + 30
    while psutil.pid_exists(live) and time.monotonic() < deadline:
        time.sleep(0.2)
    out["drain_launch_after"] = _launch(machine, "after-drained")
    out["drain_running_after"] = _running(machine, live, 150)
    out["drain_owed_after"] = _owed(machine)
    _clear(machine)


def _readiness(machine, out: dict) -> None:
    dead = _paused_gateway(machine)
    out["ready_seed"] = _run_driver(machine, "ready-seed", f"""
orphan({{"default": {dead}, "ghost": {dead}}}, services=["{_MISSING_SERVICE}"],
       expected_services=["{_MISSING_SERVICE}"], restarted_services=[],
       service_profiles={{"{_MISSING_SERVICE}": "svc"}})
""")
    out["ready_launch"] = _launch(machine, "readiness")
    out["ready_running"] = _running(machine, dead, 150)
    out["ready_owed"] = _owed(machine)
    _clear(machine)


def _gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    return not psutil.pid_exists(pid)


# The real pause, killed on one line of the installed update_cmd_windows (texts present before and
# after round 6, so the same cell runs on the red branch).
_PAUSE = """
from hermes_cli import update_cmd_windows as w
stop_at(getattr(w, sys.argv[2]), (sys.argv[3],), "kill")
w._pause_windows_gateways_for_update()
"""


def _never_asked(machine, out: dict) -> None:
    live = _fresh_gateway(machine)
    out["never_kill"] = _run_driver(machine, "never-asked-update", _PAUSE, "_pause_windows_gateways_for_update",
                                    "profiles = _stop_windows_gateways(")
    out["never_intent"] = _owed(machine)
    out["never_live_before"] = psutil.pid_exists(live)
    out["never_launch_while"] = _launch(machine, "never-asked-while-serving")
    out["never_live_after_launch"] = psutil.pid_exists(live)
    out["never_owed_while"] = _owed(machine)
    # The user's deliberate stop, after the update's obligation was retired.
    out["never_user_stop"] = machine.hermes("gateway", "stop", label="never-asked-user-stop", timeout=180)
    out["never_gone"] = _gone(live, 60)
    out["never_launch_after"] = _launch(machine, "never-asked-after-user-stop")
    out["never_running_after"] = _running(machine, live, 30)
    out["never_owed_after"] = _owed(machine)
    _clear(machine)


def _asked(machine, out: dict) -> None:
    live = _fresh_gateway(machine)
    out["asked_kill"] = _run_driver(machine, "asked-update", _PAUSE, "_request_socket_pauses",
                                    "ack = pause_gateway_for_update(Path(proc.path))")
    out["asked_owed_at_death"] = _owed(machine)
    out["asked_gone"] = _gone(live, 120)  # the gateway acts on the request on disk by itself
    out["asked_launch"] = _launch(machine, "asked-after-exit")
    out["asked_running"] = _running(machine, live, 150)
    out["asked_again"] = _launch(machine, "asked-again")
    out["asked_owed"] = _owed(machine)
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
            # The cells start and stop the gateway ~10 times in a few minutes on purpose; the respawn-storm
            # breaker (5 starts / 120 s, then a 40 s sleep before boot) would otherwise outlast a resume's
            # 30 s readiness window and make "not verified" a property of the harness, not the code.
            off = machine.hermes("config", "set", "gateway.respawn_storm.max_starts", "0", label="storm-breaker-off")
            assert off.returncode == 0, fail_with(machine, "harness: could not disable the respawn-storm breaker", off)
            with machine.gateway_phase():
                for cell in (_claim_race, _publish_crash, _draining, _readiness, _never_asked, _asked):
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
    first = f"--- first launch (driver) ---\n{journey['race_first']}"
    assert journey["race_parked"], fail_with(m, f"premise: the first launch never reached the boundary\n{first}")
    restarts = _restarts(journey["race_first"]) + _restarts(second.stdout)
    assert restarts == 1, fail_with(
        m, f"a set claimed while a second launch ran was restarted {restarts} times\n{first}", second)
    assert journey["race_running"], fail_with(m, f"the paused gateway did not come back\n{first}", second)
    assert journey["race_owed"] == [], fail_with(m, f"a restarted set is still on disk: {journey['race_owed']}\n{first}", second)


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
    assert journey["drain_asked"] and journey["drain_live_before"], fail_with(
        m, f"premise: no live gateway with a recorded stop request\n{journey['drain_updater']}")
    assert journey["drain_live_after_launch"] and _restarts(during.stdout) == 0, fail_with(
        m, "a launch restarted (replaced) a gateway that was still draining", during)
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


def test_a_gateway_the_update_never_asked_is_left_serving_and_a_user_stop_is_final(journey) -> None:
    _no_error(journey, "never_asked")
    m, during, after = journey["machine"], journey["never_launch_while"], journey["never_launch_after"]
    rc, text = journey["never_kill"]
    assert rc == 71 and len(journey["never_intent"]) == 1 and journey["never_live_before"], fail_with(
        m, f"premise: the pause did not die between its intent and its request (rc={rc}, "
           f"records={journey['never_intent']})\n{text}")
    assert journey["never_live_after_launch"] and _restarts(during.stdout) == 0, fail_with(
        m, "a launch restarted (replaced) a gateway the update never asked to stop", during)
    assert journey["never_owed_while"] == [], fail_with(
        m, f"a gateway never asked to stop is held as draining (owed while it serves: {journey['never_owed_while']})",
        during)
    assert journey["never_gone"], fail_with(m, "premise: the user's `hermes gateway stop` did not stop it",
                                            journey["never_user_stop"])
    assert _restarts(after.stdout) == 0 and journey["never_running_after"] is None, fail_with(
        m, "a gateway the user stopped was restarted for an update that never stopped it", after)
    assert journey["never_owed_after"] == [], fail_with(m, f"debt left: {journey['never_owed_after']}", after)


def test_a_gateway_the_update_asked_is_restarted_once_after_it_exits(journey) -> None:
    _no_error(journey, "asked")
    m, run = journey["machine"], journey["asked_launch"]
    rc, text = journey["asked_kill"]
    assert rc == 71 and len(journey["asked_owed_at_death"]) == 1, fail_with(
        m, f"premise: the pause did not die right after issuing its request (rc={rc}, "
           f"records={journey['asked_owed_at_death']})\n{text}")
    assert journey["asked_gone"], fail_with(m, "premise: the asked gateway never acted on the request on disk", run)
    assert _restarts(run.stdout) == 1 and journey["asked_running"], fail_with(
        m, "a gateway the update asked to stop was not started again after it exited", run)
    assert _restarts(journey["asked_again"].stdout) == 0 and journey["asked_owed"] == [], fail_with(
        m, f"a restarted gateway was owed again: {journey['asked_owed']}", journey["asked_again"])
