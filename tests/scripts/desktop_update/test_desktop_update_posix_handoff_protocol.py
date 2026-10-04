"""posix.sh / marker.sh hand-off protocol 2 and the A7 marker lock, against real processes.

One or two invariant tests per Round 5 finding:
  R3  every marker mutation is decided under the sidecar kernel lock (flock(1) and perl);
  R4  an OLD packaged Desktop's bridge (its launcher's pid, v1) is adopted by lineage only, by
      the one launcher rule bash and PowerShell share (lineage_rule_cases.py);
  R5  with --handoff-run only the Desktop's live bridge carrying that run is adopted;
  R6  the marker outlives a survivor still holding the checkout lock; the Desktop's reclaim
      helper answers `held` for it;
  plus the pre-publication kill cell, the process-group probe kill and the line-2 refresh
  (also through the R6 release wait), and a checkout-lock probe that never makes a concurrent
  acquire fail (Round 6).
"""

from __future__ import annotations

import fcntl  # windows-footgun: ok — linux-only module (platforms("linux"))
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

import pytest

from tests.scripts.desktop_update.lineage_rule_cases import ENV_CASES, RULE_CASES
from tests.scripts.desktop_update.lineage_rule_cases import FACTS as LINEAGE_FACTS
from tests.scripts.desktop_update.test_desktop_update_posix_marker import POSIX, _calls, _ct, _install

pytestmark = pytest.mark.platforms("linux")  # /proc ancestry and creation times

MARKER_SH = POSIX.with_name("marker.sh")


def _env(tmp_path: Path, home: Path, **extra: str) -> dict:
    env = {**os.environ, "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "HERMES_HOME": str(home),
           "HANDOFF_CAPTURE": str(tmp_path / "calls.txt"), "HERMES_RUNTIME_DIR": str(tmp_path / "store"),
           "HERMES_UPDATE_SHIM_GRACE_SECONDS": "0"}
    for key in ("PYTHONPATH", "PYTHONHOME", "HERMES_UPDATE_STARTED_AT", "MARKER_LOCK_TOOL"):
        env.pop(key, None)
    env.update(extra)
    return env


def _helper(tmp_path: Path, home: Path, install: Path, op: str, *args: str) -> str:
    out = subprocess.run(["bash", str(POSIX), "--marker-op", op, "--install-root", str(install), *args],
                         env=_env(tmp_path, home), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.fixture
def procs():
    started: list[subprocess.Popen] = []
    yield started
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# ── R3 / A7 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", ["flock", "perl"])
def test_dead_marker_is_reclaimed_by_exactly_one_of_two_live_claimants(tmp_path, procs, tool):
    """Claimant A judges the dead marker and is slow to delete it (its real `rm` sleeps first);
    claimant B arrives meanwhile. Without the lock both reclaim and both claim (R3)."""
    home = tmp_path / "home"; home.mkdir()
    marker = home / ".hermes-update-in-progress"
    marker.write_text(f"99999999\n{int(time.time())}\nct:1.000\n", encoding="utf-8")
    prelude = (f"log() {{ :; }}; MARKER={shlex.quote(str(marker))} INSTALL_ROOT={shlex.quote(str(home))} "
               f"DESKTOP_PID=0 HANDOFF_RUN='' STARTED_AT=$(date +%s) MARKER_CLAIMED=0; . {shlex.quote(str(MARKER_SH))}; ")
    slow = 'rm() { touch "$MARKER.deciding"; sleep 1.5; command rm "$@"; }; '
    report = 'marker_claim; echo "rc=$? claimed=$MARKER_CLAIMED pid=$$"; exec >&-; sleep 6'  # A stays alive while B judges
    env = {**os.environ, "MARKER_LOCK_TOOL": tool}
    a = subprocess.Popen(["bash", "-c", prelude + slow + report], env=env, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    procs.append(a)
    deadline = time.monotonic() + 10
    while not Path(str(marker) + ".deciding").exists():
        assert time.monotonic() < deadline and a.poll() is None
        time.sleep(0.01)
    b = subprocess.Popen(["bash", "-c", prelude + report], env=env, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    procs.append(b)
    out_a, out_b = a.communicate(timeout=30)[0], b.communicate(timeout=30)[0]

    claimed = [o for o in (out_a, out_b) if "claimed=1" in o]
    assert len(claimed) == 1, (out_a, out_b)
    assert "claimed=1" in out_a and "rc=1 claimed=0" in out_b
    assert marker.read_text(encoding="utf-8-sig").splitlines()[0] == str(a.pid)
    assert Path(str(marker) + ".lock").exists()


# ── R4: an older packaged Desktop ───────────────────────────────────────────


def _old_desktop_handoff(tmp_path: Path, home: Path, install: Path, procs, *, bridge: str) -> tuple[str | None, Path]:
    """A stand-in Desktop (P) spawns the launcher (X) exactly like be3fd671d70 checkout.ts: X then
    starts the daemon as its child, and P (bridge="launcher") writes `X\\n<startedAt>\\n` over the
    marker right after the spawn. bridge="unrelated" leaves a v1 claim of some other live process."""
    started = int(time.time())
    marker = home / ".hermes-update-in-progress"
    report = tmp_path / "launched.txt"
    launcher = (f'echo $$ > {shlex.quote(str(report))}; sleep 0.5; '
                f'bash {shlex.quote(str(POSIX))} --daemonized --no-ui --self-test-marker --install-root {shlex.quote(str(install))} '
                f'--desktop-pid $PPID; echo "rc=$?" >> {shlex.quote(str(report))}; sleep 2')
    desktop_code = (
        "import subprocess, sys, time, os\n"
        f"x = subprocess.Popen(['bash', '-c', {launcher!r}], env=dict(os.environ, HERMES_UPDATE_STARTED_AT='{started}'))\n"
        f"bridge = {bridge!r}\n"
        f"if bridge == 'launcher': open({str(marker)!r}, 'w', encoding='utf-8').write(f'{{x.pid}}\\n{started}\\n')\n"  # windows-footgun: ok — a write, inside generated code
        "x.wait()\n"
    )
    body = None
    if bridge == "unrelated":
        other = subprocess.Popen(["sleep", "60"]); procs.append(other)
        body = f"{other.pid}\n{started}\n"
        marker.write_text(body, encoding="utf-8")
    desktop = subprocess.Popen(["python3", "-c", desktop_code], env=_env(tmp_path, home)); procs.append(desktop)
    desktop.wait(timeout=60)
    return body, report


def test_old_desktop_bridge_naming_its_launcher_is_adopted_by_lineage(tmp_path, procs):
    home, install = _install(tmp_path)
    _desktop, report = _old_desktop_handoff(tmp_path, home, install, procs, bridge="launcher")

    lines = report.read_text(encoding="utf-8-sig").split()
    assert lines[-1] == "rc=0", (home / "logs" / "desktop-update-handoff.log").read_text(encoding="utf-8-sig")
    log = (home / "logs" / "desktop-update-handoff.log").read_text(encoding="utf-8-sig")
    assert re.search(rf"adopted the Desktop's update marker \(desktop pid \d+, bridge pid {lines[0]} -> \d+\)", log)
    body = (home / ".hermes-update-in-progress").read_text(encoding="utf-8-sig").splitlines()
    assert body[0] != lines[0] and body[2].startswith("ct:")  # the daemon owns it now, v2


def test_old_desktop_handoff_never_adopts_an_unrelated_live_claim(tmp_path, procs):
    home, install = _install(tmp_path)
    before, report = _old_desktop_handoff(tmp_path, home, install, procs, bridge="unrelated")

    assert report.read_text(encoding="utf-8-sig").split()[-1] == "rc=2"
    assert (home / ".hermes-update-in-progress").read_text(encoding="utf-8-sig") == before


def test_launcher_lineage_rule_matches_the_shared_table(tmp_path):
    """One lineage rule for bash and PowerShell (round 5 D11): the same table runs in both."""
    calls = [f"log() {{ :; }}; MARKER=/dev/null INSTALL_ROOT=/dev/null; . {shlex.quote(str(MARKER_SH))}"]
    for case in RULE_CASES:
        args = " ".join("1" if case["facts"][name] else "0" for name in LINEAGE_FACTS)
        calls.append(f"if marker_launcher_rule {args}; then echo {case['id']}=1; else echo {case['id']}=0; fi")
    for case in ENV_CASES:
        calls.append(f"if HERMES_UPDATE_STARTED_AT={shlex.quote(case['env'])} marker_env_started_matches {shlex.quote(case['line2'])}; "
                     f"then echo env_{case['id']}=1; else echo env_{case['id']}=0; fi")
    out = subprocess.run(["bash", "-c", "\n".join(calls)], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True).stdout
    got = dict(line.split("=") for line in out.split())
    want = {c["id"]: "1" if c["expect"] else "0" for c in RULE_CASES} | {f"env_{c['id']}": "1" if c["expect"] else "0" for c in ENV_CASES}
    assert got == want


@pytest.mark.parametrize("env_matches", [True, False], ids=["env-matches", "env-differs"])
def test_reparented_live_launcher_is_adopted_only_with_the_handoff_started_at(tmp_path, procs, env_matches):
    """The table's divergent row on real processes: the launcher X is our live parent, but no
    longer the Desktop's child (the Desktop quit first); line 2 == HERMES_UPDATE_STARTED_AT."""
    home, install = _install(tmp_path)
    desktop = subprocess.Popen(["sleep", "60"]); procs.append(desktop)  # not X's parent
    started = int(time.time())
    marker = home / ".hermes-update-in-progress"
    report = tmp_path / "launched.txt"
    launcher = (f'printf "%s\\n{started}\\n" $$ > {shlex.quote(str(marker))}; '
                f'HERMES_UPDATE_STARTED_AT={started if env_matches else started - 7} bash {shlex.quote(str(POSIX))} --daemonized --no-ui '
                f'--self-test-marker --install-root {shlex.quote(str(install))} --desktop-pid {desktop.pid}; echo "rc=$?" > {shlex.quote(str(report))}')
    x = subprocess.run(["bash", "-c", launcher], env=_env(tmp_path, home), timeout=60)
    assert x.returncode == 0
    rc = report.read_text(encoding="utf-8-sig").strip()
    body = marker.read_text(encoding="utf-8-sig").splitlines()
    if env_matches:
        assert rc == "rc=0", (home / "logs" / "desktop-update-handoff.log").read_text(encoding="utf-8-sig")
        assert body[1] == str(started) and body[2].startswith("ct:")
    else:
        assert rc == "rc=2" and len(body) == 2 and body[1] == str(started)


# ── R5: protocol 2 adopts only the Desktop's bridge carrying its run ────────


@pytest.mark.parametrize("bridge_run, arg_run, adopted", [
    ("desk-1-a-0001", "desk-1-a-0001", True),
    ("desk-1-a-0001", "desk-1-b-0002", False),
    (None, "desk-1-a-0001", False),
], ids=["same-run", "other-run", "no-run"])
def test_handoff_run_adopts_only_the_matching_live_bridge(tmp_path, procs, bridge_run, arg_run, adopted):
    home, install = _install(tmp_path)
    desktop = subprocess.Popen(["sleep", "60"]); procs.append(desktop)
    marker = home / ".hermes-update-in-progress"
    body = f"{desktop.pid}\n{int(time.time()) - 5}\nct:{_ct(desktop.pid)}\n" + (f"run:{bridge_run}\n" if bridge_run else "")
    marker.write_text(body, encoding="utf-8")

    result = subprocess.run(["bash", str(POSIX), "--daemonized", "--no-ui", "--self-test-marker", "--install-root", str(install),
                             "--desktop-pid", str(desktop.pid), "--handoff-run", arg_run],
                            env=_env(tmp_path, home), capture_output=True, text=True, timeout=60)

    if adopted:
        assert result.returncode == 0, result.stdout + result.stderr
        pid, started, ct, run = marker.read_text(encoding="utf-8-sig").splitlines()
        assert pid != str(desktop.pid) and started == body.split("\n")[1] and ct.startswith("ct:")
        assert run == f"run:{arg_run}"
    else:
        assert result.returncode == 2, result.stdout + result.stderr
        assert marker.read_text(encoding="utf-8-sig") == body
    assert _calls(tmp_path) == []


def test_withdraw_removes_only_this_desktops_bridge_and_reports_a_taker(tmp_path, procs):
    home, install = _install(tmp_path)
    desktop, script = subprocess.Popen(["sleep", "60"]), subprocess.Popen(["sleep", "60"])
    procs.extend([desktop, script])
    marker = home / ".hermes-update-in-progress"
    run = "desk-9-x-0001"
    taken = f"{script.pid}\n{int(time.time())}\nct:{_ct(script.pid)}\nrun:{run}\n"
    marker.write_text(taken, encoding="utf-8")
    assert _helper(tmp_path, home, install, "withdraw", "--desktop-pid", str(desktop.pid), "--handoff-run", run) == f"taken {script.pid}"
    assert _helper(tmp_path, home, install, "withdraw", "--desktop-pid", str(desktop.pid), "--handoff-run", "desk-9-y-0002") == "foreign"
    assert marker.read_text(encoding="utf-8-sig") == taken
    marker.write_text(f"{desktop.pid}\n{int(time.time())}\nct:{_ct(desktop.pid)}\nrun:{run}\n", encoding="utf-8")
    assert _helper(tmp_path, home, install, "withdraw", "--desktop-pid", str(desktop.pid), "--handoff-run", run) == "withdrawn"
    assert not marker.exists()


# ── R6: a survivor holding the checkout lock keeps the marker ───────────────


def test_marker_outlives_a_survivor_that_still_holds_the_checkout_lock(tmp_path):
    home, install = _install(tmp_path, legacy=True)
    marker = home / ".hermes-update-in-progress"
    completion = tmp_path / "release-completion"
    env = _env(tmp_path, home, HANDOFF_COMPLETION=str(completion), HANDOFF_CHECKOUT_LOCK=str(install / ".hermes-update.lock"))
    script = subprocess.Popen(["bash", str(POSIX), "--daemonized", "--no-ui", "--install-root", str(install)], env=env, cwd=tmp_path,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    log = home / "logs" / "desktop-update-handoff.log"
    try:
        deadline = time.monotonic() + 60
        while "keeping the update marker" not in (log.read_text(encoding="utf-8-sig") if log.exists() else ""):
            assert time.monotonic() < deadline and script.poll() is None, log.read_text(encoding="utf-8-sig")
            time.sleep(0.05)
        time.sleep(1.5)
        assert script.poll() is None and marker.read_text(encoding="utf-8-sig").splitlines()[0] == str(script.pid)
        assert _helper(tmp_path, home, install, "reclaim") == f"live {script.pid}"
        os.killpg(script.pid, signal.SIGKILL); script.wait()  # windows-footgun: ok — linux-only test  # the script dies too: dead marker, lock still held
        assert _helper(tmp_path, home, install, "reclaim") == "held"
        assert marker.exists()
    finally:
        completion.touch()
        if script.poll() is None:
            os.killpg(script.pid, signal.SIGKILL); script.wait()  # windows-footgun: ok — linux-only test
    deadline = time.monotonic() + 10
    while _helper(tmp_path, home, install, "reclaim") == "held":
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert not marker.exists()


def test_release_waits_for_the_survivor_then_removes_the_marker(tmp_path):
    home, install = _install(tmp_path, legacy=True)
    completion = tmp_path / "release-completion"
    env = _env(tmp_path, home, HANDOFF_COMPLETION=str(completion), HANDOFF_CHECKOUT_LOCK=str(install / ".hermes-update.lock"))
    script = subprocess.Popen(["bash", str(POSIX), "--daemonized", "--no-ui", "--install-root", str(install)], env=env, cwd=tmp_path,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = home / "logs" / "desktop-update-handoff.log"
    deadline = time.monotonic() + 60
    while "keeping the update marker" not in (log.read_text(encoding="utf-8-sig") if log.exists() else ""):
        assert time.monotonic() < deadline and script.poll() is None
        time.sleep(0.05)
    time.sleep(1.0)
    assert script.poll() is None and (home / ".hermes-update-in-progress").exists()
    completion.touch()
    assert script.wait(timeout=30) == 0
    assert not (home / ".hermes-update-in-progress").exists()


def test_line_two_stays_young_through_the_r6_release_wait(tmp_path):
    """An old packaged Desktop deletes a marker whose line 2 is 20 minutes old, live owner or
    not. The release wait can last hours, so the refresher must outlive `hermes update`."""
    home, install = _install(tmp_path, legacy=True)
    marker = home / ".hermes-update-in-progress"
    completion = tmp_path / "release-completion"
    env = _env(tmp_path, home, HANDOFF_COMPLETION=str(completion), HANDOFF_CHECKOUT_LOCK=str(install / ".hermes-update.lock"),
               HERMES_UPDATE_STARTED_AT=str(int(time.time()) - 600))
    script = subprocess.Popen(["bash", str(POSIX), "--daemonized", "--no-ui", "--install-root", str(install),
                               "--self-test-refresh-every", "1"], env=env, cwd=tmp_path,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    log = home / "logs" / "desktop-update-handoff.log"
    try:
        deadline = time.monotonic() + 60
        while "keeping the update marker" not in (log.read_text(encoding="utf-8-sig") if log.exists() else ""):
            assert time.monotonic() < deadline and script.poll() is None, log.read_text(encoding="utf-8-sig")
            time.sleep(0.05)
        for _ in range(4):  # several refresh intervals, all inside the wait
            time.sleep(1.5)
            pid, started = marker.read_text(encoding="utf-8-sig").splitlines()[:2]
            assert script.poll() is None and pid == str(script.pid)
            assert time.time() - int(started) <= 3.5, started
    finally:
        completion.touch()
        if script.poll() is None:
            assert script.wait(timeout=30) == 0
    assert not marker.exists()


def test_checkout_lock_probe_never_makes_a_concurrent_acquire_fail(tmp_path):
    """checkout_lock_held takes the REAL lock; a `hermes update` acquiring it non-blocking in a
    tight loop meanwhile must (practically) never see it busy. A probe that holds it across a
    process exit and a shell close refused ~7% of such attempts."""
    lock = tmp_path / ".hermes-update.lock"
    lock.touch()
    loop = (f"log() {{ :; }}; MARKER=/dev/null INSTALL_ROOT={shlex.quote(str(tmp_path))}; . {shlex.quote(str(MARKER_SH))}; "
            "end=$((SECONDS + 4)); n=0; while [ $SECONDS -lt $end ]; do checkout_lock_held && echo held; n=$((n + 1)); done; echo \"probes=$n\"")
    env = {k: v for k, v in os.environ.items() if k != "MARKER_LOCK_TOOL"}
    prober = subprocess.Popen(["bash", "-c", loop], env=env, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    attempts = refused = 0
    while prober.poll() is None:
        fd = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            refused += 1
        finally:
            os.close(fd)
        attempts += 1
        time.sleep(0.002)
    out = prober.communicate(timeout=10)[0]
    assert attempts > 300 and "probes=" in out, (attempts, out)
    assert refused * 100 <= attempts, f"{refused} of {attempts} acquires refused by the probe alone"


# ── pre-publication kill cell ───────────────────────────────────────────────


def test_script_killed_before_the_delegate_line_appears_runs_no_update(tmp_path):
    """The update child exists but the marker does not name it yet (the script waits for the A7
    lock, which this test holds). SIGKILL the script there: the child must never run the update,
    and the marker must read dead (reclaimable), never `dead while an update runs`."""
    home, install = _install(tmp_path)
    marker = home / ".hermes-update-in-progress"
    script = subprocess.Popen(["bash", str(POSIX), "--daemonized", "--no-ui", "--install-root", str(install)],
                              env=_env(tmp_path, home), cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lock_fd = None
    try:
        deadline = time.monotonic() + 30
        while not (marker.exists() and marker.read_text(encoding="utf-8-sig").split("\n")[0] == str(script.pid)):
            assert time.monotonic() < deadline and script.poll() is None
            time.sleep(0.005)
        lock_fd = os.open(str(marker) + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        log = home / "logs" / "desktop-update-handoff.log"
        while "running:" not in log.read_text(encoding="utf-8-sig"):
            assert time.monotonic() < deadline and script.poll() is None
            time.sleep(0.01)
        time.sleep(0.5)  # the update child is spawned and parked behind the go-file
        children = [int(p) for p in Path(f"/proc/{script.pid}/task/{script.pid}/children").read_text(encoding="utf-8").split()]
        # The (sub)shell that will exec `hermes update`; the other child is flock(1) waiting on us.
        gate = [p for p in children if Path(f"/proc/{p}/comm").read_text(encoding="utf-8").strip() == "bash"]
        assert gate, "the update child should exist before the delegate line is published"
        script.kill(); script.wait()
        time.sleep(1.0)
        assert _calls(tmp_path) == [] or not any(c.startswith("update") for c in _calls(tmp_path))
        assert all(not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split(") ")[1][:1]
                   for pid in gate)
        assert "delegate:" not in marker.read_text(encoding="utf-8-sig")
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if script.poll() is None:
            script.kill(); script.wait()
    assert _helper(tmp_path, home, install, "reclaim") == "reclaimed"


# ── bounded probes and the old-reader line-2 refresh ────────────────────────


def test_timed_out_probe_is_killed_with_its_whole_process_tree(tmp_path):
    body = POSIX.read_text(encoding="utf-8-sig")
    found = re.search(r"^run_bounded\(\) \{.*?^\}\n", body, re.S | re.M)
    assert found
    run_bounded = found.group(0)
    pidfile = tmp_path / "grandchild.pid"
    code = (f"log() {{ :; }}\n{run_bounded}\n"
            f"run_bounded 1 bash -c 'sleep 60 & echo $! > {shlex.quote(str(pidfile))}; wait'; echo \"rc=$?\"")
    out = subprocess.run(["bash", "-c", code], env={**os.environ, "TMPDIR": str(tmp_path)}, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert "rc=124" in out.stdout, out
    grandchild = int(pidfile.read_text(encoding="utf-8-sig"))
    time.sleep(0.2)
    assert not Path(f"/proc/{grandchild}").exists() or Path(f"/proc/{grandchild}/stat").read_text(encoding="utf-8").split(") ")[1].startswith("Z")


def test_line_two_is_refreshed_only_while_the_claim_is_still_ours(tmp_path, procs):
    home = tmp_path / "home"; home.mkdir()
    marker = home / ".hermes-update-in-progress"
    run = (f"log() {{ :; }}; MARKER={shlex.quote(str(marker))} INSTALL_ROOT={shlex.quote(str(home))} DESKTOP_PID=0 HANDOFF_RUN=''; "
           f". {shlex.quote(str(MARKER_SH))}; marker_now() {{ echo 1999999999; }}; ")
    ours = (run + 'MY_CT="$(proc_ct $$)"; printf "%s\\n100\\nct:%s\\ndelegate:4242 ct:5.000\\nrun:desk-1-a-0001\\n" $$ "$MY_CT" > "$MARKER"; '
            'marker_locked marker_refresh_locked; cat "$MARKER"')
    out = subprocess.run(["bash", "-c", ours], capture_output=True, text=True, encoding="utf-8", timeout=30).stdout.splitlines()
    assert out[1] == "1999999999" and out[3] == "run:desk-1-a-0001" and len(out) == 4  # dead delegate dropped
    other = subprocess.Popen(["sleep", "60"]); procs.append(other)
    foreign = f"{other.pid}\n100\nct:{_ct(other.pid)}\n"
    marker.write_text(foreign, encoding="utf-8")
    subprocess.run(["bash", "-c", run + "marker_locked marker_refresh_locked"], check=True, timeout=30)
    assert marker.read_text(encoding="utf-8-sig") == foreign
