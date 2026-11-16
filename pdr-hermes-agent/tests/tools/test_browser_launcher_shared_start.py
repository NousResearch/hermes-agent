"""Starting a persistent (shared) profile's browser is one step under the profile's cross-process lock.

Two Hermes processes starting one profile at once must end with ONE Chromium both drive: the later one waits,
then attaches (a second launch would wipe the first's ``DevToolsActivePort`` and lose Chromium's singleton). An
attach probes and leases under the same lock, so the owner's stop cannot fall between the live probe and the
lease and hand out a dead endpoint (``tools/browser_launcher.py``). Runs the real launcher against a fake Chromium
that takes the profile singleton and serves ``/json/version`` like the real one.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.tools.browser_launcher_fakes import kill_fake, pids, write_fake_chromium
from tests.waiting import wait_until
from tools import browser_launcher as launcher

pytestmark = [pytest.mark.platforms("linux"),
              pytest.mark.live_system_guard_bypass]  # the fake browsers are ours to stop if a leg failed

_REPO = Path(__file__).resolve().parents[2]

# A Hermes process on a persistent profile: prints "ready", launches once it reads "go", prints
# "<pid> <owned> <cdp_url>" (or "ERR ..."), then exits normally on the next line.
_RACER = textwrap.dedent("""\
    import os, sys
    import tools.browser_session  # registers the exit teardown
    from tools import browser_launcher as launcher
    launcher.needs_chromium_sandbox_bypass = lambda: False
    exe, udd = sys.argv[1], sys.argv[2]
    print("ready", flush=True)
    sys.stdin.readline()
    browser, err = launcher.launch_chromium("profile:" + udd, executable=exe, user_data_dir=udd, ephemeral=False,
                                            headed=False, env=dict(os.environ), timeout=30)
    print(f"{browser.pid} {browser.owned} {browser.cdp_url}" if browser else f"ERR {err!r}", flush=True)
    sys.stdin.readline()
""")


def _racer(script: Path, udd: Path, env: dict) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", _RACER, str(script), str(udd)], cwd=_REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready", proc.stderr.read()[-2000:]
    return proc


def _gone(pid: int) -> bool:
    return wait_until(lambda: not launcher._pid_alive(pid), timeout=15, what=f"pid {pid} to exit")


def test_two_processes_racing_a_first_launch_get_one_browser(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "pinned-profile"
    env = {**os.environ, "FAKE_CHROMIUM_READY_DELAY": "2"}  # both check before the first one is ready
    racers = [_racer(script, udd, env), _racer(script, udd, env)]
    try:
        for proc in racers:
            proc.stdin.write("go\n")
            proc.stdin.flush()
        results = [proc.stdout.readline().split(" ", 2) for proc in racers]
        assert not [r for r in results if r[0] == "ERR"], results
        assert len({r[0] for r in results}) == 1 and len({r[2] for r in results}) == 1, results
        assert sorted(r[1] for r in results) == ["False", "True"]
        assert pids(udd, "spawns") == [int(results[0][0])], "a second Chromium was started on the same profile"

        for proc in racers:
            proc.communicate("\n", timeout=60)
        assert _gone(int(results[0][0]))
    finally:
        for proc in racers:
            proc.kill()
        for pid in pids(udd, "spawns"):
            kill_fake(pid, script)


def test_an_owners_stop_cannot_fall_between_the_attach_probe_and_the_lease(tmp_path, monkeypatch):
    """The owner exits (its stop runs) right after this process found its browser live: the stop waits for the
    attach's lease and hands the browser over, so the endpoint this process got stays up."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "pinned-profile"
    owner = _racer(script, udd, dict(os.environ))
    owner.stdin.write("go\n")
    owner.stdin.flush()
    pid = int(owner.stdout.readline().split()[0])
    real_probe, fired = launcher._devtools_endpoint_live, []

    def probe_then_owner_exits(port, path):
        live = real_probe(port, path)
        if live and not fired:
            fired.append(True)
            owner.stdin.write("\n")  # the owner's normal exit runs its stop now
            owner.stdin.flush()
            try:
                owner.wait(timeout=3)  # a stop not serialized with this attach finishes within this window
            except subprocess.TimeoutExpired:
                pass
        return live

    monkeypatch.setattr(launcher, "_devtools_endpoint_live", probe_then_owner_exits)
    key = f"profile:{udd}"
    try:
        attached, err = launcher.launch_chromium(key, executable=str(script), user_data_dir=str(udd),
                                                 ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
        assert fired and err is None and attached.pid == pid and not attached.owned
        owner.wait(timeout=30)
        assert launcher._pid_alive(pid), "the owner stopped the browser this process had just attached to"

        launcher.stop_browser(key)  # now the last holder
        assert _gone(pid)
    finally:
        owner.kill()
        kill_fake(pid, script)


def test_the_reaper_skips_a_profile_whose_first_launch_is_still_starting(tmp_path):
    """A first launch holds the profile's lock for its whole readiness wait (up to 120 s): the janitor's reap
    passes over that profile instead of stalling behind it, and leaves the starting browser alone."""
    import threading
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "pinned-profile"
    starter = _racer(script, udd, {**os.environ, "FAKE_CHROMIUM_READY_DELAY": "600"})
    reap = threading.Thread(target=launcher.reap_orphaned_browsers, daemon=True)
    try:
        starter.stdin.write("go\n")
        starter.stdin.flush()
        [pid] = wait_until(lambda: pids(udd, "launches"), timeout=30, what="the first launch's Chromium to start")
        reap.start()
        reap.join(timeout=20)
        assert not reap.is_alive(), "the reaper is blocked behind a launch still in its readiness wait"
        assert launcher._pid_alive(pid)
    finally:
        starter.kill()
        starter.wait(timeout=10)
        for spawned in pids(udd, "spawns"):
            kill_fake(spawned, script)
        reap.join(timeout=30)
