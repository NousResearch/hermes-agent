"""One persistent (shared) profile has ONE Chromium, whichever Hermes home or launcher key reaches it.

Two homes (profiles) pinning the same ``browser.user_data_dir``, or one process reaching one dir under two keys
(a symlinked path), drive one browser: they share its launch record, leases and lock, so the LAST holder stops
it and a crashed owner's browser is reaped from any home (``tools/browser_launcher.py``,
``tools/browser_launcher_holders.py``). Runs the real launcher against a fake Chromium that takes the profile
singleton and serves ``/json/version`` like the real one.
"""

from __future__ import annotations

import os
import stat
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

# A Hermes process on a persistent profile: launches or attaches, prints "<pid> <owned>", waits for a line on
# stdin, then exits normally (the session layer's exit teardown lets go of its browsers).
_PROFILE_USER = textwrap.dedent("""\
    import os, sys
    import tools.browser_session  # registers the exit teardown
    from tools import browser_launcher as launcher
    launcher.needs_chromium_sandbox_bypass = lambda: False
    exe, udd = sys.argv[1], sys.argv[2]
    browser, err = launcher.launch_chromium("profile:" + udd, executable=exe, user_data_dir=udd, ephemeral=False,
                                            headed=False, env=dict(os.environ), timeout=30)
    print(f"{browser.pid} {browser.owned}" if browser else f"ERR {err}", flush=True)
    sys.stdin.readline()
""")


def _user(home: Path, script: Path, udd: Path) -> tuple[subprocess.Popen, int, bool]:
    home.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", _PROFILE_USER, str(script), str(udd)], cwd=_REPO,
                            env={**os.environ, "HERMES_HOME": str(home)}, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline().split()
    assert line and line[0].isdigit(), proc.stderr.read()[-2000:] if proc.poll() is not None else line
    return proc, int(line[0]), line[1] == "True"


def _gone(pid: int) -> bool:
    return wait_until(lambda: not launcher._pid_alive(pid), timeout=15, what=f"pid {pid} to exit")


def _launcher_state(*homes: Path) -> list[str]:
    """Every launch record and lease left anywhere: the homes' records dirs and the shared-profile root."""
    roots = [h / "cache" / "browser" / "chromium-records" for h in homes] + [launcher._shared_profiles_root()]
    return sorted(p.name for root in roots if root.is_dir() for p in root.rglob("*")
                  if p.suffix in (".json", ".lease"))


def test_two_homes_pinning_one_profile_share_its_browser_until_the_last_one_leaves(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd, h1, h2 = tmp_path / "pinned-profile", tmp_path / "home-1", tmp_path / "home-2"
    a, pid, a_owned = _user(h1, script, udd)
    b = None
    try:
        b, b_pid, b_owned = _user(h2, script, udd)
        assert (a_owned, b_owned, b_pid) == (True, False, pid)

        a.communicate("\n", timeout=60)  # the launcher (home 1) exits normally while home 2 still drives it
        assert launcher._pid_alive(pid), "the other home's launcher stopped a browser this home still holds"

        b.communicate("\n", timeout=60)  # the last holder exits normally
        assert _gone(pid)
        assert _launcher_state(h1, h2) == []
        assert pids(udd, "launches") == [pid]
    finally:
        for proc in (a, b):
            if proc is not None:
                proc.kill()
        kill_fake(pid, script)


def test_shared_profile_state_lives_in_a_private_per_user_dir_outside_every_home(tmp_path):
    from hermes_constants import get_hermes_home
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "pinned-profile"
    browser, err = launcher.launch_chromium(f"profile:{udd}", executable=str(script), user_data_dir=str(udd),
                                            ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
    try:
        assert err is None
        record = launcher._record_path(browser)
        root = launcher._shared_profiles_root()
        assert record.exists() and record.parent.parent == root
        assert Path(get_hermes_home()) not in record.parents
        for state in (root, record.parent):
            assert stat.S_IMODE(state.lstat().st_mode) == 0o700 and not state.is_symlink()
    finally:
        launcher.stop_browser(f"profile:{udd}")
        if browser is not None:
            kill_fake(browser.pid, script)


def test_a_crashed_launchers_shared_browser_is_reaped_from_another_home(tmp_path):
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "pinned-profile"
    owner, pid, _ = _user(tmp_path / "home-1", script, udd)
    try:
        owner.kill()
        owner.wait(timeout=10)
        assert launcher._pid_alive(pid)

        assert launcher.reap_orphaned_browsers() == 1  # this process serves only its own (other) home
        assert _gone(pid)
    finally:
        owner.kill()
        kill_fake(pid, script)


def test_one_process_holding_a_profile_under_two_keys_keeps_it_until_both_let_go(tmp_path):
    """One dir reached by two paths (a symlink) gives two launcher keys on one Chromium: dropping the key that
    launched it must not stop it, nor drop this process's lease, while the other key still drives it."""
    script = write_fake_chromium(tmp_path / "chromium")
    udd = tmp_path / "pinned-profile"
    alias = tmp_path / "alias"
    alias.symlink_to(udd)
    kw = dict(executable=str(script), ephemeral=False, headed=False, env=dict(os.environ), timeout=30)
    first, err = launcher.launch_chromium(f"profile:{udd}", user_data_dir=str(udd), **kw)
    assert err is None and first.owned
    try:
        second, err = launcher.launch_chromium(f"profile:{alias}", user_data_dir=str(alias), **kw)
        assert err is None and second.pid == first.pid and not second.owned

        launcher.stop_browser(f"profile:{udd}")
        assert launcher._pid_alive(first.pid), "dropping one key stopped the browser the other key still drives"
        assert launcher.get_browser(f"profile:{alias}") is second

        launcher.stop_browser(f"profile:{alias}")
        assert _gone(first.pid)
        assert _launcher_state() == []
    finally:
        kill_fake(first.pid, script)
