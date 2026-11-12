"""After the checkout moves, the running ``hermes update`` process imports nothing from the tree.

Once git (or the ZIP swap) writes the new code, every module this interpreter has not loaded yet is
the REPLACEMENT tree's: importing one mixes generations in one process, and it is also why every
name such code reaches has to stay frozen (``tests/compat/old_updater_surface.json``).

Each case runs in a fresh interpreter: it loads the update closure, arms the commit point (the last
step before the tree moves) on a sandboxed install, then installs a ``sys.meta_path`` finder that
records every module still found in the checkout, and drives one path the old process runs after
the move. Supervisors are fake executables on ``PATH``; nothing reaches the network or a real gateway.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]

_DRIVER = r'''
import importlib.machinery, json, os, sys
from pathlib import Path

REPO, ROOT, CASE, LABEL, CLONE = sys.argv[1:6]
sys.path.insert(0, REPO)
import hermes_cli.main
from hermes_cli import update_cmd, update_cmd_commit, update_cmd_posix_pause, update_receipt
import hermes_cli.update_lock  # main.py imports it before the update runs

home = os.environ["HERMES_HOME"]
GONE = 2 ** 22 + 7  # a PID no process holds


def request(token=None):
    return {"source": ROOT, "home": home, "receipt": dict(update_receipt._current.get().data),
            "windows_resume": token, "gateway_mode": False, "expected_sha": "0" * 40}


def posix_token(*entries, bare=()):
    # The verdict is set: the test never runs the real pause (it would stop this host's gateways).
    return {"platform": "posix", "resume_needed": True, "profiles": {}, "unmapped_pids": [],
            "unmapped": list(bare), "posix_units": list(entries), "posix_pause_verdict": None}


if CASE == "systemd_resume":
    os.geteuid = lambda: 0  # root: a legacy system unit is rewritten, not just named
    from hermes_cli import gateway  # loaded by the pause that stopped the unit
    gateway._SYSTEM_UNIT_DIR = Path(ROOT) / "system-units"
    gateway._SYSTEM_UNIT_DIR.mkdir()
    unit_name = gateway.get_service_name()
    (gateway._SYSTEM_UNIT_DIR / f"{unit_name}.service").write_text(
        f"[Service]\nUser={os.environ.get('USER') or 'nobody'}\nExecStart=/bin/false\n")
if CASE == "launchd_resume":
    from hermes_cli import gateway  # loaded by the pause that stopped the job
    plist = Path(ROOT) / "job.plist"
    plist.write_text("<plist/>")
    gateway.get_launchd_plist_path = lambda: plist
token = None
if CASE == "lost_completion":
    token = posix_token(bare=[{"pid": GONE, "home": str(Path(ROOT) / "bare-home"), "argv": [sys.executable, "-c", f"open({str(Path(ROOT) / 'supervisor.log')!r}, 'a').write('bare-replay\\n')"]}])
if CASE == "systemd_resume":
    token = posix_token({"kind": "systemd", "scope": "system", "unit": unit_name, "pid": GONE})
if CASE == "launchd_resume":
    token = posix_token({"kind": "launchd", "label": LABEL, "domain": "gui/0", "plist": str(plist), "pid": GONE})
if CASE == "windows_resume":
    import subprocess
    real_popen = subprocess.Popen

    class Watcher:
        pid, returncode = GONE + 4, None

        def poll(self):
            return None

    def record_watcher(args, *rest, **kwargs):
        # The restart watcher (``python -c <watcher> <pid> <argv>``) would start a real gateway.
        if isinstance(args, (list, tuple)) and len(args) > 2 and args[1] == "-c" and "start_gateway_job" in args[2]:
            with open(Path(ROOT) / "supervisor.log", "a", encoding="utf-8") as log:
                log.write("restart watcher\n")
            return Watcher()
        return real_popen(args, *rest, **kwargs)
    subprocess.Popen = record_watcher
    token = {"platform": "windows", "resume_needed": True, "profiles": {"default": GONE}, "unmapped": [],
             "services": []}
if CASE == "shallow_count":
    import urllib.error, urllib.request

    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")
    urllib.request.urlopen = offline
    hermes_cli.main.PROJECT_ROOT = Path(CLONE)

# The run's own steps before its first tree move, then the commit point (``arm_commit_point``'s order).
from hermes_cli.update_stage import publish_stage
publish_stage("Downloading updates")
update_receipt.begin_update_receipt()
update_receipt._current.get().data["pre_update"]["sha"] = "1" * 40  # the code this process started on
if token is not None and token["platform"] == "posix":
    update_cmd_posix_pause._RUN = token
update_cmd_posix_pause.pause_at_commit_point()
update_cmd_commit.arm_commit_obligations(Path(ROOT), "")

found = []


class NewTreeFinder:
    def find_spec(self, name, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        where = [spec.origin or "", *(spec.submodule_search_locations or [])] if spec else []
        if any(w.startswith(REPO + os.sep) and "site-packages" not in w for w in map(str, where)):
            found.append(name)
        return None


sys.meta_path.insert(0, NewTreeFinder())
outcome = "returned"
try:
    if CASE == "interrupt_after_commit":
        def interrupted(request):
            raise KeyboardInterrupt
        update_cmd.run_completion = interrupted
        update_cmd._complete_source_update(request())
    elif CASE == "lost_completion":
        def lost(request):
            raise OSError("completion process could not start")
        update_cmd.run_completion = lost
        update_cmd._complete_source_update(request(token))
    elif CASE in ("systemd_resume", "launchd_resume", "windows_resume"):
        update_cmd._resume_paused_gateways_at_exit(token, request(token))
    elif CASE == "shallow_count":
        plan = update_cmd._prepare_checkout_for_update(
            ["git"], "main", "main", is_fork=False, assume_yes=True, gateway_mode=False,
            gw_input_fn=input, switch_branch=False, _windows_gateway_resume=None)
        outcome = f"commit_count={plan.commit_count}"
except BaseException as exc:
    outcome = f"{type(exc).__name__}: {exc}"
print("RESULT " + json.dumps({"found": found, "outcome": outcome}))
'''

# What each driven path ends with: proof it ran to the post-move code, not into a setup error.
# A restart path also has to reach its supervisor (the fakes append their argv to supervisor.log).
_SUPERVISOR_CALLS = {
    "lost_completion": "bare-replay",
    "systemd_resume": "start hermes-gateway",
    "launchd_resume": "bootstrap gui/0",
    "windows_resume": "restart watcher",
}
_OUTCOMES = {
    "interrupt_after_commit": "SystemExit: 130",
    "lost_completion": "returned",
    "systemd_resume": "returned",
    "launchd_resume": "returned",
    "windows_resume": "returned",
    "shallow_count": "commit_count=-1",
}


def _git(cwd, env, *args):
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _shallow_clone_behind(tmp_path: Path, env: dict) -> Path:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, env, "init", "-q", "-b", "main")
    for n in range(1, 6):
        (origin / "f.txt").write_text(f"{n}\n", encoding="utf-8")
        _git(origin, env, "add", "f.txt")
        _git(origin, env, "commit", "-q", "-m", f"c{n}")
        if n == 3:
            _git(tmp_path, env, "clone", "-q", "--depth", "1", origin.as_uri(), "clone")
    _git(tmp_path / "clone", env, "fetch", "-q", "--depth", "1", "origin", "main")
    return tmp_path / "clone"


_POSIX_CASES = sorted(set(_OUTCOMES) - {"windows_resume"})


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", _POSIX_CASES)
def test_post_move_path_imports_nothing_from_the_checkout(case, tmp_path, monkeypatch):
    _run_case(case, tmp_path, monkeypatch)


@pytest.mark.platforms("windows")
def test_windows_restart_after_the_move_imports_nothing_from_the_checkout(tmp_path, monkeypatch):
    _run_case("windows_resume", tmp_path, monkeypatch)


def _run_case(case, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "hermes-home").mkdir()
    root = tmp_path / "install"
    root.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for supervisor in ("systemctl", "launchctl", "sudo") if sys.platform != "win32" else ():
        (fake_bin / supervisor).write_text(f'#!/bin/sh\necho "{supervisor} $*" >> "{root}/supervisor.log"\nexit 1\n')
        (fake_bin / supervisor).chmod(0o755)
    # A home of its own (not the account default): the unit and job names carry its suffix.
    env = {**os.environ, "HOME": str(home), "HERMES_HOME": str(tmp_path / "hermes-home"),
           "USERPROFILE": str(home), "LOCALAPPDATA": str(home / "AppData" / "Local"),
           "APPDATA": str(home / "AppData" / "Roaming"), "XDG_STATE_HOME": str(tmp_path / "state"), "XDG_CONFIG_HOME": str(tmp_path / "config"),
           "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}", "GITHUB_TOKEN": "offline",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_NOSYSTEM": "1"}
    env.pop("PYTEST_CURRENT_TEST", None)
    label = clone = ""
    if case == "launchd_resume":
        monkeypatch.setenv("HERMES_HOME", env["HERMES_HOME"])
        from hermes_agent.hermes_cli.gateway_launchd import get_launchd_label

        label = get_launchd_label()
    if case == "shallow_count":
        clone = str(_shallow_clone_behind(tmp_path, env))
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(driver), str(REPO), str(root), case, label, clone],
        cwd=clone or tmp_path, env=env, capture_output=True, text=True, timeout=240, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, f"driver failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    result = json.loads(lines[-1][len("RESULT "):])
    assert result["outcome"] == _OUTCOMES[case], proc.stdout[-3000:]
    if case in _SUPERVISOR_CALLS:
        log = root / "supervisor.log"
        calls = log.read_text(encoding="utf-8") if log.exists() else ""
        assert _SUPERVISOR_CALLS[case] in calls, f"{case} never reached its restart: {calls!r}"
    assert result["found"] == [], f"{case} imported from the moved checkout: {sorted(set(result['found']))}"
