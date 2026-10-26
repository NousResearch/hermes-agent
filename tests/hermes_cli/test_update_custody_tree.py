"""#144439: where there is no parent-death signal (macOS), updater git custody owns the git
TREE, not git's pid: a timeout, or the owner's death, ends the helpers git started too.

Real git, real processes. The helper is the fetch's upload-pack (a shell git starts), which
stands in for git-remote-https / index-pack. Linux runs the same code with the parent-death
signal switched off; the macOS tests-os lane runs it natively.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env() -> dict:
    return dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if Path(f"/proc/{pid}/stat").exists():  # Linux: a zombie is dead
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return stat[stat.rfind(")") + 2:][:1] != "Z"
    return True


def _gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def fetch_rig(tmp_path, monkeypatch):
    for key, value in _env().items():
        monkeypatch.setenv(key, value)
    repo, remote = tmp_path / "repo", tmp_path / "remote.git"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["commit", "-q", "--allow-empty", "-m", "one"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(remote)], check=True)
    helper_pid = tmp_path / "helper.pid"
    helper = tmp_path / "helper.sh"
    helper.write_text(f"#!/bin/sh\necho $$ > {helper_pid}.tmp && mv {helper_pid}.tmp {helper_pid}\nsleep 60\n",
                      encoding="utf-8")
    helper.chmod(0o755)
    yield repo, helper, helper_pid
    if helper_pid.exists():
        subprocess.run(["kill", "-9", helper_pid.read_text(encoding="utf-8").strip()], capture_output=True, check=False)


def _wait_pid(path: Path) -> int:
    for _ in range(400):
        if path.exists():
            return int(path.read_text(encoding="utf-8").strip())
        time.sleep(0.05)
    raise AssertionError("the fetch never started its helper")


@pytest.mark.platforms("posix")
def test_a_timed_out_fetch_takes_its_helpers_with_it(fetch_rig, tmp_path, monkeypatch):
    from hermes_cli import update_custody

    repo, helper, helper_pid = fetch_rig
    monkeypatch.setattr(update_custody, "_held", lambda: {"fd": None})
    monkeypatch.setattr(update_custody, "_death_signal_preexec", lambda: None)
    with pytest.raises(subprocess.TimeoutExpired):
        update_custody.run_git(["git"], ["fetch", f"--upload-pack={helper}", "origin"], cwd=str(repo),
                               capture_output=True, timeout=2)
    assert _gone_within(_wait_pid(helper_pid), 5), "the fetch's helper outlived the timed-out fetch"


@pytest.mark.platforms("posix")
def test_a_killed_owner_takes_the_fetch_helpers_with_it(fetch_rig, tmp_path):
    repo, helper, helper_pid = fetch_rig
    owner_script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from pathlib import Path
        from hermes_cli import update_custody, update_lock as ul, update_cmd
        update_custody._death_signal_preexec = lambda: None  # what macOS has
        lock = ul.UpdateLock(path=Path(sys.argv[2]), install_root=sys.argv[1])
        assert lock.acquire()
        update_cmd._git_run(["git"], ["fetch", "--upload-pack=" + sys.argv[3], "origin"], cwd=sys.argv[1], network=True)
    """)
    owner = subprocess.Popen([sys.executable, "-c", owner_script, str(repo), str(tmp_path / "marker"), str(helper)],
                             env=_env())
    try:
        pid = _wait_pid(helper_pid)
        owner.kill()
        owner.wait(timeout=10)
        assert _gone_within(pid, 10), "the fetch's helper outlived its killed update owner"
    finally:
        if owner.poll() is None:
            owner.kill()
