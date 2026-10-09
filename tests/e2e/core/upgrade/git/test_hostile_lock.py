"""Hostile-concurrency cells for the update lock, with real ``hermes update`` processes.

A real git checkout (a clone of the commit under test, served by a local bare origin) is updated
by real ``python -m hermes_cli.main update`` processes inside the bwrap sandbox. A ``git`` shim
parks every ``fetch`` while a hold file exists and logs which home reached it, so "an update got
past the lock" is observed directly: the fetch is the first step after the lock.

Cells (contract C1 + C1.7):
(a) two updates of ONE checkout from two different HERMES_HOMEs: the second must exit 2.
(b) a live owner whose v2 marker is 25 minutes old (matching creation time): refused, not aged out.
(c) a v2 marker whose pid was reused (creation time mismatch): reclaimed, the update proceeds.
(d) the update owner is SIGKILLed while a process of its tree still holds the checkout lock:
    a new update cannot start until that process exits.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
]

REFUSAL = "Another Hermes update is already running"

# Runs inside the sandbox (its own pid namespace), so owners, updates and completion children
# see each other's pids and creation times exactly as on a real machine.
_DRIVER = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path

root, checkout, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
envs = json.loads((root / "envs.json").read_text(encoding="utf-8"))
ARGV = [sys.executable, "-m", "hermes_cli.main", "update", "--yes", "--branch", "main", "--no-gateway-restart"]
# A running server's "is an update in flight" probe, in a fresh process of another home.
READER = """
import json, sys
sys.path.insert(0, sys.argv[1])
from hermes_cli.web_server_skew_exit import _update_in_progress
print(json.dumps({"skew_reader": _update_in_progress()}))
"""


def ct(pid):
    stat = Path(f"/proc/{pid}/stat").read_bytes()
    ticks = int(stat[stat.rindex(b")") + 2:].split()[19])
    btime = next(int(l.split()[1]) for l in Path("/proc/stat").read_bytes().splitlines() if l.startswith(b"btime "))
    return btime + ticks / os.sysconf("SC_CLK_TCK")


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def update(name, timeout=240):
    cp = subprocess.run(ARGV, env=envs[name], cwd=checkout, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return cp.returncode, cp.stdout


def completion_child(home):
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes()
            env = (p / "environ").read_bytes()
        except OSError:
            continue
        if b"update_completion.py" in cmd and f"HERMES_HOME={home}".encode() in env.split(b"\0"):
            return int(p.name)
    return None


result = {}
if mode == "owner":
    age, offset = float(sys.argv[4]), float(sys.argv[5])
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    time.sleep(0.2)
    marker = Path(envs["home-a"]["HERMES_HOME"]) / ".hermes-update-in-progress"
    marker.parent.mkdir(parents=True, exist_ok=True)
    body = f"{owner.pid}\n{int(time.time() - age)}\nct:{ct(owner.pid) + offset:.3f}\n"
    marker.write_text(body, encoding="utf-8")
    result["rc"], result["out"] = update("home-a")
    result["marker_kept"] = marker.exists() and marker.read_text(encoding="utf-8") == body
    owner.kill()
elif mode == "kill-mid-completion":
    first = subprocess.Popen(ARGV, env=envs["home-a"], cwd=checkout, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    deadline = time.time() + 240
    while child is None and time.time() < deadline and first.poll() is None:
        child = completion_child(envs["home-a"]["HERMES_HOME"])
        time.sleep(0.05)
    result["child_seen"] = child is not None
    os.kill(first.pid, signal.SIGKILL)
    first.wait()
    result["child_alive_before"] = child is not None and alive(child)
    result["rc"], result["out"] = update("home-b")
    result["child_alive_after"] = child is not None and alive(child)
    if child is not None and alive(child):
        os.kill(child, signal.SIGKILL)
elif mode == "orphan-reader":
    first = subprocess.Popen(ARGV, env=envs["home-a"], cwd=checkout, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    deadline = time.time() + 240
    while child is None and time.time() < deadline and first.poll() is None:
        child = completion_child(envs["home-a"]["HERMES_HOME"])
        time.sleep(0.05)
    result["child_seen"] = child is not None
    os.kill(first.pid, signal.SIGKILL)
    first.wait()
    # The killed owner's marker is gone (any reader compare-and-deletes a dead claim).
    (Path(envs["home-a"]["HERMES_HOME"]) / ".hermes-update-in-progress").unlink(missing_ok=True)
    reader = subprocess.run([sys.executable, "-c", READER, checkout], env=envs["home-b"], cwd=checkout,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    result["reader_out"] = reader.stdout[-2000:] + reader.stderr[-2000:]
    result["reader"] = json.loads(reader.stdout.strip().splitlines()[-1]) if reader.returncode == 0 else None
    result["child_alive_after"] = child is not None and alive(child)
    if child is not None and alive(child):
        os.kill(child, signal.SIGKILL)
result["out"] = result.get("out", "")[-4000:]
print(json.dumps(result))
'''


class Rig:
    def __init__(self, root: Path):
        self.root = root
        origin = I.make_origin(root, I.head_sha())
        self.checkout = root / "checkout"
        I.git("clone", "-q", "--shared", "-b", "main", str(origin), str(self.checkout), cwd=root)
        self.hold = root / "hold-fetch"
        self.fetch_log = root / "fetch.log"
        wrap = root / "wrap"
        wrap.mkdir()
        real_git = subprocess.run(["sh", "-c", "command -v git"], text=True, encoding="utf-8",
                                  capture_output=True).stdout.strip()
        (wrap / "git").write_text(
            "#!/usr/bin/env bash\n"
            'for a in "$@"; do if [ "$a" = fetch ]; then\n'
            f'  echo "fetch $HERMES_HOME" >> "{self.fetch_log}"\n'
            f'  for _ in $(seq 600); do [ -e "{self.hold}" ] || break; sleep 0.2; done\n'
            "fi; done\n"
            f'exec "{real_git}" "$@"\n', encoding="utf-8")
        (wrap / "git").chmod(0o755)
        self.wrap = wrap

    def env(self, name: str) -> dict[str, str]:
        env = H.isolated_env(self.root / name, extra_path=[self.wrap], pythonpath=self.checkout)
        return env

    def driver(self, *args: str, timeout: float = 300) -> dict:
        """Run ``_DRIVER`` in ONE sandbox: the pid namespace is shared by every process of the cell."""
        envs = {name: self.env(name) for name in ("home-a", "home-b")}
        (self.root / "envs.json").write_text(json.dumps(envs), encoding="utf-8")
        (self.root / "driver.py").write_text(_DRIVER, encoding="utf-8")
        argv = H.sandbox_argv([sys.executable, str(self.root / "driver.py"), str(self.root), str(self.checkout), *args],
                              writable=[self.root])
        cp = subprocess.run(argv, env=envs["home-a"], cwd=self.checkout, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=timeout)
        assert cp.returncode == 0, f"driver failed:\n{cp.stdout[-3000:]}\n{cp.stderr[-3000:]}"
        return json.loads(cp.stdout.strip().splitlines()[-1])

    def update(self, name: str) -> subprocess.Popen:
        env = self.env(name)
        argv = H.sandbox_argv([sys.executable, "-m", "hermes_cli.main", "update", "--yes", "--branch", "main",
                               "--no-gateway-restart"], writable=[self.root])
        return subprocess.Popen(argv, env=env, cwd=self.checkout, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def fetchers(self) -> list[str]:
        return self.fetch_log.read_text(encoding="utf-8-sig").splitlines() if self.fetch_log.exists() else []

    def marker(self, name: str) -> Path:
        return Path(self.env(name)["HERMES_HOME"]) / ".hermes-update-in-progress"


def _finish(proc: subprocess.Popen, timeout: float = 120) -> tuple[int, str]:
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        H.kill_tree(proc)
        out, _ = proc.communicate()
        return -999, out
    return proc.returncode, out


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def test_two_homes_cannot_update_one_checkout_at_once(rig):
    """(a) The checkout lock is keyed on the install root, not on HERMES_HOME (cli V3)."""
    rig.hold.touch()
    first = rig.update("home-a")
    try:
        H.wait_for(lambda: rig.fetchers(), timeout=180, what="the first update to reach its fetch")
        # --ignored: a lock file in the worktree shows whatever the tree's .gitignore says.
        status = I.git("status", "--porcelain", "--ignored", "--untracked-files=all", cwd=rig.checkout)
        held = [line for line in status.splitlines() if "hermes-update" in line]
        assert not held, f"the live update's lock is a worktree file (autostash takes it): {held}"
        code, out = _finish(rig.update("home-b"), timeout=60)
        assert code == 2 and REFUSAL in out, f"second home's update was not refused (rc={code}):\n{out[-4000:]}"
        assert len(rig.fetchers()) == 1, f"two updates reached the fetch concurrently: {rig.fetchers()}"
    finally:
        rig.hold.unlink()
        H.kill_tree(first)
        first.communicate()


def test_live_owner_is_not_aged_out(rig):
    """(b) A live owner (matching creation time) with a 25-minute-old claim still owns the lock."""
    r = rig.driver("owner", str(25 * 60), "0")
    assert r["rc"] == 2 and REFUSAL in r["out"], f"a live 25-min-old owner was aged out (rc={r['rc']}):\n{r['out']}"
    assert not rig.fetchers(), "the update ran past a live owner's lock"
    assert r["marker_kept"], "the live owner's marker was rewritten or deleted"


def test_reused_pid_marker_is_reclaimed(rig):
    """(c) A live pid whose creation time does not match the claim is a recycled pid: reclaim."""
    r = rig.driver("owner", "0", "-500")
    assert r["rc"] != 2 and REFUSAL not in r["out"], f"a recycled pid blocked the update (rc={r['rc']}):\n{r['out']}"
    assert rig.fetchers(), f"the update never got past the lock:\n{r['out']}"


def test_killed_owner_keeps_the_checkout_locked_while_its_tree_runs(rig):
    """(d) SIGKILL `hermes update` while its completion child runs: the child inherited the locked
    fd, so a new update (any home) is refused until the child exits; never starts mid-mutation."""
    r = rig.driver("kill-mid-completion", timeout=600)
    assert r["child_seen"] and r["child_alive_before"], f"no completion child outlived the owner: {r}"
    assert r["rc"] == 2 and REFUSAL in r["out"], (
        f"a new update started while the killed update's completion child ran (rc={r['rc']}, "
        f"child alive after: {r['child_alive_after']}):\n{r['out']}")


def test_orphaned_update_tree_reads_as_in_progress(rig):
    """(e) SIGKILL `hermes update` mid-completion and lose its marker: the completion child still
    holds the checkout lock, so every "is an update running" reader must still say yes."""
    r = rig.driver("orphan-reader", timeout=600)
    assert r["child_seen"] and r["child_alive_after"], f"premise: no completion child outlived the owner: {r}"
    assert r["reader"] is not None, f"reader process failed:\n{r['reader_out']}"
    assert r["reader"]["skew_reader"] is True, f"the server's update probe missed the live update tree: {r}"
