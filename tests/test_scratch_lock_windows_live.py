"""Live Windows E2E: the scratch prune's ownership decision against REAL Windows processes.

Runs only in the on-demand ``wine2e`` lane (``windows-venv-e2e.yml``, ``-m "platforms and
integration"``): two tests step the system clock (``Set-Date``, admin runner), which must never
happen on a shared OS lane. Every process is real (python.exe / pythonw.exe / cmd.exe, a Job
Object, TerminateProcess); the prune is the production one; nothing is mocked.
"""
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_constants import get_scratch_dir, prune_scratch_dir
from tests.waiting import wait_until

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.integration]

REPO = str(Path(__file__).resolve().parents[1])
NO_WINDOW = 0x08000000

GATEWAY = r'''
import json, os, subprocess, sys, time
repo, lane, out, shell, go, job = sys.argv[1:7]
sys.path.insert(0, repo)
from hermes_cli.process_identity import attach_self_to_kill_on_close_job, register_self
ok = register_self("gateway")
in_job = attach_self_to_kill_on_close_job() if job == "job" else False
if go != "-":
    with open(out + ".registered", "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    while not os.path.exists(go):
        time.sleep(0.02)
python = sys.executable.replace("pythonw.exe", "python.exe")
argv = (["cmd.exe", "/d", "/c", "ping -n 900 127.0.0.1 >NUL"] if shell == "cmd"
        else [python, "-c", "import time; time.sleep(900)"])
sh = subprocess.Popen(argv, cwd=lane, stdin=subprocess.DEVNULL, creationflags=0x08000000)
with open(out + ".tmp", "w", encoding="utf-8") as f:
    f.write(json.dumps({"gateway": os.getpid(), "shell": sh.pid, "registered": ok, "in_job": in_job}))
os.replace(out + ".tmp", out)
time.sleep(900)
'''

ORPHAN = r'''
import subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(900)"], cwd=sys.argv[1],
                         stdin=subprocess.DEVNULL, creationflags=0x08000000)
open(sys.argv[2], "w", encoding="utf-8").write(str(child.pid))
'''

REGISTRY = r'''
import json, os, sys, time
repo, lane, out, role = sys.argv[1:5]
sys.path.insert(0, repo)
from hermes_cli.process_identity import register_self
from tools.process_registry import ProcessRegistry
register_self("gateway")
registry = ProcessRegistry()
python = sys.executable.replace("\\", "/")
info = {"registry": os.getpid()}
if role == "spawn":
    # A native child the registry tracks (adopt_local: the yield-to-background path). A Git Bash
    # background session dies with a TerminateProcess'd gateway on Windows (seen live), so only a
    # process that outlives its gateway can be re-adopted at all.
    import subprocess
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(900)"], cwd=lane, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=0x08000000)
    info["worker"] = registry.adopt_local(proc, command="bg-worker", cwd=lane).pid
elif role == "hermes":
    # A background `hermes ...` the agent starts: it takes its own owner lock at start.
    report = (out + ".child").replace("\\", "/")  # forward slashes: the command goes through bash
    child = (f"import os, sys, time, json; sys.path.insert(0, {repo.replace(chr(92), '/')!r}); t = time.monotonic(); "
             "from hermes_constants_scratch_lock import hold_owner_lock; ok = hold_owner_lock(); "
             f"open({report!r}, 'w', encoding='utf-8').write(json.dumps([os.getpid(), ok, time.monotonic() - t])); "
             "time.sleep(900)")
    registry.spawn_local(f'"{python}" -c "{child}"', cwd=lane)
    while not os.path.exists(report):
        time.sleep(0.05)
    time.sleep(0.2)
    info["child"] = json.load(open(report, encoding="utf-8-sig"))
else:
    if role == "boot-recover":
        # The restarted gateway's scratch prune is due at boot and runs (detached) before the
        # gateway re-adopts what the checkpoint records: wait for its verdict on the lane first.
        from hermes_constants import get_scratch_dir
        get_scratch_dir()
        log = os.path.join(os.environ["HERMES_HOME"], "logs", "scratch-prune.log")
        read = lambda: open(log, encoding="utf-8-sig", errors="replace").read() if os.path.exists(log) else ""
        deadline = time.monotonic() + 90
        while "lane" not in read() and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(1)  # the pass's reap (TERM, then KILL) settles
        info["boot_prune_log"] = read()[-1500:]
    checkpoint = os.path.join(os.environ["HERMES_HOME"], "processes.json")
    entries = json.load(open(checkpoint, encoding="utf-8-sig")) if os.path.exists(checkpoint) else []
    info["checkpoint"] = [(e.get("pid"), e.get("host_start_time"), registry._safe_host_start_time(e.get("pid")),
                           registry._detached_host_fate(e.get("pid"), e.get("host_start_time")), e.get("output_log"))
                          for e in entries]
    info.update(recovered=registry.recover_from_checkpoint(), pids=[s.pid for s in registry._running.values()])
with open(out + ".tmp", "w", encoding="utf-8") as f:
    f.write(json.dumps(info))
os.replace(out + ".tmp", out)
time.sleep(900)
'''

TUI_BACKEND = r'''
import json, os, subprocess, sys, threading, time
repo, lane, out = sys.argv[1:4]
sys.path.insert(0, repo)
import hermes_constants_scratch_lock as owner_locks
import tui_gateway.entry as entry
helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(900)"], cwd=lane,
                          stdin=subprocess.DEVNULL, creationflags=0x08000000)


def ready():  # once main() holds the backend's owner lock
    own = os.path.join(owner_locks.owners_dir(), f"{os.getpid()}.lock")
    while not owner_locks._ours(own):
        time.sleep(0.05)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        f.write(json.dumps({"backend": os.getpid(), "helper": helper.pid}))
    os.replace(out + ".tmp", out)


threading.Thread(target=ready, daemon=True).start()
entry.main()  # serves JSON-RPC on stdin until it closes
'''


class World:
    """A throwaway Hermes home with a scratch dir whose entries can be aged idle."""

    def __init__(self, tmp: Path):
        self.tmp, self.home = tmp, tmp / "home"
        self.home.mkdir()
        self.scratch = get_scratch_dir(self.home, prune=False)
        self.spawned: list[int] = []
        (tmp / "gateway.py").write_text(GATEWAY, encoding="utf-8")
        (tmp / "orphan.py").write_text(ORPHAN, encoding="utf-8")
        (tmp / "registry.py").write_text(REGISTRY, encoding="utf-8")
        (tmp / "tui_backend.py").write_text(TUI_BACKEND, encoding="utf-8")

    def entry(self, name):
        path = self.scratch / name
        path.mkdir()
        (path / "f.txt").write_text("fixture", encoding="utf-8")
        return path

    def age(self):
        ancient = time.time() - 30 * 3600
        for path in [*self.scratch.rglob("*"), self.scratch]:
            os.utime(path, (ancient, ancient))

    def gateway(self, lane, *, shell="python", interpreter=sys.executable, go="-", job="-"):
        """A registered entry point (``register_self``) running a shell inside *lane*."""
        out = self.tmp / f"gw-{lane.name}.json"
        proc = subprocess.Popen(
            [interpreter, str(self.tmp / "gateway.py"), REPO, str(lane), str(out), shell, go, job],
            cwd=str(self.tmp), stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
            env={**os.environ, "HERMES_HOME": str(self.home)})
        self.spawned.append(proc.pid)
        if go != "-":
            wait_until(lambda: Path(str(out) + ".registered").exists(), timeout=90, what="gateway registered")
            return out
        return json.loads(wait_until(lambda: out.exists() and out.read_text(encoding="utf-8-sig"), timeout=90, what="gateway up"))

    def _run(self, script, lane, tag, *extra, stdin=subprocess.DEVNULL, stdout=None):
        out = self.tmp / f"{tag}.json"
        proc = subprocess.Popen([sys.executable, str(self.tmp / script), REPO, str(lane), str(out), *extra],
                                cwd=str(self.tmp), stdin=stdin, stdout=stdout, creationflags=NO_WINDOW,
                                env={**os.environ, "HERMES_HOME": str(self.home)})
        self.spawned.append(proc.pid)
        info = json.loads(wait_until(lambda: out.exists() and out.read_text(encoding="utf-8-sig"), timeout=120,
                                     what=f"{script} {tag} up"))
        return proc, info

    def registry(self, lane, role):
        """A gateway-like process whose process registry spawns a background worker inside *lane*
        (``spawn``), a background Hermes command there (``hermes``), or recovers the workers
        recorded in the checkpoint (``recover``)."""
        return self._run("registry.py", lane, f"registry-{role}", role)[1]

    def tui_backend(self, lane):
        """The real ``tui_gateway.entry.main`` (the TUI's Python backend) with a child inside *lane*
        (an MCP helper it spawned); its stdin stays open so it keeps serving."""
        return self._run("tui_backend.py", lane, "tui-backend", stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)[1]

    def orphan(self, cwd):
        """A sleeper whose spawner exits right away (a leaked headless browser)."""
        pidfile = self.tmp / f"orphan-{cwd.name}.pid"
        subprocess.run([sys.executable, str(self.tmp / "orphan.py"), str(cwd), str(pidfile)], check=True,
                       stdin=subprocess.DEVNULL, timeout=60, creationflags=NO_WINDOW)
        pid = int(pidfile.read_text(encoding="utf-8-sig"))
        self.spawned.append(pid)
        return pid

    def prune(self):
        return prune_scratch_dir(self.scratch)

    def log(self):
        path = self.home / "logs" / "scratch-prune.log"
        return path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    for pid in w.spawned:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, stdin=subprocess.DEVNULL,
                       creationflags=NO_WINDOW, timeout=60)


def alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False


def gone(pid: int) -> bool:
    return not alive(pid)


@contextlib.contextmanager
def clock_stepped_back(hours: int = 1):
    """Step the Windows system clock back (an NTP/VM-resume correction), restore it after."""
    def step(delta):
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                        f"Set-Date -Adjust ([TimeSpan]::FromHours({delta})) | Out-Null"],
                       check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=60, creationflags=NO_WINDOW)
    before = time.time()
    step(-hours)
    assert time.time() < before - hours * 3600 + 600, "the clock did not step back (runner not admin?)"
    try:
        yield
    finally:
        step(hours)


def test_registered_gateways_cmd_shell_in_an_idle_entry_is_kept(world):
    lane = world.entry("lane")
    gw = world.gateway(lane, shell="cmd")
    world.age()
    world.prune()
    assert alive(gw["shell"]), "the live gateway's cmd.exe shell was killed"
    assert lane.exists()
    assert f"kept {str(lane)!r} — live Hermes process pid=" in world.log(), world.log()


def test_windowless_pythonw_gateway_keeps_its_childs_entry(world):
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    assert pythonw.exists(), pythonw
    lane = world.entry("lane")
    gw = world.gateway(lane, interpreter=str(pythonw))
    world.age()
    world.prune()
    assert alive(gw["shell"]) and lane.exists()


def test_gateway_in_a_kill_on_close_job_keeps_its_childs_entry(world):
    lane = world.entry("lane")
    gw = world.gateway(lane, job="job")
    world.age()
    world.prune()
    assert alive(gw["shell"]) and lane.exists(), gw


def test_gateway_killed_by_terminate_process_frees_its_shell_for_the_reap(world):
    lane = world.entry("lane")
    gw = world.gateway(lane, shell="cmd")
    psutil.Process(gw["gateway"]).kill()  # TerminateProcess
    wait_until(lambda: gone(gw["gateway"]), timeout=30, what="gateway terminated")
    world.age()
    world.prune()
    wait_until(lambda: gone(gw["shell"]), timeout=30, what="the orphaned shell reaped")
    assert not lane.exists()


def _gateway_whose_owners_dir_is_unusable(world, lane):
    """A file squats on the owners dir's path, so the gateway's owner lock cannot be taken:
    ownership is unknown, not \"nobody\"."""
    (world.home / "locks").mkdir()
    (world.home / "locks" / "scratch-owners").write_text("not a dir", encoding="utf-8")
    return world.gateway(lane, shell="cmd")["shell"], "scratch owner locks unavailable"


def _recovered_worker(world, lane):
    """A gateway's registry tracks a background worker inside *lane*, the gateway is terminated
    (TerminateProcess, so its vouch is gone), and the next gateway re-adopts the worker from the
    checkpoint. Returns the worker pid."""
    first = world.registry(lane, "spawn")
    worker = first["worker"]
    world.spawned.append(worker)
    assert alive(worker), first
    psutil.Process(first["registry"]).kill()
    wait_until(lambda: gone(first["registry"]), timeout=30, what="first gateway terminated")
    assert alive(worker), "the worker died with the gateway that spawned it"
    nxt = world.registry(lane, "recover")
    assert nxt["recovered"] == 1 and worker in nxt["pids"], (nxt, first)
    return worker, ""


def _worker_kept_by_the_boot_prune_then_recovered(world, lane):
    """The gateway is terminated; the next one's boot-time prune is due and runs BEFORE it
    re-adopts the worker, so the worker holds no lock in that window. The checkpoint still records
    it as the same process: the prune keeps it, and the next gateway re-adopts it."""
    first = world.registry(lane, "spawn")
    worker = first["worker"]
    world.spawned.append(worker)
    psutil.Process(first["registry"]).kill()
    wait_until(lambda: gone(first["registry"]), timeout=30, what="first gateway terminated")
    assert alive(worker), "the worker died with the gateway that spawned it"
    world.age()
    nxt = world.registry(lane, "boot-recover")
    assert alive(worker), f"the boot prune killed the worker before its re-adoption:\n{nxt['boot_prune_log']}"
    assert nxt["recovered"] == 1 and worker in nxt["pids"], (nxt, first)
    return worker, "recorded in"


def _tui_backend_helper(world, lane):
    return world.tui_backend(lane)["helper"], ""


def _background_hermes_command(world, lane):
    """The registry starts a background Hermes command inside *lane*: it must hold its own owner
    lock at once (never wait on the spawner's vouch), and keep it after the spawner is gone."""
    info = world.registry(lane, "hermes")
    child, held, took = info["child"]
    world.spawned.append(child)
    assert held and took < 1.0, info
    psutil.Process(info["registry"]).kill()
    wait_until(lambda: gone(info["registry"]), timeout=30, what="the spawning gateway terminated")
    return child, ""


@pytest.mark.parametrize("make", [_recovered_worker, _worker_kept_by_the_boot_prune_then_recovered,
                                  _tui_backend_helper, _background_hermes_command,
                                  _gateway_whose_owners_dir_is_unusable],
                         ids=["recovered-by-next-gateway", "boot-prune-before-recovery", "tui-backend-helper",
                              "background-hermes-command", "owners-dir-unusable"])
def test_a_process_hermes_owns_or_may_own_keeps_its_idle_entry(world, make):
    """Each process is Hermes-owned through an owner lock, or its ownership cannot be known: neither
    it nor its idle entry may be pruned, and an unknown is named in the audit log."""
    lane = world.entry("lane")
    pid, must_log = make(world, lane)
    world.age()
    world.prune()
    assert alive(pid), f"the process was killed:\n{world.log()}"
    assert lane.exists(), f"its entry was deleted:\n{world.log()}"
    assert must_log in world.log(), world.log()


def test_orphan_with_cwd_in_an_idle_entry_is_reaped_and_the_entry_removed(world):
    lane = world.entry("lane")
    orphan = world.orphan(lane)
    world.age()
    assert world.prune() == 1
    wait_until(lambda: gone(orphan), timeout=30, what="orphan reaped")
    assert not lane.exists()


def test_corrupt_spawn_ledger_does_not_disown_a_live_gateway(world):
    lane = world.entry("lane")
    gw = world.gateway(lane, shell="cmd")
    (world.home / "spawn-ledger.json").write_text("{not json", encoding="utf-8")
    world.age()
    world.prune()
    assert alive(gw["shell"]) and lane.exists()


def test_a_backward_clock_step_does_not_disown_a_gateways_new_shell(world):
    """The gateway starts, the clock steps back an hour, then it opens a shell: the shell's
    recorded start is now EARLIER than its parent's. Ownership must not depend on that order."""
    lane = world.entry("lane")
    go = world.tmp / "go"
    out = world.gateway(lane, shell="cmd", go=str(go))
    gateway = psutil.Process(int(Path(str(out) + ".registered").read_text(encoding="utf-8-sig")))
    born_before = gateway.create_time()
    with clock_stepped_back():
        go.touch()
        gw = json.loads(wait_until(lambda: out.exists() and out.read_text(encoding="utf-8-sig"), timeout=90, what="shell up"))
        world.age()
        world.prune()
        diag = {"gateway_create_time_before_step": born_before, "during_step": gateway.create_time(),
                "shell_create_time": psutil.Process(gw["shell"]).create_time() if alive(gw["shell"]) else None}
    assert alive(gw["shell"]), f"the gateway's shell was killed after a clock step: {diag}\n{world.log()}"
    assert lane.exists()


DETACHED_SHELL = r'''@echo off
:wait
if not exist "%GO%" (ping -n 2 127.0.0.1 >NUL & goto wait)
"%PY%" -c "import os, sys; from pathlib import Path; os.chdir(r'%AWAY%'); sys.path.insert(0, r'%REPO%'); import hermes_constants_scratch_detach as d; d.spawn_prune(Path(r'%SCRATCH%'))"
ping -n 900 127.0.0.1 >NUL
'''


def test_detached_pass_spares_the_launchers_shell_after_a_clock_step(world):
    """A cmd.exe session works inside an idle entry and, after the clock stepped back, runs a
    Hermes command that hands the prune to a detached pass and exits. The pass must still know
    that session as the launcher's ancestor: it is never killed and its entry is kept."""
    lane, other = world.entry("lane"), world.entry("other")
    orphan = world.orphan(other)
    # The Hermes command steps out of the entry before spawning (the pass and its venv launcher
    # must not keep the lane by their own cwd), and a large idle entry makes the pass's scan
    # outlast the command, so at reap time only the recorded lineage can still name the session.
    bulk = world.scratch / "bulk"
    for d in range(40):
        (bulk / f"d{d}").mkdir(parents=True)
        for f in range(500):
            (bulk / f"d{d}" / f"f{f}").touch()
    go = world.tmp / "go"
    script = world.tmp / "session.cmd"
    script.write_text(DETACHED_SHELL, encoding="utf-8")
    world.age()
    env = {**os.environ, "HERMES_HOME": str(world.home), "GO": str(go), "PY": sys.executable, "REPO": REPO,
           "SCRATCH": str(world.scratch), "AWAY": str(world.tmp)}
    session = subprocess.Popen(["cmd.exe", "/d", "/c", str(script)], cwd=str(lane), env=env,
                               stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    world.spawned.append(session.pid)  # born before the step; it launches Hermes only after it
    with clock_stepped_back():
        go.touch()
        wait_until(lambda: gone(orphan), timeout=120, what="the detached pass reaped the other entry's orphan")
        wait_until(lambda: not any("hermes_constants_scratch_detach" in " ".join(p.info["cmdline"] or [])
                                   for p in psutil.process_iter(["cmdline"])), timeout=120,
                   what="the detached pass finished")
    log = world.log()
    assert alive(session.pid), f"the launcher's own session was killed by its detached prune:\n{log}"
    assert lane.exists(), f"the entry the launcher's session works in was deleted:\n{log}"
    assert "('cmd.exe') works inside" in log, f"the lane was kept for another reason:\n{log}"
    assert not other.exists()
