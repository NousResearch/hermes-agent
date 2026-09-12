"""Install lifetimes use private files and harmless child processes only."""
import fcntl
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins/hermes-realms"
load = runpy.run_path(str(PLUGIN / "realms/_binding.py"))["load_runtime"]
pytestmark = pytest.mark.linux_only


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    pytest.fail("lifetime fixture did not reach the expected state")


def unlocked(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def test_privileged_lifetime_retains_lock_after_backend_death(tmp_path):
    """A root-like operation may outlive its owner, but never its exclusion."""
    lock = tmp_path / "active.lock"
    witness = tmp_path / "started"
    done = tmp_path / "done"
    child = (
        "import pathlib,time; "
        f"pathlib.Path({str(witness)!r}).touch(); "
        "time.sleep(1.5); "
        f"pathlib.Path({str(done)!r}).touch()"
    )
    backend_code = """
import fcntl, os, runpy, sys
load = runpy.run_path(sys.argv[1])['load_runtime']
fd = os.open(sys.argv[2], os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
load('setup_worker').run_child([sys.executable, '-c', sys.argv[3]],
    env={'PATH':'/usr/bin:/bin'}, timeout=4, lock_fd=fd, privileged=True)
"""
    backend = subprocess.Popen([sys.executable, "-I", "-c", backend_code,
                                str(PLUGIN / "realms/_binding.py"), str(lock), child])
    try:
        wait_for(witness.exists)
        backend.kill()
        backend.wait(timeout=5)
        assert not unlocked(lock), "backend death released the live install lock"
        wait_for(done.exists)
        wait_for(lambda: unlocked(lock))
    finally:
        if backend.poll() is None:
            backend.kill()
        backend.wait(timeout=5)


def test_pidfd_fallback_observes_real_process_exit(monkeypatch):
    import select
    lifetime = load("setup_process")
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    fd = None
    try:
        fd = lifetime.open_pidfd(child.pid)
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        assert not poller.poll(0)
        child.kill()
        child.wait(timeout=5)
        assert poller.poll(1000)
    finally:
        if fd is not None:
            os.close(fd)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_packages_have_root_enforced_literal_timeout(tmp_path, monkeypatch):
    from dataclasses import asdict
    worker = load("setup_worker")
    config = load("config")
    home = tmp_path / "profile"
    plan = {"home": str(home), "kind": "omarchy-vm", "action": "start",
            "packages": ["qemu-full"], "config": asdict(config.Config())}
    calls = []
    monkeypatch.setattr(worker, "run_child", lambda argv, **kw: calls.append((argv, kw)))
    worker.install_plan(plan, lambda phase: None, lock_fd=77)
    argv, options = calls[0]
    assert argv == ["/usr/bin/pkexec", "/usr/bin/timeout", "--signal=TERM",
                    "--kill-after=10s", "1800s", "/usr/bin/pacman", "-S",
                    "--needed", "--noconfirm", "--", "qemu-full"]
    assert options["privileged"] is True
    assert options["lock_fd"] == 77
    assert options["timeout"] > 1810


def test_install_refuses_changed_config_before_launch(tmp_path, monkeypatch):
    from dataclasses import asdict
    worker = load("setup_worker")
    config = load("config")
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  realms:\n    vm:\n      memory: 4096\n")
    plan = {"home": str(home), "kind": "omarchy-vm", "action": "install",
            "packages": [], "config": asdict(config.Config())}
    monkeypatch.setattr(worker, "run_child", lambda *a, **kw: pytest.fail("launched with drift"))
    with pytest.raises(ValueError, match="config|proposal"):
        worker.install_plan(plan, lambda phase: None)


def test_qemu_binding_exists_before_vendor_launch(tmp_path, monkeypatch):
    """Inspect the exact drop-in that systemd applies to the vendor's unit."""
    lifetime = load("setup_process")
    unit = "hermes-vm-base-" + "a" * 32 + ".service"
    monkeypatch.setattr(lifetime, "unit_directory", lambda name: tmp_path / (name + ".d"))
    calls = []
    def control(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ActiveState=inactive\n", "")
    monkeypatch.setattr(lifetime.subprocess, "run", control)
    with lifetime.vm_lifetime(unit, timeout=45, env={"PATH": "/usr/bin:/bin"}):
        dropin = (tmp_path / (unit + ".d") / "50-setup-lifetime.conf").read_text()
        assert "BindsTo=" + unit.replace(".service", "-guard.service") in dropin
        assert "After=" + unit.replace(".service", "-guard.service") in dropin
        assert "RuntimeMaxSec=45" in dropin
        assert "TimeoutStopSec=10" in dropin
        assert "KillMode=control-group" in dropin
        guardian = next(argv for argv in calls if argv[0] == "/usr/bin/systemd-run")
        assert "--property=RuntimeMaxSec=165" in guardian
        assert "--watch" in guardian
    assert not (tmp_path / (unit + ".d")).exists()


def test_vm_cleanup_unknown_retains_exclusion(tmp_path, monkeypatch):
    lifetime = load("setup_process")
    unit = "hermes-vm-base-" + "b" * 32 + ".service"
    calls = []
    class StillExcluded(Exception):
        pass
    def control(argv, **kwargs):
        calls.append(argv)
        # A failed query with a plausible-looking inactive body is NOT proof.
        return subprocess.CompletedProcess(argv, 1, "ActiveState=inactive\n", "bus unavailable")
    monkeypatch.setattr(lifetime.subprocess, "run", control)
    monkeypatch.setattr(lifetime.time, "sleep", lambda seconds: (_ for _ in ()).throw(StillExcluded()))
    with pytest.raises(StillExcluded):
        lifetime.stop_vm(unit, env={})
    assert any("show" in argv for argv in calls)


@pytest.mark.integration
@pytest.mark.parametrize("end", ["backend-death", "deadline"])
def test_real_systemd_qemu_install_lifetime(tmp_path, end):
    """Real QEMU/systemd, no image, guest, package install or live profile."""
    import shutil
    import uuid
    if not shutil.which("qemu-system-x86_64"):
        pytest.skip("QEMU required for native install lifetime gate")
    lifetime = load("setup_process")
    env = load("setup_worker").worker_env(tmp_path / "profile")
    unit = "hermes-vm-base-" + uuid.uuid4().hex + ".service"
    lock = tmp_path / "active.lock"
    ready = tmp_path / "qemu-ready"
    child_code = """
import os, pathlib, runpy, subprocess, sys, time
load = runpy.run_path(sys.argv[1])['load_runtime']
with load('setup_process').vm_lifetime(sys.argv[2], timeout=30, env=dict(os.environ)):
    subprocess.run(['/usr/bin/systemd-run', '--user', '--quiet', '--collect',
        '--service-type=exec', '--unit=' + sys.argv[2],
        '/usr/bin/qemu-system-x86_64', '-machine', 'none', '-nodefaults',
        '-display', 'none', '-monitor', 'none', '-serial', 'none', '-S'], check=True)
    pathlib.Path(sys.argv[3]).touch()
    time.sleep(30)
"""
    backend_code = """
import fcntl, os, runpy, sys
load = runpy.run_path(sys.argv[1])['load_runtime']
fd = os.open(sys.argv[2], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
load('setup_worker').run_child([sys.executable, '-I', '-c', sys.argv[3],
    sys.argv[1], sys.argv[4], sys.argv[5]], env=dict(os.environ),
    timeout=float(sys.argv[6]), lock_fd=fd, cleanup_unit=sys.argv[4])
"""
    backend = subprocess.Popen([sys.executable, "-I", "-c", backend_code,
        str(PLUGIN / "realms/_binding.py"), str(lock), child_code, unit, str(ready),
        "30" if end == "backend-death" else "4"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(ready.exists)
        info = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
                               "--property=ActiveState,BindsTo,RuntimeMaxUSec,KillMode"],
                              env=env, capture_output=True, text=True, check=True).stdout
        assert "ActiveState=active" in info
        assert unit.replace(".service", "-guard.service") in info
        assert "RuntimeMaxUSec=30s" in info
        assert "KillMode=control-group" in info
        assert not unlocked(lock)
        if end == "backend-death":
            backend.kill()
        backend.wait(timeout=15)
        wait_for(lambda: unlocked(lock), timeout=15)
        observed = subprocess.run(["/usr/bin/systemctl", "--user", "show", unit,
                                   "--property=ActiveState"], env=env,
                                  capture_output=True, text=True, check=True).stdout
        assert observed.strip() in {"ActiveState=inactive", "ActiveState=failed"}
        assert not lifetime.unit_directory(unit).exists()
    finally:
        if backend.poll() is None:
            backend.kill()
        backend.wait(timeout=5)
        # Cleanup only this random test generation, never the native rig's VM.
        subprocess.run(["/usr/bin/systemctl", "--user", "stop", unit,
                        unit.replace(".service", "-guard.service")],
                       env=env, capture_output=True, timeout=15)
        lifetime._remove_dropin(unit, env)
