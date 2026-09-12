"""Independent installer supervisor (also executable under Python -I).

The backend may disappear at any point. This process retains its inherited
flock until the real operation and any detached QEMU unit have stopped. Never
LOCK_UN an inherited descriptor: that unlocks every holder of that description.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import time


def open_pidfd(pid):
    """Some standalone Python builds omit os.pidfd_open on capable kernels."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)  # windows-footgun: ok — Linux-only plugin
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.pidfd_open
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.set_inheritable(fd, False)
    return fd


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)  # windows-footgun: ok — Linux-only plugin
    except ProcessLookupError:
        pass
    except PermissionError:
        # pkexec may already have exec'd the root timeout. It enforces its own
        # deadline; retain exclusion until it exits, never pretend killpg worked.
        pass


def supervise(argv, env, timeout, owner_fd, *, privileged=False):
    """Root operations finish under their root-side timer after owner death."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid setup timeout")
    watcher = select.poll()
    watcher.register(owner_fd, select.POLLIN)
    if watcher.poll(0):
        return 125
    process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    deadline = time.monotonic() + timeout
    interrupted = False
    try:
        while process.poll() is None:
            owner_gone = bool(watcher.poll(50))
            expired = time.monotonic() >= deadline
            if expired or (owner_gone and not privileged):
                interrupted = True
                _kill_group(process)
                # A privileged process may be unsignalable. Its root timeout
                # must finish before this lock holder is allowed to go away.
                process.wait()
                break
            if owner_gone:
                time.sleep(.05)
        return 124 if interrupted else process.returncode
    finally:
        _kill_group(process)
        process.wait()


def _validate_unit(unit):
    if not re.fullmatch(r"hermes-vm-base-[0-9a-f]{32}\.service", unit):
        raise ValueError("Invalid setup unit")


def unit_directory(unit):
    _validate_unit(unit)
    return Path(f"/run/user/{os.getuid()}/systemd/user/{unit}.d")  # windows-footgun: ok — Linux-only plugin


def _guard_unit(unit):
    _validate_unit(unit)
    return unit.removesuffix(".service") + "-guard.service"


def _control(argv, env):
    return subprocess.run(["/usr/bin/systemctl", "--user", *argv], env=env,
                          capture_output=True, text=True, timeout=30)


def stop_vm(unit, *, env):
    """Do not release exclusion on unknown or merely requested cleanup."""
    names = (unit, _guard_unit(unit))
    while True:
        stopped = True
        for name in names:
            try:
                _control(["stop", name], env)
                observed = _control(["show", name, "--property=ActiveState"], env)
                info = dict(line.split("=", 1) for line in observed.stdout.splitlines() if "=" in line)
                stopped &= observed.returncode == 0 and info.get("ActiveState") in {"inactive", "failed"}
            except (OSError, subprocess.SubprocessError):
                stopped = False
        if stopped:
            return
        time.sleep(1)


def _remove_dropin(unit, env):
    directory = unit_directory(unit)
    (directory / "50-setup-lifetime.conf").unlink(missing_ok=True)
    try:
        directory.rmdir()
    except FileNotFoundError:
        pass
    _control(["daemon-reload"], env).check_returncode()


def _start_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def watch_owner(pid, ticks):
    """Capture a pidfd then validate its process generation, never kill -0."""
    try:
        fd = open_pidfd(pid)
    except ProcessLookupError:
        return
    try:
        try:
            if _start_ticks(pid) != ticks:
                return
        except FileNotFoundError:
            return
        watcher = select.poll()
        watcher.register(fd, select.POLLIN)
        watcher.poll()
    finally:
        os.close(fd)


@contextmanager
def vm_lifetime(unit, *, timeout, env):
    """Apply QEMU's bounds BEFORE the unmodified vendor starts its unit."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid setup timeout")
    directory = unit_directory(unit)
    guard = _guard_unit(unit)
    directory.mkdir(mode=0o700, parents=True)
    dropin = directory / "50-setup-lifetime.conf"
    dropin.write_text(
        f"[Unit]\nBindsTo={guard}\nAfter={guard}\n"
        f"[Service]\nRuntimeMaxSec={timeout:g}\nTimeoutStopSec=10\nKillMode=control-group\n",
        encoding="utf-8",
    )
    armed = False
    try:
        _control(["daemon-reload"], env).check_returncode()
        armed = True
        subprocess.run([
            "/usr/bin/systemd-run", "--user", "--quiet", "--collect",
            "--service-type=exec", "--unit=" + guard,
            f"--property=RuntimeMaxSec={timeout + 120:g}", "--property=TimeoutStopSec=10",
            sys.executable, "-I", str(Path(__file__).resolve()), "--watch",
            str(os.getpid()), _start_ticks(os.getpid()),
        ], env=env, capture_output=True, timeout=30, check=True)
        yield
    finally:
        if armed:
            stop_vm(unit, env=env)
        _remove_dropin(unit, env)


def main():
    if sys.argv[1] == "--watch":
        watch_owner(int(sys.argv[2]), sys.argv[3])
        return 0
    owner_fd = int(sys.argv[1])
    payload = json.load(sys.stdin)
    try:
        return supervise(payload["argv"], payload["env"], payload["timeout"], owner_fd,
                         privileged=payload.get("privileged", False))
    finally:
        if payload.get("cleanup_unit"):
            stop_vm(payload["cleanup_unit"], env=payload["env"])
            _remove_dropin(payload["cleanup_unit"], payload["env"])


if __name__ == "__main__":
    raise SystemExit(main())
