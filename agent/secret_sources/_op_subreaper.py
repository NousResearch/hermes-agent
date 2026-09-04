"""Isolated Linux lifecycle helper for one 1Password CLI invocation.

This module is executed as a standalone, stdlib-only child.  It becomes a
child subreaper before launching ``op`` so any double-forked daemon remains an
attestable descendant instead of escaping directly to PID 1.
"""

from __future__ import annotations

import ctypes
import errno
import os
import select
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

PR_SET_CHILD_SUBREAPER = 36
OP_TIMEOUT_SECONDS = 30.0
TERM_TIMEOUT_SECONDS = 5.0
ADOPTION_TIMEOUT_SECONDS = 2.0
EMPTY_SCANS_REQUIRED = 5
EMPTY_SCAN_INTERVAL_SECONDS = 0.1
HOLD_EXIT = 125
OP_DAEMON_CMDLINE = bytes((111, 112, 0, 100, 97, 101, 109, 111, 110, 0))


def _libc_function(name: str):
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        return getattr(libc, name)
    except AttributeError as exc:
        raise RuntimeError(f"{name} unavailable; helper HOLD") from exc


def _set_child_subreaper() -> None:
    function = _libc_function("prctl")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    function.restype = ctypes.c_int
    if function(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _pidfd_open(pid: int) -> int:
    function = _libc_function("pidfd_open")
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _pidfd_send_sigterm(pid_fd: int) -> None:
    function = _libc_function("pidfd_send_signal")
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(pid_fd, signal.SIGTERM, None, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _wait_pidfd_exit(pid_fd: int, timeout: float = TERM_TIMEOUT_SECONDS) -> bool:
    poller = select.poll()
    poller.register(pid_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    deadline = time.monotonic() + timeout
    while True:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        try:
            if poller.poll(remaining_ms):
                return True
        except InterruptedError:
            continue
        return False


def _proc_identity(pid: int) -> dict:
    proc = Path("/proc") / str(pid)
    raw_stat = (proc / "stat").read_text(encoding="ascii")
    marker = raw_stat.rfind(") ")
    if marker < 0:
        raise RuntimeError("malformed child stat; helper HOLD")
    fields = raw_stat[marker + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError("short child stat; helper HOLD")
    executable = (proc / "exe").stat()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "start_ticks": int(fields[19]),
        "uid": proc.stat().st_uid,
        "cmdline": (proc / "cmdline").read_bytes(),
        "cgroup": (proc / "cgroup").read_bytes(),
        "exe": (executable.st_dev, executable.st_ino),
    }


def _child_pids() -> list[int]:
    path = Path("/proc/self/task") / str(os.getpid()) / "children"
    raw = path.read_text(encoding="ascii").strip()
    return [int(value) for value in raw.split()] if raw else []


def _reap_exited_children() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _exact_adopted_daemon(
    identity: dict,
    *,
    binary_identity: tuple[int, int],
    expected_cgroup: bytes,
) -> bool:
    return (
        identity["ppid"] == os.getpid()
        and identity["uid"] == os.getuid()  # windows-footgun: ok
        and identity["cmdline"] == OP_DAEMON_CMDLINE
        and identity["cgroup"] == expected_cgroup
        and identity["exe"] == binary_identity
    )


def _cleanup_adopted_children(binary: Path) -> None:
    binary_stat = binary.resolve(strict=True).stat()
    binary_identity = (binary_stat.st_dev, binary_stat.st_ino)
    expected_cgroup = Path("/proc/self/cgroup").read_bytes()
    deadline = time.monotonic() + ADOPTION_TIMEOUT_SECONDS
    consecutive_empty = 0
    while True:
        _reap_exited_children()
        exact: list[dict] = []
        unknown: list[int] = []
        for pid in _child_pids():
            try:
                identity = _proc_identity(pid)
            except FileNotFoundError:
                continue
            except (OSError, RuntimeError, ValueError):
                unknown.append(pid)
                continue
            if _exact_adopted_daemon(
                identity,
                binary_identity=binary_identity,
                expected_cgroup=expected_cgroup,
            ):
                exact.append(identity)
            else:
                unknown.append(pid)

        if len(exact) > 1:
            raise RuntimeError("multiple adopted op daemons; helper HOLD")
        if exact:
            identity = exact[0]
            pid_fd = _pidfd_open(identity["pid"])
            try:
                current = _proc_identity(identity["pid"])
                if current != identity or not _exact_adopted_daemon(
                    current,
                    binary_identity=binary_identity,
                    expected_cgroup=expected_cgroup,
                ):
                    raise RuntimeError("adopted daemon identity drift; helper HOLD")
                _pidfd_send_sigterm(pid_fd)
                if not _wait_pidfd_exit(pid_fd):
                    raise RuntimeError("adopted daemon ignored SIGTERM; helper HOLD")
            finally:
                os.close(pid_fd)
            consecutive_empty = 0
            continue

        if unknown:
            consecutive_empty = 0
            if time.monotonic() >= deadline:
                raise RuntimeError("unknown adopted op child; helper HOLD")
            time.sleep(EMPTY_SCAN_INTERVAL_SECONDS)
            continue

        consecutive_empty += 1
        if consecutive_empty >= EMPTY_SCANS_REQUIRED:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("adopted child quiescence timeout; helper HOLD")
        time.sleep(EMPTY_SCAN_INTERVAL_SECONDS)


def _terminate_direct_child(proc: subprocess.Popen[bytes], pid_fd: int) -> None:
    try:
        _pidfd_send_sigterm(pid_fd)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise
    if not _wait_pidfd_exit(pid_fd):
        raise RuntimeError("op child ignored SIGTERM; helper HOLD")


def _main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print("invalid op helper argv", file=sys.stderr)
        return HOLD_EXIT
    command = argv[2:]
    binary = Path(command[0])
    try:
        _set_child_subreaper()
        proc = subprocess.Popen(
            command,
            env=dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pid_fd = _pidfd_open(proc.pid)
        try:
            try:
                stdout, stderr = proc.communicate(timeout=OP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate_direct_child(proc, pid_fd)
                try:
                    proc.communicate(timeout=TERM_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
                raise RuntimeError("op read timeout; helper HOLD")
        finally:
            os.close(pid_fd)

        _cleanup_adopted_children(binary)
    except Exception as exc:
        print(
            f"1Password isolated lifecycle failed: {type(exc).__name__}; helper HOLD",
            file=sys.stderr,
        )
        return HOLD_EXIT

    if proc.returncode != 0:
        if stderr:
            sys.stderr.buffer.write(stderr)
        return proc.returncode if 0 < proc.returncode < 125 else HOLD_EXIT
    sys.stdout.buffer.write(stdout)
    if stderr:
        sys.stderr.buffer.write(stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
