"""Darwin-only primitive observation and exact-child lifecycle support."""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

GIT_COMMANDS = (
    ("/usr/bin/git", "rev-parse", "--show-toplevel"),
    ("/usr/bin/git", "rev-parse", "--absolute-git-dir"),
    ("/usr/bin/git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
    ("/usr/bin/git", "rev-parse", "--is-inside-work-tree"),
    ("/usr/bin/git", "rev-parse", "--is-bare-repository"),
)
GIT_ENV = ("LC_ALL=C", "LANG=C", "GIT_CONFIG_NOSYSTEM=1", "HOME=/nonexistent")
_STDOUT_CAP = 4096
_STDERR_CAP = 4096
_TERMINATION_GRACE_NS = 250_000_000


def _libsystem():
    return ctypes.CDLL(ctypes.util.find_library("System"), use_errno=True)


def _spawn_git_at_fd(workdir_fd: int, argv: tuple[str, ...], *, timeout_seconds: float) -> dict:
    """Spawn exact /usr/bin/git at a directory FD and own/reap exactly its PID."""
    if argv not in GIT_COMMANDS:
        raise ValueError("unapproved git argv")
    lib = _libsystem()
    actions = ctypes.create_string_buffer(256)
    init = lib.posix_spawn_file_actions_init
    addfchdir = lib.posix_spawn_file_actions_addfchdir
    adddup2 = lib.posix_spawn_file_actions_adddup2
    addclose = lib.posix_spawn_file_actions_addclose
    destroy = lib.posix_spawn_file_actions_destroy
    spawn = lib.posix_spawn
    for fn in (init, destroy):
        fn.argtypes = [ctypes.c_void_p]; fn.restype = ctypes.c_int
    addfchdir.argtypes = [ctypes.c_void_p, ctypes.c_int]; addfchdir.restype = ctypes.c_int
    adddup2.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]; adddup2.restype = ctypes.c_int
    addclose.argtypes = [ctypes.c_void_p, ctypes.c_int]; addclose.restype = ctypes.c_int
    spawn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
    spawn.restype = ctypes.c_int
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    for fd in (out_r, out_w, err_r, err_w):
        os.set_inheritable(fd, False)
        os.set_blocking(fd, False)
    pid = ctypes.c_int(0)
    initialized = False
    child_reaped = False
    status = None
    try:
        rc = init(actions); initialized = rc == 0
        returns = [rc]
        if rc == 0:
            for call in ((addfchdir, workdir_fd), (adddup2, out_w, 1), (adddup2, err_w, 2),
                         (addclose, out_r), (addclose, err_r), (addclose, out_w), (addclose, err_w)):
                rc = call[0](actions, *call[1:]); returns.append(rc)
                if rc: break
        if any(returns):
            raise OSError(next(x for x in returns if x), "posix_spawn file action")
        encoded = [item.encode() for item in argv]
        argv_c = (ctypes.c_char_p * (len(encoded) + 1))(*encoded, None)
        env_encoded = [item.encode() for item in GIT_ENV]
        env_c = (ctypes.c_char_p * (len(env_encoded) + 1))(*env_encoded, None)
        rc = spawn(ctypes.byref(pid), b"/usr/bin/git", actions, None, argv_c, env_c)
        if rc:
            raise OSError(rc, "posix_spawn")
        os.close(out_w); out_w = -1
        os.close(err_w); err_w = -1
        deadline = time.monotonic() + timeout_seconds
        stdout_data, stderr_data = bytearray(), bytearray()
        buffers = {out_r: stdout_data, err_r: stderr_data}
        while buffers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("git operation deadline")
            try:
                ready, _, _ = select.select(tuple(buffers), (), (), remaining)
            except InterruptedError:
                continue
            for fd in ready:
                try: chunk = os.read(fd, 1024)
                except InterruptedError: continue
                if chunk:
                    buffers[fd].extend(chunk)
                    cap = _STDOUT_CAP if fd == out_r else _STDERR_CAP
                    if len(buffers[fd]) > cap:
                        raise ValueError("git output cap")
                else:
                    buffers.pop(fd); os.close(fd)
                    if fd == out_r: out_r = -1
                    else: err_r = -1
        while True:
            try:
                waited, status = os.waitpid(pid.value, os.WNOHANG)
            except InterruptedError:
                continue
            if waited == pid.value:
                break
            if waited != 0:
                raise RuntimeError("wrong child reaped")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("git operation deadline")
            time.sleep(min(remaining, 0.01))
        child_reaped = True
        exit_code = os.waitstatus_to_exitcode(status)
        # The bytearrays remain live after their descriptors are removed from
        # ``buffers`` at EOF, so return the exact concurrently drained bytes.
        return {"stdout": bytes(stdout_data), "stderr": bytes(stderr_data), "exitCode": exit_code, "exactChildReaped": True, "remainingOpenFds": ()}
    finally:
        for fd in (out_r, out_w, err_r, err_w):
            if fd >= 0:
                try: os.close(fd)
                except OSError: pass
        if pid.value and not child_reaped:
            cleanup_started_ns = time.monotonic_ns()
            term_deadline_ns = cleanup_started_ns + _TERMINATION_GRACE_NS
            cleanup_deadline_ns = term_deadline_ns + _TERMINATION_GRACE_NS
            while True:
                try:
                    os.kill(pid.value, signal.SIGTERM)
                    break
                except InterruptedError:
                    continue
                except ProcessLookupError:
                    break
            alive = True
            while alive and time.monotonic_ns() < term_deadline_ns:
                try:
                    waited, _ = os.waitpid(pid.value, os.WNOHANG)
                except InterruptedError:
                    continue
                except ChildProcessError:
                    alive = False
                    break
                if waited == pid.value:
                    alive = False
                    child_reaped = True
                    break
                if waited != 0:
                    break
                time.sleep(min(max(term_deadline_ns - time.monotonic_ns(), 0) / 1_000_000_000, 0.01))
            if alive:
                while True:
                    try:
                        os.kill(pid.value, signal.SIGKILL)
                        break
                    except InterruptedError:
                        continue
                    except ProcessLookupError:
                        alive = False
                        break
            while alive and time.monotonic_ns() < cleanup_deadline_ns:
                try:
                    waited, _ = os.waitpid(pid.value, os.WNOHANG)
                except InterruptedError:
                    continue
                except ChildProcessError:
                    break
                if waited == pid.value:
                    child_reaped = True
                    break
                if waited != 0:
                    break
                time.sleep(min(max(cleanup_deadline_ns - time.monotonic_ns(), 0) / 1_000_000_000, 0.01))
        if initialized:
            destroy(actions)


def observe_repository_identity(request: dict) -> dict:
    """Observe only primitive Darwin facts; no marker creation or activation."""
    if sys.platform != "darwin":
        return {"platformSupported": False, "firstFault": "PLATFORM_UNSUPPORTED"}
    # Full enrollment remains fail-closed: this source slice never creates markers.
    return {"platformSupported": True, "firstFault": "MARKER_MISSING", "identityDrift": False}


def _run_native_abi_probe_for_test() -> dict[str, dict]:
    candidate = Path("/Users/ykliu/.hermes/profiles/dev/artifacts/repo-governance/2026-08-11-repo-governance-d1-3-i3-candidate")
    expected = json.loads((candidate / "native-abi-vectors.v2.json").read_text())["expectedIds"]
    completed = subprocess.run([sys.executable, str(candidate / "native-abi-probe.py")], env={"PYTHONDONTWRITEBYTECODE": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    observed = json.loads(completed.stdout)["rows"]
    rows = {}
    for row in observed:
        cleanup = row["cleanup"]
        rows[row["id"]] = {"passed": cleanup["fdsCreated"] == cleanup["fdsClosed"] and cleanup["childrenCreated"] == cleanup["childrenReaped"] and cleanup["childOwnedEnd"] is False, "residue": cleanup["residue"], "primitive": row["primitive"]}
    if set(rows) != set(expected):
        raise ValueError("native ABI row registry mismatch")
    return rows
