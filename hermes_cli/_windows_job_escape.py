"""Detach a child from a Windows terminal command's membership job (stdlib only).

``LocalEnvironment._run_bash`` contains each Git Bash wrapper in a membership-only Job
Object WITHOUT ``JOB_OBJECT_LIMIT_BREAKAWAY_OK``: MSYS fork/exec itself requests
``CREATE_BREAKAWAY_FROM_JOB``, so allowing breakaway would let the MSYS-handoff orphans
the job exists to catch escape it. The price is that Hermes' own intentional detaches
(gateway start/restart watcher, post-update relaunch) get ERROR_ACCESS_DENIED for
``CREATE_BREAKAWAY_FROM_JOB`` when run from inside a terminal command, and would die with
the job's final ``TerminateJobObject`` sweep.

(A launcher-shim python.exe puts the caller in a nested job, where breakaway can "succeed"
yet still land in the terminal job, so a retry-on-denied design is not enough.)

A job flag cannot tell Hermes' breakaway from MSYS's, so the detach is routed through a
process that is not a member instead: ``_run_bash`` publishes its own owner (the Hermes
process that created the job, never a member) as an *anchor* in the command's environment,
and a short-lived stdlib trampoline spawns with ``PROC_THREAD_ATTRIBUTE_PARENT_PROCESS`` set
to that anchor. The kernel takes job membership from the nominated parent. The trampoline
starts the real child and exits before we return, so the child is outside both the terminal
job AND the anchor's live process tree (Desktop quit / gateway force-stop use tree kills).
Ordinary children never take this path: only the explicit Hermes detach call sites use
:func:`popen_detached`, and plain MSYS fork/exec is untouched.

Identity: the anchor is a (pid, creation-time) pair verified on the opened handle, never a
bare PID, name or argv. Imported by the update watcher on a bare interpreter, so this
module must stay stdlib-only.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from ctypes import wintypes
from typing import NamedTuple

ANCHOR_ENV = "HERMES_TERMINAL_JOB_ANCHOR"

_STILL_ACTIVE = 259
_PROCESS_DUP_HANDLE = 0x0040
_PROCESS_CREATE_PROCESS = 0x0080
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_STARTF_USESTDHANDLES = 0x00000100
_DUPLICATE_CLOSE_SOURCE = 0x1
_DUPLICATE_SAME_ACCESS = 0x2
_ATTR_PARENT_PROCESS = 0x00020000
_ATTR_HANDLE_LIST = 0x00020002


class DetachedProcess(NamedTuple):
    """What every detach call site needs from a spawn: the child's PID."""
    pid: int


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


class _Filetime(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


def _kernel32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(_Filetime)] * 4
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.DuplicateHandle.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)]
    k32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_void_p]
    k32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    k32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL,
        wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(_StartupInfoEx),
        ctypes.POINTER(_ProcessInformation)]
    return k32


def _creation_time(k32, handle) -> int:
    created, other = _Filetime(), _Filetime()
    if not k32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(other),
                               ctypes.byref(other), ctypes.byref(other)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (created.hi << 32) | created.lo


def current_anchor() -> str:
    """This process as ``pid:creation-time``: what a job's members are told to detach through.

    Call only from the process that CREATES the job — a member would anchor on the job."""
    k32 = _kernel32()
    return f"{os.getpid()}:{_creation_time(k32, k32.GetCurrentProcess())}"


def _open_anchor(k32, anchor: str):
    """Handle to the live process ``anchor`` names, or None if it is gone or was recycled."""
    try:
        pid_text, created_text = anchor.split(":")
        pid, created = int(pid_text), int(created_text)
    except ValueError:
        return None
    handle = k32.OpenProcess(_PROCESS_CREATE_PROCESS | _PROCESS_DUP_HANDLE
                             | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    code = wintypes.DWORD()
    try:
        if (_creation_time(k32, handle) == created and k32.GetExitCodeProcess(handle, ctypes.byref(code))
                and code.value == _STILL_ACTIVE):
            return handle
    except OSError:
        pass
    k32.CloseHandle(handle)
    return None


def _env_block(env: Mapping[str, str]) -> ctypes.Array:
    # Windows names are case-insensitive; keep the last spelling, like Popen.
    unique = {key.upper(): (key, value) for key, value in env.items()}
    text = "".join(f"{key}={value}\0" for _, (key, value) in sorted(unique.items())) + "\0"
    return ctypes.create_unicode_buffer(text, len(text))


def _std_handle(spec, devnull):
    """OS handle for a detach stdio spec: a file object, or NUL for DEVNULL/None (a detached
    child never wants its caller's console)."""
    import msvcrt
    if spec is None or spec == subprocess.DEVNULL:
        return msvcrt.get_osfhandle(devnull.fileno())
    return msvcrt.get_osfhandle(spec.fileno())


def _spawn_with_parent(k32, parent, argv, cwd, env, flags, std_handles, extra_handles=()):
    """CreateProcessW under ``parent`` with exactly ``std_handles`` (already valid in it)."""
    size = ctypes.c_size_t()
    k32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
    attrs = ctypes.create_string_buffer(size.value)
    if not k32.InitializeProcThreadAttributeList(attrs, 2, 0, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        parent_value = wintypes.HANDLE(parent)
        handles = [*std_handles, *extra_handles]
        listed = (wintypes.HANDLE * len(handles))(*handles)
        for attribute, value, value_size in (
                (_ATTR_PARENT_PROCESS, ctypes.byref(parent_value), ctypes.sizeof(parent_value)),
                (_ATTR_HANDLE_LIST, listed, ctypes.sizeof(listed))):
            if not k32.UpdateProcThreadAttribute(attrs, 0, attribute, value, value_size, None, None):
                raise ctypes.WinError(ctypes.get_last_error())
        startup = _StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = std_handles
        startup.lpAttributeList = ctypes.addressof(attrs)
        info = _ProcessInformation()
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline([os.fspath(a) for a in argv]))
        if not k32.CreateProcessW(
                None, command, None, None, True,
                flags | _EXTENDED_STARTUPINFO_PRESENT | _CREATE_UNICODE_ENVIRONMENT,
                ctypes.cast(_env_block(env), ctypes.c_void_p), None if cwd is None else os.fspath(cwd),
                ctypes.byref(startup), ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        k32.CloseHandle(info.hThread)
        return info.hProcess
    finally:
        k32.DeleteProcThreadAttributeList(attrs)


def _spawn_via_anchor(anchor, argv, *, cwd, env, creationflags, stdin, stdout, stderr) -> int | None:
    """Start an exiting trampoline outside the terminal job, then return the real child's PID."""
    import msvcrt

    k32 = _kernel32()
    parent = _open_anchor(k32, anchor)
    if parent is None:
        return None
    own = k32.GetCurrentProcess()
    duplicated: list = []
    read_fd, write_fd = os.pipe()
    process = None
    with open(os.devnull, "r+b", buffering=0) as devnull:
        try:
            # Handles come from the NOMINATED parent's table. The fourth handle is a private
            # acknowledgement pipe, not inherited by the real child; no env/argv goes to disk.
            sources = [_std_handle(spec, devnull) for spec in (stdin, stdout, stderr)]
            sources.append(msvcrt.get_osfhandle(write_fd))
            for source in sources:
                copy = wintypes.HANDLE()
                if not k32.DuplicateHandle(own, source, parent, ctypes.byref(copy),
                                           0, True, _DUPLICATE_SAME_ACCESS):
                    raise ctypes.WinError(ctypes.get_last_error())
                duplicated.append(copy.value)
            # The base interpreter has no venv launcher shim to stay alive under the anchor.
            trampoline = [sys._base_executable, "-I", os.path.abspath(__file__),
                          str(duplicated[3]), str(creationflags),
                          json.dumps([os.fspath(a) for a in argv]), json.dumps(duplicated[:3])]
            try:
                process = _spawn_with_parent(k32, parent, trampoline, cwd, env, creationflags,
                                             duplicated[:3], duplicated[3:])
            except OSError as exc:
                if exc.winerror != 5 or not creationflags & subprocess.CREATE_BREAKAWAY_FROM_JOB:
                    raise
                # Parent nomination still supplies membership when an outer job denies breakaway.
                process = _spawn_with_parent(k32, parent, trampoline, cwd, env,
                                             creationflags & ~subprocess.CREATE_BREAKAWAY_FROM_JOB,
                                             duplicated[:3], duplicated[3:])
            # The trampoline now owns its inherited pipe handle. Drop both spare writers so
            # an empty/malformed acknowledgement fails closed instead of blocking on our own fd.
            k32.DuplicateHandle(parent, duplicated.pop(), None, None, 0, False, _DUPLICATE_CLOSE_SOURCE)
            os.close(write_fd)
            write_fd = None
            # Waiting on the opened process HANDLE proves the trampoline is gone before return.
            # A timeout/crash is an uncertain spawn, NOT an OSError eligible for a duplicate retry.
            if k32.WaitForSingleObject(process, 30000) != 0:
                raise RuntimeError("Windows detach trampoline did not exit; spawn outcome unknown")
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(process, ctypes.byref(code)) or code.value != 0:
                raise RuntimeError("Windows detach trampoline failed; spawn outcome unknown")
            try:
                result = json.loads(os.read(read_fd, 1024))
                if not isinstance(result, dict) or not ("error" in result or "pid" in result):
                    raise ValueError("missing child identity")
            except (OSError, ValueError) as exc:
                raise RuntimeError("Windows detach acknowledgement failed; spawn outcome unknown") from exc
            if "error" in result:
                raise ctypes.WinError(result["error"])
            return result["pid"]
        finally:
            if process is not None:
                k32.CloseHandle(process)
            for value in duplicated:
                k32.DuplicateHandle(parent, value, None, None, 0, False, _DUPLICATE_CLOSE_SOURCE)
            k32.CloseHandle(parent)
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)


def _trampoline() -> None:
    """Stdlib-only, short-lived parent. Only the three stdio handles reach the real child."""
    import msvcrt

    ack = msvcrt.open_osfhandle(int(sys.argv[1]), os.O_WRONLY)
    flags, argv = int(sys.argv[2]), json.loads(sys.argv[3])
    # Popen's explicit stdio path duplicates these and builds its own handle list. In particular
    # the acknowledgement handle must not leak to the child, which can live indefinitely.
    with contextlib.ExitStack() as stack:
        reply = stack.enter_context(os.fdopen(ack, "wb", buffering=0))
        # pythonw sets sys.stdin/stdout/stderr to None even with valid OS handles. Wrap the
        # explicitly inherited handles instead, preserving the gateway's sidecar log there too.
        streams = [stack.enter_context(os.fdopen(msvcrt.open_osfhandle(handle, flags), mode,
                                                 buffering=0))
                   for handle, flags, mode in zip(json.loads(sys.argv[4]),
                                                  (os.O_RDONLY, os.O_WRONLY, os.O_WRONLY),
                                                  ("rb", "wb", "wb"))]
        stdio = dict(zip(("stdin", "stdout", "stderr"), streams))
        try:
            try:
                proc = subprocess.Popen(argv, creationflags=flags, close_fds=True, **stdio)
            except OSError:
                if not flags & subprocess.CREATE_BREAKAWAY_FROM_JOB:
                    raise
                # We already left the terminal job. An unrelated outer host job may deny
                # breakaway; the legacy no-breakaway retry is safe HERE, not in the caller.
                proc = subprocess.Popen(argv,
                                        creationflags=flags & ~subprocess.CREATE_BREAKAWAY_FROM_JOB,
                                        close_fds=True, **stdio)
        except OSError as exc:
            reply.write(json.dumps({"error": exc.winerror or 1}).encode())
        else:
            reply.write(json.dumps({"pid": proc.pid}).encode())


def popen_detached(argv: Sequence[str], *, creationflags: int, env: Mapping[str, str] | None = None,
                   cwd=None, stdin=None, stdout=None, stderr=None) -> DetachedProcess:
    """``subprocess.Popen`` for Hermes' intentional breakaway spawns; returns the child's PID.

    Inside a terminal command (its anchor is in the environment and alive) the child is spawned
    via an exiting trampoline under the anchor, so it never joins the command's job and is
    not left in the anchor's live process tree. This is tried FIRST, not only
    after an ERROR_ACCESS_DENIED: a launcher shim (venv ``python.exe``) puts the caller in its own
    nested job, where ``CREATE_BREAKAWAY_FROM_JOB`` succeeds but only leaves that innermost job and
    lands the child in the terminal job with no error to catch. Anywhere else, or when the anchor
    path cannot run, this is a plain ``Popen`` and failures raise exactly as ``Popen`` would, so
    every caller's existing no-breakaway fallback still applies. The anchor is never passed on: a
    detached child is not part of any terminal command."""
    child_env = {key: value for key, value in (os.environ if env is None else env).items()
                 if key != ANCHOR_ENV}
    anchor = os.environ.get(ANCHOR_ENV)
    if sys.platform == "win32" and anchor:
        try:
            pid = _spawn_via_anchor(anchor, argv, cwd=cwd, env=child_env, creationflags=creationflags,
                                    stdin=stdin, stdout=stdout, stderr=stderr)
        except OSError:
            pid = None  # e.g. the anchor's own job forbids breakaway: let the plain path decide
        if pid is not None:
            return DetachedProcess(pid)
    proc = subprocess.Popen(list(argv), creationflags=creationflags, env=child_env, cwd=cwd,
                            stdin=stdin, stdout=stdout, stderr=stderr, close_fds=True)
    return DetachedProcess(getattr(proc, "pid", 0))


if __name__ == "__main__":
    _trampoline()
