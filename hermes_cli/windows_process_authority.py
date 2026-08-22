"""Generation-bound Windows process authority for Desktop-owned backends.

A PID is observation, never destructive authority. The Electron Desktop marks
each backend launch with a fresh generation plus the parent process creation
time. At Python bootstrap, before Hermes can spawn any children, this module:

* creates an unnamed Job Object with ``KILL_ON_JOB_CLOSE``;
* verifies the parent PID against its creation time and retains that process
  HANDLE for the lifetime of the backend;
* assigns the current process to the Job Object; and
* closes the Job Object when the retained parent HANDLE signals.

Normal root shutdown and Desktop crashes both close the last Job Object HANDLE,
so Windows terminates the whole backend tree without a later PID lookup. Any
setup/identity failure aborts the Desktop-owned backend instead of falling back
to ``taskkill`` or a reconstructed PID (#89614).
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

AUTHORITY_MODE_ENV = "HERMES_DESKTOP_PROCESS_AUTHORITY"
AUTHORITY_MODE = "windows-job-v1"
GENERATION_ENV = "HERMES_DESKTOP_PROCESS_GENERATION"
PARENT_PID_ENV = "HERMES_DESKTOP_PARENT_PID"
PARENT_STARTED_AT_ENV = "HERMES_DESKTOP_PARENT_STARTED_AT_MS"

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_INFINITE = 0xFFFFFFFF
_WAIT_OBJECT_0 = 0x00000000
_WINDOWS_EPOCH_100NS = 116444736000000000
_PARENT_START_TOLERANCE_MS = 5_000
_GENERATION_RE = re.compile(r"^[A-Za-z0-9._-]{16,128}$")


class ProcessAuthorityError(RuntimeError):
    """Desktop process authority could not be established safely."""


@dataclass(frozen=True)
class AuthoritySpec:
    generation: str
    parent_pid: int
    parent_started_at_ms: int


@dataclass
class InstalledAuthority:
    spec: AuthoritySpec
    api: "WindowsAuthorityApi"
    job_handle: Any
    parent_handle: Any
    watcher: Any


class WindowsAuthorityApi(Protocol):
    def create_job(self) -> Any: ...

    def enable_kill_on_close(self, job_handle: Any) -> None: ...

    def current_process_handle(self) -> Any: ...

    def assign_current_process(self, job_handle: Any, process_handle: Any) -> None: ...

    def open_parent(self, pid: int) -> Any: ...

    def process_started_at_ms(self, process_handle: Any) -> int: ...

    def wait_for_process_exit(self, process_handle: Any) -> None: ...

    def close_handle(self, handle: Any) -> None: ...


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _Kernel32AuthorityApi:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise ProcessAuthorityError("Windows process authority requested off Windows")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32 = kernel32

        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

    @staticmethod
    def _raise_last_error(operation: str) -> None:
        code = ctypes.get_last_error()
        raise ProcessAuthorityError(f"{operation} failed: {ctypes.WinError(code)}")

    def create_job(self) -> Any:
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            self._raise_last_error("CreateJobObjectW")
        return handle

    def enable_kill_on_close(self, job_handle: Any) -> None:
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = self._kernel32.SetInformationJobObject(
            job_handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            self._raise_last_error("SetInformationJobObject")

    def current_process_handle(self) -> Any:
        return self._kernel32.GetCurrentProcess()

    def assign_current_process(self, job_handle: Any, process_handle: Any) -> None:
        if not self._kernel32.AssignProcessToJobObject(job_handle, process_handle):
            self._raise_last_error("AssignProcessToJobObject")

    def open_parent(self, pid: int) -> Any:
        handle = self._kernel32.OpenProcess(
            _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not handle:
            self._raise_last_error("OpenProcess(parent)")
        return handle

    def process_started_at_ms(self, process_handle: Any) -> int:
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not self._kernel32.GetProcessTimes(
            process_handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            self._raise_last_error("GetProcessTimes(parent)")
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return int((ticks - _WINDOWS_EPOCH_100NS) / 10_000)

    def wait_for_process_exit(self, process_handle: Any) -> None:
        result = self._kernel32.WaitForSingleObject(process_handle, _INFINITE)
        if result != _WAIT_OBJECT_0:
            raise ProcessAuthorityError(
                "WaitForSingleObject(parent) returned unexpected status "
                f"0x{result:08x}"
            )

    def close_handle(self, handle: Any) -> None:
        if handle:
            self._kernel32.CloseHandle(handle)


_install_lock = threading.Lock()
_installed: InstalledAuthority | None = None


def _read_spec(environ: Mapping[str, str]) -> AuthoritySpec | None:
    mode = (environ.get(AUTHORITY_MODE_ENV) or "").strip()
    if not mode:
        return None
    if mode != AUTHORITY_MODE:
        raise ProcessAuthorityError(
            f"unsupported desktop process authority mode: {mode!r}"
        )

    generation = (environ.get(GENERATION_ENV) or "").strip()
    if not _GENERATION_RE.fullmatch(generation):
        raise ProcessAuthorityError("desktop process generation is missing or malformed")

    try:
        parent_pid = int((environ.get(PARENT_PID_ENV) or "").strip())
        parent_started_at_ms = int(
            (environ.get(PARENT_STARTED_AT_ENV) or "").strip()
        )
    except ValueError as exc:
        raise ProcessAuthorityError("desktop parent identity is malformed") from exc

    if parent_pid <= 0 or parent_started_at_ms <= 0:
        raise ProcessAuthorityError("desktop parent identity must be positive")

    return AuthoritySpec(
        generation=generation,
        parent_pid=parent_pid,
        parent_started_at_ms=parent_started_at_ms,
    )


def install_windows_process_authority(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    api: WindowsAuthorityApi | None = None,
    thread_factory: Callable[..., Any] = threading.Thread,
    parent_start_tolerance_ms: int = _PARENT_START_TOLERANCE_MS,
) -> InstalledAuthority | None:
    """Install and retain the Desktop-owned Windows Job Object authority.

    Returns ``None`` when the process was not launched under the Desktop
    authority envelope or when running off Windows. Once the envelope is
    present on Windows, every validation/setup failure raises
    :class:`ProcessAuthorityError`; there is deliberately no PID fallback.
    """

    global _installed

    actual_platform = sys.platform if platform is None else platform
    source = os.environ if environ is None else environ
    spec = _read_spec(source)
    if spec is None or actual_platform != "win32":
        return None

    if parent_start_tolerance_ms < 0:
        raise ValueError("parent_start_tolerance_ms must be non-negative")

    with _install_lock:
        if _installed is not None:
            if _installed.spec == spec:
                return _installed
            raise ProcessAuthorityError(
                "a different desktop process generation already owns this process"
            )

        winapi = api or _Kernel32AuthorityApi()
        job_handle = None
        parent_handle = None
        try:
            job_handle = winapi.create_job()
            winapi.enable_kill_on_close(job_handle)

            # Bind the parent PID to the creation marker passed at spawn time
            # before granting it lifecycle authority. A recycled PID cannot
            # satisfy both values.
            parent_handle = winapi.open_parent(spec.parent_pid)
            observed_start = winapi.process_started_at_ms(parent_handle)
            if (
                abs(observed_start - spec.parent_started_at_ms)
                > parent_start_tolerance_ms
            ):
                raise ProcessAuthorityError(
                    "desktop parent process generation mismatch: "
                    f"expected {spec.parent_started_at_ms}, "
                    f"observed {observed_start}"
                )

            # The current-process pseudo-handle is already generation-bound;
            # no PID lookup occurs here.
            winapi.assign_current_process(
                job_handle,
                winapi.current_process_handle(),
            )

            def _watch_parent() -> None:
                try:
                    winapi.wait_for_process_exit(parent_handle)
                finally:
                    # Closing the last job handle is the tree-wide teardown
                    # primitive. It terminates this root and every descendant
                    # still in the job without resolving any PID.
                    winapi.close_handle(parent_handle)
                    winapi.close_handle(job_handle)

            watcher = thread_factory(
                target=_watch_parent,
                name="hermes-desktop-parent-authority",
                daemon=True,
            )
            installed = InstalledAuthority(
                spec=spec,
                api=winapi,
                job_handle=job_handle,
                parent_handle=parent_handle,
                watcher=watcher,
            )
            _installed = installed
            try:
                watcher.start()
            except Exception:
                _installed = None
                raise
            return installed
        except Exception:
            if parent_handle is not None:
                winapi.close_handle(parent_handle)
            if job_handle is not None:
                winapi.close_handle(job_handle)
            raise


def _reset_process_authority_for_tests() -> None:
    """Clear the process-global installation latch for isolated unit tests."""

    global _installed
    with _install_lock:
        _installed = None
