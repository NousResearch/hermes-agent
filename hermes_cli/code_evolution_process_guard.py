"""Process-tree supervisor for frozen code-evolution verification.

This module is launched from the trusted Hermes checkout.  It keeps a parent
alive while the frozen verifier runs, records descendants, and terminates any
that survive the verifier.  Linux subreaper mode closes the normal orphaning
race when a descendant creates a new session before its parent exits.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from math import isfinite

import psutil

_CLEANUP_FAILURE_EXIT = 125
_PR_SET_CHILD_SUBREAPER = 36


def _strict_containment_supported(platform: str) -> bool:
    """Return whether this POSIX supervisor has a kernel orphan-reaper."""

    return platform.startswith("linux")


def _enable_linux_subreaper(platform: str) -> None:
    if not _strict_containment_supported(platform):
        raise OSError(
            errno.ENOTSUP,
            "strict verifier process containment is unavailable on " + platform,
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _capture_descendants(
    parent: psutil.Process,
    tracked: dict[tuple[int, float], psutil.Process],
) -> None:
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    except (psutil.AccessDenied, OSError) as exc:
        raise OSError(f"could not inspect verifier descendants: {exc}") from exc
    for child in children:
        try:
            tracked[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            continue


def _is_running(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _terminate_tracked_processes(
    processes: Sequence[psutil.Process],
    *,
    deadline: float,
    label: str,
) -> str | None:
    errors: list[str] = []
    live: list[psutil.Process] = []
    seen: set[tuple[int, float]] = set()
    for process in processes:
        try:
            identity = (process.pid, process.create_time())
            if identity not in seen and _is_running(process):
                seen.add(identity)
                live.append(process)
        except psutil.NoSuchProcess:
            continue
        except (AttributeError, psutil.AccessDenied, OSError):
            identity = (process.pid, float(id(process)))
            if identity not in seen and process.is_running():
                seen.add(identity)
                live.append(process)
    if not live:
        return None

    signalled: list[psutil.Process] = []
    for process in live:
        if time.monotonic() >= deadline:
            errors.append(f"deadline expired before terminating {label} {process.pid}")
            continue
        try:
            process.kill()
            signalled.append(process)
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError) as exc:
            errors.append(f"could not terminate {label} {process.pid}: {exc}")

    survivors: list[psutil.Process] = []
    if signalled:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            survivors = [process for process in signalled if _is_running(process)]
            if survivors:
                errors.append(f"deadline expired while verifying {label} termination")
        else:
            try:
                _, survivors = psutil.wait_procs(signalled, timeout=remaining)
            except (psutil.Error, OSError) as exc:
                survivors = [process for process in signalled if _is_running(process)]
                errors.append(f"could not verify {label} termination: {exc}")
    survivors = [process for process in survivors if _is_running(process)]
    if survivors:
        errors.append(
            f"{label} survived cleanup: "
            + ", ".join(str(process.pid) for process in survivors)
        )
    return "; ".join(errors) if errors else None


def _terminate_descendants(
    supervisor: psutil.Process,
    tracked: dict[tuple[int, float], psutil.Process],
    *,
    deadline: float,
) -> str | None:
    while True:
        if time.monotonic() >= deadline:
            try:
                _capture_descendants(supervisor, tracked)
            except OSError as exc:
                return str(exc)
            survivors = [
                process for process in tracked.values() if _is_running(process)
            ]
            if survivors:
                return "deadline expired with verifier descendants alive: " + ", ".join(
                    str(process.pid) for process in survivors
                )
            return None
        try:
            _capture_descendants(supervisor, tracked)
        except OSError as exc:
            return str(exc)
        live = [process for process in tracked.values() if _is_running(process)]
        if not live:
            return None
        cleanup_error = _terminate_tracked_processes(
            live,
            deadline=deadline,
            label="verifier descendant",
        )
        if cleanup_error:
            return cleanup_error


def run_guarded(
    argv: Sequence[str],
    *,
    platform: str | None = None,
    cleanup_timeout: float = 0.75,
) -> int:
    if not argv:
        print("process guard requires a verifier command", file=sys.stderr)
        return _CLEANUP_FAILURE_EXIT
    active_platform = sys.platform if platform is None else platform
    if not isfinite(cleanup_timeout) or cleanup_timeout <= 0:
        print("process guard cleanup timeout must be positive", file=sys.stderr)
        return _CLEANUP_FAILURE_EXIT
    try:
        _enable_linux_subreaper(active_platform)
    except OSError as exc:
        print(f"could not enable verifier process guard: {exc}", file=sys.stderr)
        return _CLEANUP_FAILURE_EXIT

    cleanup_requested = False

    def request_cleanup(_signum, _frame) -> None:
        nonlocal cleanup_requested
        cleanup_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_cleanup)
    try:
        verifier = subprocess.Popen(list(argv), start_new_session=True)
    except OSError as exc:
        signal.signal(signal.SIGTERM, previous_handler)
        print(f"could not start frozen verifier: {exc}", file=sys.stderr)
        return _CLEANUP_FAILURE_EXIT
    supervisor = psutil.Process(os.getpid())
    tracked: dict[tuple[int, float], psutil.Process] = {}
    try:
        monitor_error: str | None = None
        try:
            while verifier.poll() is None and not cleanup_requested:
                _capture_descendants(supervisor, tracked)
                time.sleep(0.005)
        except (OSError, psutil.Error, KeyboardInterrupt) as exc:
            monitor_error = f"verifier process guard failed: {exc}"
            cleanup_requested = True
        verifier_returncode = verifier.poll()
        deadline = time.monotonic() + cleanup_timeout
        errors = [monitor_error] if monitor_error else []
        if verifier_returncode is None:
            try:
                os.killpg(verifier.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(f"could not terminate verifier process group: {exc}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append(
                    "deadline expired before verifier termination was verified"
                )
            else:
                try:
                    verifier.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    errors.append("frozen verifier survived process-guard cleanup")
        cleanup_error = _terminate_descendants(
            supervisor,
            tracked,
            deadline=deadline,
        )
        if cleanup_error:
            errors.append(cleanup_error)
        if errors:
            print("; ".join(errors), file=sys.stderr)
            return _CLEANUP_FAILURE_EXIT
        if cleanup_requested:
            return _CLEANUP_FAILURE_EXIT
        return int(verifier.returncode or 0)
    except (OSError, psutil.Error) as exc:
        print(f"verifier process guard failed: {exc}", file=sys.stderr)
        return _CLEANUP_FAILURE_EXIT
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-timeout", type=float, default=0.75)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_guarded(command, cleanup_timeout=args.cleanup_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
