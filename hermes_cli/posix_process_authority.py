"""Retained POSIX execution-scope authority for Desktop-owned backends.

Electron keeps a supervisor as its retained ``ChildProcess``. The real backend
runs as a fresh session leader, descendants default to that contained scope,
nested child controllers retain narrower owned scopes, and deliberate escapes
require a completed child-side handoff to a receiving owner. Persisted PIDs and
caller-authored receipt strings never become kill authority.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import Mapping

from hermes_cli import _posix_process_authority_state as S
from hermes_cli._posix_process_authority_state import (
    AuthoritySpec,
    InstalledPosixAuthority,
    ProcessAuthorityError,
    ProcessTransferGrant,
)
from hermes_cli._posix_process_guard import (
    install_descendant_guard as _install_descendant_guard,
    install_posix_descendant_guard,
    reset_guard_for_tests as _reset_guard_for_tests,
)
from hermes_cli._posix_process_transfer import (
    begin_process_transfer,
    desktop_child_env,
)

AUTHORITY_MODE = S.AUTHORITY_MODE
AUTHORITY_MODE_ENV = S.AUTHORITY_MODE_ENV
GENERATION_ENV = S.GENERATION_ENV
PARENT_PID_ENV = S.PARENT_PID_ENV
PARENT_STARTED_AT_ENV = S.PARENT_STARTED_AT_ENV
ROLE_ENV = S.ROLE_ENV
LIFETIME_ENV = S.LIFETIME_ENV
TRANSFER_RECEIPT_ENV = S.TRANSFER_RECEIPT_ENV
TRANSFER_TOKEN_ENV = S.TRANSFER_TOKEN_ENV
TRANSFER_RECEIVER_ENV = S.TRANSFER_RECEIVER_ENV
DESCENDANT_GUARD_ENV = S.DESCENDANT_GUARD_ENV
LIFETIME_CONTAINED = S.LIFETIME_CONTAINED
LIFETIME_TRANSFERRED = S.LIFETIME_TRANSFERRED
LIFETIME_FOREIGN = S.LIFETIME_FOREIGN


_SIGNAL_SELF = getattr(os, "kill", None)


def _signal_group(pgid: int, sig: int) -> None:
    if S.original_killpg is None:
        raise ProcessAuthorityError("POSIX process authority requires killpg()")
    try:
        S.original_killpg(pgid, sig)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise ProcessAuthorityError(
            f"permission denied signalling owned process group: {exc}"
        ) from exc


def _scope_exists(pgid: int) -> bool:
    if S.original_killpg is None:
        return False
    try:
        S.original_killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_child(pid: int, timeout: float) -> int | None:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            waited, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return 0
        if waited == pid:
            return status
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _wait_scope_empty(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(timeout, 0.0)
    while _scope_exists(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _exit_from_status(status: int) -> None:
    if os.WIFEXITED(status):
        os._exit(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        try:
            signal.signal(sig, signal.SIG_DFL)
            if _SIGNAL_SELF is None:
                raise OSError("POSIX self-signal is unavailable")
            _SIGNAL_SELF(os.getpid(), sig)
        except (OSError, ValueError):
            os._exit(128 + sig)
    os._exit(1)


def _drain_residue(pgid: int) -> None:
    """Let retained nested owners clean their subgroups before final force."""

    if not _scope_exists(pgid):
        return
    _signal_group(pgid, signal.SIGTERM)
    if _wait_scope_empty(pgid, S.DEFAULT_GRACE_SECONDS):
        return
    _signal_group(pgid, S.SIGKILL)
    _wait_scope_empty(pgid, S.DEFAULT_FORCE_SECONDS)


def _stop_scope(child_pid: int, *, force: bool) -> int:
    if force:
        _signal_group(child_pid, S.SIGKILL)
        status = _wait_child(child_pid, S.DEFAULT_FORCE_SECONDS)
        _wait_scope_empty(child_pid, S.DEFAULT_FORCE_SECONDS)
    else:
        _signal_group(child_pid, signal.SIGTERM)
        status = _wait_child(child_pid, S.DEFAULT_GRACE_SECONDS)
        if status is not None:
            # The backend may have exited while nested retained owners are
            # still draining their exact child groups. Do not kill those
            # owners immediately; give the root group the rest of its grace.
            if _wait_scope_empty(child_pid, S.DEFAULT_GRACE_SECONDS):
                return status
        _signal_group(child_pid, S.SIGKILL)
        if status is None:
            status = _wait_child(child_pid, S.DEFAULT_FORCE_SECONDS)
        _wait_scope_empty(child_pid, S.DEFAULT_FORCE_SECONDS)

    if status is None:
        status = _wait_child(child_pid, S.DEFAULT_FORCE_SECONDS)
    return 1 if status is None else status


def _run_supervisor(spec: AuthoritySpec, child_pid: int) -> None:
    requested: list[str] = []

    def request_graceful(_sig, _frame) -> None:
        requested[:] = ["graceful"]

    def request_force(_sig, _frame) -> None:
        requested[:] = ["force"]

    signal.signal(signal.SIGTERM, request_graceful)
    signal.signal(signal.SIGINT, request_graceful)
    for force_signal in (S.SIGHUP, S.SIGUSR2):
        if force_signal is not None:
            signal.signal(force_signal, request_force)

    while True:
        try:
            waited, status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            waited, status = child_pid, 0

        if waited == child_pid:
            # Natural backend exit does not imply nested owners are gone.
            # Drain the retained root group first; those owners then terminate
            # their private subgroups before this supervisor relinquishes it.
            _drain_residue(child_pid)
            _exit_from_status(status)

        if os.getppid() != spec.parent_pid:
            status = _stop_scope(child_pid, force=True)
            _exit_from_status(status)

        if requested:
            status = _stop_scope(child_pid, force=requested[-1] == "force")
            _exit_from_status(status)

        time.sleep(0.05)


def install_posix_process_authority(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> InstalledPosixAuthority | None:
    """Install marked authority; leave unmarked processes untouched."""

    actual_platform = sys.platform if platform is None else platform
    source = os.environ if environ is None else environ
    spec = S.read_spec(source)
    if spec is None:
        return None
    if actual_platform == "win32":
        raise ProcessAuthorityError("POSIX process authority requested on Windows")

    with S.install_lock:
        if S.installed is not None:
            if S.installed.spec == spec:
                return S.installed
            raise ProcessAuthorityError(
                "a different Desktop generation already owns this process"
            )

    if os.getppid() != spec.parent_pid:
        raise ProcessAuthorityError(
            "Desktop parent mismatch before POSIX authority installation: "
            f"expected {spec.parent_pid}, observed {os.getppid()}"
        )

    fork = getattr(os, "fork", None)
    if fork is None:
        raise ProcessAuthorityError("POSIX process authority requires fork()")
    child_pid = fork()
    if child_pid == 0:
        try:
            if S.original_setsid is None:
                raise ProcessAuthorityError(
                    "POSIX process authority requires setsid()"
                )
            pgid = S.original_setsid()
            if pgid is None:
                pgid = os.getpgrp()
        except Exception:
            os._exit(126)
        os.environ[ROLE_ENV] = "posix-backend"
        installed = InstalledPosixAuthority(
            spec,
            "posix-backend",
            int(pgid),
        )
        with S.install_lock:
            S.installed = installed
        _install_descendant_guard()
        return installed

    os.environ[ROLE_ENV] = "posix-supervisor"
    _run_supervisor(spec, child_pid)
    raise AssertionError("POSIX process supervisor returned unexpectedly")


def _reset_process_authority_for_tests() -> None:
    with S.install_lock:
        S.installed = None
    _reset_guard_for_tests()
    with S.transfer_lock:
        S.pending_transfers.clear()
