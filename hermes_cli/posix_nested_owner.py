"""Retained owner for a nested POSIX child-control scope."""

from __future__ import annotations

import base64
import json
import os
import select
import signal
import sys
import time
from typing import Any

from hermes_cli import _posix_process_authority_state as S

_PROTOCOL = "desktop-posix-nested-v1"
_EXEC_ACK_SECONDS = 3.0
_GRACE_SECONDS = 2.0
_FORCE_SECONDS = 1.0
_MAX_ACK_BYTES = 4096
_FORK = getattr(os, "fork", None)


class NestedOwnerError(RuntimeError):
    """A nested owner could not retain or prove its exact child scope."""


def _decode_spec(raw: str) -> dict[str, Any]:
    try:
        spec = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise NestedOwnerError("malformed nested process specification") from exc
    if not isinstance(spec, dict) or spec.get("protocol") != _PROTOCOL:
        raise NestedOwnerError("unsupported nested process protocol")
    if not isinstance(spec.get("ack_fd"), int) or spec["ack_fd"] < 0:
        raise NestedOwnerError("nested process acknowledgement descriptor is invalid")
    if not isinstance(spec.get("parent_pid"), int) or spec["parent_pid"] <= 0:
        raise NestedOwnerError("nested process parent identity is invalid")
    if not isinstance(spec.get("executable"), str) or not spec["executable"]:
        raise NestedOwnerError("nested process executable is missing")
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
        raise NestedOwnerError("nested process argv is malformed")
    pass_fds = spec.get("target_pass_fds", [])
    if not isinstance(pass_fds, list) or not all(isinstance(fd, int) and fd >= 0 for fd in pass_fds):
        raise NestedOwnerError("nested process descriptor list is malformed")
    return spec


def _close(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _write_ack(fd: int, payload: dict[str, Any]) -> bool:
    encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    if len(encoded) > _MAX_ACK_BYTES:
        return False
    try:
        os.write(fd, encoded)
        return True
    except (BrokenPipeError, OSError):
        return False


def _error(kind: str, detail: str) -> dict[str, Any]:
    return {"type": kind, "detail": detail}


def _kill_scope(scope_id: int, sig: int) -> None:
    if S.original_killpg is None:
        raise NestedOwnerError("nested owner requires process-group signalling")
    try:
        S.original_killpg(scope_id, sig)
    except (ProcessLookupError, PermissionError):
        pass


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


def _stop_scope(child_pid: int, *, force: bool) -> int | None:
    if force:
        _kill_scope(child_pid, S.SIGKILL)
        return _wait_child(child_pid, _FORCE_SECONDS)
    _kill_scope(child_pid, signal.SIGTERM)
    status = _wait_child(child_pid, _GRACE_SECONDS)
    if status is None:
        _kill_scope(child_pid, S.SIGKILL)
        status = _wait_child(child_pid, _FORCE_SECONDS)
    return status


def _exit_from_status(status: int | None) -> None:
    if status is None:
        os._exit(1)
    if os.WIFEXITED(status):
        os._exit(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        os._exit(128 + os.WTERMSIG(status))
    os._exit(1)


def _install_handlers(requested: list[int]) -> tuple[int, ...]:
    handled: list[int] = []

    def request(sig: int, _frame: Any) -> None:
        requested[:] = [sig]

    for candidate in (signal.SIGTERM, signal.SIGINT, S.SIGHUP, S.SIGUSR2):
        if candidate is None or candidate in handled:
            continue
        signal.signal(candidate, request)
        handled.append(candidate)
    return tuple(handled)


def _reset_handlers(handled: tuple[int, ...]) -> None:
    for sig in handled:
        signal.signal(sig, signal.SIG_DFL)


def _run_target(
    spec: dict[str, Any],
    status_read: int,
    status_write: int,
    handled: tuple[int, ...],
) -> None:
    _close(status_read)
    _reset_handlers(handled)
    try:
        if S.original_setsid is None:
            raise NestedOwnerError("nested owner requires setsid()")
        S.original_setsid()
        for fd in spec.get("target_pass_fds", []):
            os.set_inheritable(fd, True)
        os.execvpe(spec["executable"], list(spec["argv"]), dict(os.environ))
    except BaseException as exc:
        payload = _error(type(exc).__name__, str(exc))
        try:
            os.write(
                status_write,
                json.dumps(payload, separators=(",", ":")).encode("utf-8")[:_MAX_ACK_BYTES],
            )
        except OSError:
            pass
        finally:
            os._exit(127)


def _read_exec_status(
    fd: int,
    *,
    parent_pid: int,
    requested: list[int],
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _EXEC_ACK_SECONDS
    while True:
        if requested:
            return _error("NestedOwnerCancelled", "owner stopped before acknowledgement")
        if os.getppid() != parent_pid:
            return _error("ParentExited", "nested parent exited before acknowledgement")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _error("ExecTimeout", "nested target did not cross exec before timeout")
        ready, _, _ = select.select([fd], [], [], min(remaining, 0.05))
        if not ready:
            continue
        payload = os.read(fd, _MAX_ACK_BYTES + 1)
        if not payload:
            return None
        if len(payload) > _MAX_ACK_BYTES:
            return _error("ExecError", "nested target exec failure was oversized")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _error("ExecError", "nested target exec failure was malformed")
        return decoded if isinstance(decoded, dict) else _error(
            "ExecError", "nested target exec failure was malformed"
        )


def _target_identity(child_pid: int) -> tuple[int, int]:
    try:
        sid = int(os.getsid(child_pid))
        pgid = int(os.getpgid(child_pid))
    except (ProcessLookupError, PermissionError) as exc:
        raise NestedOwnerError("nested target exited before acknowledgement") from exc
    if sid != child_pid or pgid != child_pid:
        raise NestedOwnerError("nested target did not enter its exact private scope")
    return sid, pgid


def _supervise(child_pid: int, parent_pid: int, requested: list[int]) -> None:
    force_signals = {sig for sig in (S.SIGHUP, S.SIGUSR2) if sig is not None}
    while True:
        try:
            waited, status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            waited, status = child_pid, 0
        if waited == child_pid:
            _kill_scope(child_pid, S.SIGKILL)
            _exit_from_status(status)
        if os.getppid() != parent_pid:
            status = _stop_scope(child_pid, force=True)
            _kill_scope(child_pid, S.SIGKILL)
            _exit_from_status(status)
        if requested:
            status = _stop_scope(child_pid, force=requested[-1] in force_signals)
            _kill_scope(child_pid, S.SIGKILL)
            _exit_from_status(status)
        time.sleep(0.05)


def main() -> None:
    if len(sys.argv) < 2:
        raise NestedOwnerError("nested process specification is missing")
    spec = _decode_spec(sys.argv[1])
    if _FORK is None or S.original_killpg is None:
        raise NestedOwnerError("nested owner requires POSIX process primitives")

    ack_fd = int(spec["ack_fd"])
    os.set_inheritable(ack_fd, False)
    requested: list[int] = []
    handled = _install_handlers(requested)

    status_read, status_write = os.pipe()
    os.set_inheritable(status_write, False)
    child_pid = _FORK()
    if child_pid == 0:
        _run_target(spec, status_read, status_write, handled)
        raise AssertionError("nested target returned from exec")

    _close(status_write)
    for fd in spec.get("target_pass_fds", []):
        if fd > 2 and fd != ack_fd:
            _close(fd)

    exec_error = _read_exec_status(
        status_read,
        parent_pid=int(spec["parent_pid"]),
        requested=requested,
    )
    _close(status_read)

    ack: dict[str, Any] = {
        "protocol": _PROTOCOL,
        "owner_pid": os.getpid(),
        "owner_sid": os.getsid(0),
        "owner_pgid": os.getpgrp(),
        "scope_id": child_pid,
    }
    if exec_error is None:
        try:
            sid, pgid = _target_identity(child_pid)
            ack["target_sid"] = sid
            ack["target_pgid"] = pgid
        except NestedOwnerError as exc:
            exec_error = _error(type(exc).__name__, str(exc))
    if exec_error is not None:
        ack["error"] = exec_error

    acknowledged = _write_ack(ack_fd, ack)
    _close(ack_fd)
    if exec_error is not None or not acknowledged:
        _stop_scope(child_pid, force=True)
        os._exit(127 if exec_error is not None else 126)

    _supervise(child_pid, int(spec["parent_pid"]), requested)


if __name__ == "__main__":
    main()
