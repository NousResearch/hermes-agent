"""Receiving owner for a completed POSIX process-authority transfer.

The old Desktop-owned backend launches this module as a fresh session leader.
This owner starts the transferred target in a dedicated process group, proves
that the target successfully crossed ``exec``, acknowledges the exact handoff,
and then remains alive to forward lifecycle commands and reap residue.
"""

from __future__ import annotations

import base64
import json
import os
import select
import signal
import sys
import time
from typing import Any

from tools.environments.local import build_subprocess_env

_PROTOCOL = "desktop-posix-transfer-v1"
_EXEC_ACK_SECONDS = 3.0
_GRACE_SECONDS = 5.0
_FORCE_SECONDS = 2.0
_MAX_ACK_BYTES = 4096
_DESCENDANT_GUARD_ENV = "_HERMES_DESKTOP_POSIX_DESCENDANT_GUARD"
_AUTHORITY_MODE = "posix-session-v1"
_LIFETIME_ENV = "HERMES_DESKTOP_PROCESS_LIFETIME"

_FORK = getattr(os, "fork", None)
_SET_PGID = getattr(os, "setpgid", None)
_KILL_GROUP = getattr(os, "killpg", None)
_GET_SID = getattr(os, "getsid", None)
_GET_PGID = getattr(os, "getpgid", None)
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
_SIGHUP = getattr(signal, "SIGHUP", None)
_SIGUSR2 = getattr(signal, "SIGUSR2", None)


class TransferOwnerError(RuntimeError):
    """The receiving owner could not establish or retain its exact scope."""


def _decode_spec(raw: str) -> dict[str, Any]:
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii"))
        spec = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransferOwnerError("malformed process-transfer specification") from exc

    if not isinstance(spec, dict) or spec.get("protocol") != _PROTOCOL:
        raise TransferOwnerError("unsupported process-transfer protocol")
    if not isinstance(spec.get("token"), str) or not spec["token"]:
        raise TransferOwnerError("process-transfer token is missing")
    if not isinstance(spec.get("receiver"), str) or not spec["receiver"]:
        raise TransferOwnerError("process-transfer receiver is missing")
    if not isinstance(spec.get("ack_fd"), int) or spec["ack_fd"] < 0:
        raise TransferOwnerError("process-transfer acknowledgement descriptor is invalid")
    if not isinstance(spec.get("sender_pid"), int) or spec["sender_pid"] <= 0:
        raise TransferOwnerError("process-transfer sender identity is invalid")
    if not isinstance(spec.get("executable"), str) or not spec["executable"]:
        raise TransferOwnerError("transferred executable is missing")
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
        raise TransferOwnerError("transferred argv is malformed")
    pass_fds = spec.get("target_pass_fds", [])
    if not isinstance(pass_fds, list) or not all(isinstance(fd, int) and fd >= 0 for fd in pass_fds):
        raise TransferOwnerError("transferred descriptor list is malformed")
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


def _error(kind: str, detail: str, *, errno: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": kind, "detail": detail}
    if errno is not None:
        payload["errno"] = errno
    return payload


def _kill_scope(scope_id: int, sig: int) -> None:
    if _KILL_GROUP is None:
        raise TransferOwnerError("receiving owner requires process-group signalling")
    try:
        _KILL_GROUP(scope_id, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _wait_child(child_pid: int, timeout: float) -> int | None:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            waited, status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            return 0
        if waited == child_pid:
            return status
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _stop_scope(child_pid: int, *, force: bool) -> int | None:
    if force:
        _kill_scope(child_pid, _SIGKILL)
        status = _wait_child(child_pid, _FORCE_SECONDS)
    else:
        _kill_scope(child_pid, signal.SIGTERM)
        status = _wait_child(child_pid, _GRACE_SECONDS)
        if status is None:
            _kill_scope(child_pid, _SIGKILL)
            status = _wait_child(child_pid, _FORCE_SECONDS)
    if status is None:
        _kill_scope(child_pid, _SIGKILL)
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

    for candidate in (signal.SIGTERM, signal.SIGINT, _SIGHUP, _SIGUSR2):
        if candidate is None or candidate in handled:
            continue
        signal.signal(candidate, request)
        handled.append(candidate)
    return tuple(handled)


def _reset_handlers(handled: tuple[int, ...]) -> None:
    for sig in handled:
        signal.signal(sig, signal.SIG_DFL)


def _read_exec_status(
    fd: int,
    *,
    sender_pid: int,
    requested: list[int],
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _EXEC_ACK_SECONDS
    while True:
        if requested:
            return _error("TransferCancelled", "receiving owner was stopped before acknowledgement")
        if os.getppid() != sender_pid:
            return _error("SenderExited", "transferring owner exited before acknowledgement")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _error("ExecTimeout", "transferred target did not cross exec before timeout")
        ready, _, _ = select.select([fd], [], [], min(remaining, 0.05))
        if not ready:
            continue

        payload = os.read(fd, _MAX_ACK_BYTES + 1)
        if not payload:
            return None
        if len(payload) > _MAX_ACK_BYTES:
            return _error("ExecError", "transferred target exec failure was oversized")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _error("ExecError", "transferred target exec failure was malformed")
        if not isinstance(decoded, dict):
            return _error("ExecError", "transferred target exec failure was malformed")
        return decoded


def _target_identity(child_pid: int, owner_pid: int) -> tuple[int, int]:
    if _GET_SID is None or _GET_PGID is None:
        raise TransferOwnerError("receiving owner requires session and process-group identity")
    try:
        target_sid = int(_GET_SID(child_pid))
        target_pgid = int(_GET_PGID(child_pid))
    except (ProcessLookupError, PermissionError) as exc:
        raise TransferOwnerError("transferred target exited before acknowledgement") from exc
    if target_sid != owner_pid or target_pgid != child_pid:
        raise TransferOwnerError("transferred target did not enter the receiving owner's exact scope")
    return target_sid, target_pgid


def _run_target(spec: dict[str, Any], status_read: int, status_write: int, handled: tuple[int, ...]) -> None:
    _close(status_read)
    _reset_handlers(handled)
    try:
        if _SET_PGID is None:
            raise TransferOwnerError("receiving owner requires process-group creation")
        _SET_PGID(0, 0)
        target_env = build_subprocess_env(
            scrub_secrets=False,
            inherit_profile_home=False,
        )
        target_env[_DESCENDANT_GUARD_ENV] = _AUTHORITY_MODE
        target_env[_LIFETIME_ENV] = "contained"
        os.execvpe(spec["executable"], list(spec["argv"]), target_env)
    except BaseException as exc:
        payload = _error(
            type(exc).__name__,
            str(exc),
            errno=getattr(exc, "errno", None),
        )
        try:
            os.write(
                status_write,
                json.dumps(payload, separators=(",", ":")).encode("utf-8")[:_MAX_ACK_BYTES],
            )
        except OSError:
            pass
        finally:
            os._exit(127)


def _supervise(child_pid: int, requested: list[int]) -> None:
    force_signals = {sig for sig in (_SIGHUP, _SIGUSR2) if sig is not None}
    while True:
        try:
            waited, status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            waited, status = child_pid, 0

        if waited == child_pid:
            _kill_scope(child_pid, _SIGKILL)
            _exit_from_status(status)

        if requested:
            status = _stop_scope(child_pid, force=requested[-1] in force_signals)
            _kill_scope(child_pid, _SIGKILL)
            _exit_from_status(status)

        time.sleep(0.05)


def main() -> None:
    if len(sys.argv) < 2:
        raise TransferOwnerError("process-transfer specification is missing")
    spec = _decode_spec(sys.argv[1])
    if _FORK is None or _SET_PGID is None or _KILL_GROUP is None:
        raise TransferOwnerError("receiving owner requires POSIX process primitives")

    ack_fd = int(spec["ack_fd"])
    os.set_inheritable(ack_fd, False)
    requested: list[int] = []
    handled = _install_handlers(requested)

    status_read, status_write = os.pipe()
    os.set_inheritable(status_write, False)
    child_pid = _FORK()
    if child_pid == 0:
        _run_target(spec, status_read, status_write, handled)
        raise AssertionError("transferred target returned from exec")

    _close(status_write)
    try:
        _SET_PGID(child_pid, child_pid)
    except (PermissionError, ProcessLookupError):
        pass

    for fd in spec.get("target_pass_fds", []):
        if fd > 2 and fd != ack_fd:
            _close(fd)

    owner_pid = os.getpid()
    if os.getppid() != int(spec["sender_pid"]):
        exec_error = _error("SenderExited", "transferring owner exited before receiver startup")
    else:
        exec_error = _read_exec_status(
            status_read,
            sender_pid=int(spec["sender_pid"]),
            requested=requested,
        )
    _close(status_read)

    ack: dict[str, Any] = {
        "protocol": spec["protocol"],
        "token": spec["token"],
        "receiver": spec["receiver"],
        "owner_pid": owner_pid,
        "owner_sid": int(_GET_SID(0)) if _GET_SID is not None else -1,
        "owner_pgid": os.getpgrp(),
        "scope_id": child_pid,
    }

    if exec_error is None:
        try:
            target_sid, target_pgid = _target_identity(child_pid, owner_pid)
            ack["target_sid"] = target_sid
            ack["target_pgid"] = target_pgid
        except TransferOwnerError as exc:
            exec_error = _error(type(exc).__name__, str(exc))
    if exec_error is not None:
        ack["error"] = exec_error

    acknowledged = _write_ack(ack_fd, ack)
    _close(ack_fd)
    if exec_error is not None or not acknowledged:
        _stop_scope(child_pid, force=True)
        os._exit(127 if exec_error is not None else 126)

    _supervise(child_pid, requested)


if __name__ == "__main__":
    main()
