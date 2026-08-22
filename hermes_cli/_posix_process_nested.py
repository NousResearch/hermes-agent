"""Retained nested POSIX process authority for contained child controllers.

A nested scope stays owned by the Desktop generation while giving an immediate
owner a narrower process group for subtree-local control.  The returned Popen
remains bound to a small owner process in the caller's retained group; the real
target runs in a fresh session and all signals route through that retained
owner.  This preserves local mutation scope without releasing outer ownership.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import types
from typing import Any

from hermes_cli import _posix_process_authority_state as S

_PROTOCOL = "desktop-posix-nested-v1"
_ACK_SECONDS = 3.0
_MAX_ACK_BYTES = 4096


def _coerce_exec_argv(raw: Any) -> list[str]:
    if isinstance(raw, (str, bytes, os.PathLike)):
        return [os.fsdecode(raw)]
    try:
        values = list(raw)
    except TypeError as exc:
        raise S.ProcessAuthorityError(
            "nested-owned Popen args must be a path or argv sequence"
        ) from exc
    if not values:
        raise S.ProcessAuthorityError("nested-owned Popen args cannot be empty")
    return [os.fsdecode(value) for value in values]


def _target_spec(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], Any, list[str], str]:
    if args:
        raw_args = args[0]
        rewritten_args = list(args)
    elif "args" in kwargs:
        raw_args = kwargs["args"]
        rewritten_args = []
    else:
        raise S.ProcessAuthorityError("nested-owned Popen launch is missing args")

    target_argv = _coerce_exec_argv(raw_args)
    requested_executable = kwargs.pop("executable", None)
    if kwargs.get("shell"):
        shell_executable = os.fsdecode(requested_executable or "/bin/sh")
        target_argv = [shell_executable, "-c", *target_argv]
        target_executable = shell_executable
        kwargs["shell"] = False
    else:
        target_executable = os.fsdecode(requested_executable or target_argv[0])
    return tuple(rewritten_args), kwargs, raw_args, target_argv, target_executable


def _read_ack(
    fd: int,
    owner_pid: int,
    expected_owner_pgid: int,
) -> int:
    import select

    ready, _, _ = select.select([fd], [], [], _ACK_SECONDS)
    if not ready:
        raise S.ProcessAuthorityError(
            "nested process owner did not acknowledge retained child scope"
        )
    payload = os.read(fd, _MAX_ACK_BYTES + 1)
    if not payload or len(payload) > _MAX_ACK_BYTES:
        raise S.ProcessAuthorityError(
            "nested process owner acknowledgement is empty or oversized"
        )
    try:
        decoded = json.loads(payload.decode("utf-8").strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S.ProcessAuthorityError(
            "nested process owner acknowledgement is malformed"
        ) from exc

    if decoded.get("protocol") != _PROTOCOL or decoded.get("owner_pid") != owner_pid:
        raise S.ProcessAuthorityError(
            "nested process owner acknowledgement did not match retained owner"
        )
    if decoded.get("owner_pgid") != expected_owner_pgid:
        raise S.ProcessAuthorityError(
            "nested process owner escaped the caller's retained scope"
        )
    error = decoded.get("error")
    if error:
        detail = error.get("detail") if isinstance(error, dict) else str(error)
        raise S.ProcessAuthorityError(
            f"nested process owner could not exec target: {detail}"
        )
    scope_id = decoded.get("scope_id")
    if not isinstance(scope_id, int) or scope_id <= 0 or scope_id == owner_pid:
        raise S.ProcessAuthorityError(
            "nested process owner acknowledgement carried an invalid child scope"
        )
    if decoded.get("target_sid") != scope_id or decoded.get("target_pgid") != scope_id:
        raise S.ProcessAuthorityError(
            "nested process owner did not prove the target's exact private scope"
        )
    return scope_id


def _abort_owner(child: subprocess.Popen[Any]) -> None:
    try:
        child.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        child.wait(timeout=S.DEFAULT_FORCE_SECONDS)
        return
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass
    try:
        child.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        child.wait(timeout=S.DEFAULT_FORCE_SECONDS)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass


def launch_nested_owned_popen(
    child: subprocess.Popen[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    env: dict[str, str],
) -> None:
    """Launch a contained private scope without releasing Desktop ownership."""

    if not kwargs.get("start_new_session"):
        raise S.ProcessAuthorityError(
            "nested-owned child must request start_new_session=True"
        )
    if kwargs.get("process_group") is not None:
        raise S.ProcessAuthorityError(
            "nested-owned child cannot combine a process_group override"
        )
    if kwargs.get("preexec_fn") is not None:
        raise S.ProcessAuthorityError(
            "nested-owned child cannot use opaque preexec_fn code"
        )

    rewritten_args, kwargs, original_args, target_argv, target_executable = (
        _target_spec(args, kwargs)
    )
    read_fd, write_fd = os.pipe()
    prior_pass_fds = tuple(kwargs.get("pass_fds") or ())
    kwargs["pass_fds"] = tuple(dict.fromkeys((*prior_pass_fds, write_fd)))
    kwargs["close_fds"] = True

    # The owner itself must stay in the caller's retained scope.  It is the
    # anchor that lets outer teardown reach and drain this nested scope.
    kwargs["start_new_session"] = False
    kwargs.pop("process_group", None)
    kwargs["env"] = env

    wrapper_spec = {
        "protocol": _PROTOCOL,
        "ack_fd": write_fd,
        "parent_pid": os.getpid(),
        "argv": target_argv,
        "executable": target_executable,
        "target_pass_fds": list(prior_pass_fds),
    }
    encoded_spec = base64.urlsafe_b64encode(
        json.dumps(wrapper_spec, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    wrapper_argv = [
        sys.executable,
        "-m",
        "hermes_cli.posix_nested_owner",
        encoded_spec,
    ]
    if rewritten_args:
        rewritten_args = (wrapper_argv, *rewritten_args[1:])
    else:
        kwargs["args"] = wrapper_argv

    expected_owner_pgid = os.getpgrp()
    try:
        S.original_popen_init(child, *rewritten_args, **kwargs)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise

    original_send_signal = child.send_signal

    def send_authority_signal(self: subprocess.Popen[Any], sig: int) -> None:
        force_control = S.SIGUSR2 or S.SIGHUP
        if sig == S.SIGKILL:
            if force_control is None:
                raise S.ProcessAuthorityError(
                    "nested process owner has no force-control signal"
                )
            original_send_signal(force_control)
            return
        allowed = {signal.SIGTERM, signal.SIGINT}
        allowed.update(value for value in (S.SIGHUP, S.SIGUSR2) if value is not None)
        if sig not in allowed:
            raise S.ProcessAuthorityError(
                f"signal {sig!r} bypasses nested process authority"
            )
        original_send_signal(sig)

    child.send_signal = types.MethodType(send_authority_signal, child)
    os.close(write_fd)
    try:
        scope_id = _read_ack(read_fd, int(child.pid), expected_owner_pgid)
        child.args = original_args
        setattr(child, "__hermes_nested_owned__", True)
        setattr(child, "__hermes_nested_scope_id__", scope_id)
    except BaseException:
        _abort_owner(child)
        raise
    finally:
        os.close(read_fd)
