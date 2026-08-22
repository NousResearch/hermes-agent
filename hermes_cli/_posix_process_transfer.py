"""One-shot child-side handoff for POSIX process authority."""

from __future__ import annotations

import base64
import json
import os
import secrets
import select
import signal
import subprocess
import sys
import time
import types
from typing import Any, Literal, Mapping

from hermes_cli import _posix_process_authority_state as S


def begin_process_transfer(receiver: str) -> S.ProcessTransferGrant:
    """Mint a one-shot handoff grant inside an installed descendant guard."""

    receiver = receiver.strip()
    if not S.RECEIVER_RE.fullmatch(receiver):
        raise ValueError("transfer receiver must be 3-128 stable identifier characters")
    if not S.guard_installed:
        raise S.ProcessAuthorityError(
            "process transfer requires an installed POSIX descendant guard"
        )

    now = time.monotonic()
    with S.transfer_lock:
        S.prune_transfers(now)
        token = secrets.token_urlsafe(32)
        while token in S.pending_transfers:
            token = secrets.token_urlsafe(32)
        S.pending_transfers[token] = S.PendingTransfer(
            receiver,
            now + S.TRANSFER_TTL_SECONDS,
        )
    return S.ProcessTransferGrant(token, receiver)


def _pending_for(grant: S.ProcessTransferGrant) -> S.PendingTransfer:
    now = time.monotonic()
    with S.transfer_lock:
        S.prune_transfers(now)
        pending = S.pending_transfers.get(grant.token)
    if pending is None or pending.receiver != grant.receiver:
        raise S.ProcessAuthorityError(
            "process transfer grant is unknown, expired, or already consumed"
        )
    return pending


def _claim_transfer(token: str, receiver: str) -> S.ProcessTransferGrant:
    now = time.monotonic()
    with S.transfer_lock:
        S.prune_transfers(now)
        pending = S.pending_transfers.pop(token, None)
    if pending is None or pending.receiver != receiver:
        raise S.ProcessAuthorityError(
            "process transfer grant is unknown, expired, or already consumed"
        )
    return S.ProcessTransferGrant(token, receiver)


def revoke_transfer(token: str) -> None:
    with S.transfer_lock:
        S.pending_transfers.pop(token, None)


def desktop_child_env(
    *,
    lifetime: Literal["contained", "transferred", "foreign"] = S.LIFETIME_CONTAINED,
    transfer: S.ProcessTransferGrant | None = None,
    transfer_receipt: str | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment with an explicit lifetime classification."""

    if lifetime not in S.ALLOWED_LIFETIMES:
        raise ValueError(f"unsupported process lifetime: {lifetime!r}")
    if transfer_receipt is not None:
        raise S.ProcessAuthorityError(
            "caller-supplied transfer receipts are not authority; "
            "use begin_process_transfer()"
        )
    if lifetime == S.LIFETIME_CONTAINED and transfer is not None:
        raise S.ProcessAuthorityError(
            "contained process lifetime cannot carry a transfer grant"
        )
    if lifetime != S.LIFETIME_CONTAINED:
        if transfer is None:
            raise S.ProcessAuthorityError(
                f"{lifetime} process lifetime requires a transfer grant"
            )
        _pending_for(transfer)

    env = dict(os.environ if base is None else base)
    S.strip_authority_envelope(env)
    S.strip_transfer_envelope(env)
    env.pop(S.DESCENDANT_GUARD_ENV, None)
    env[S.LIFETIME_ENV] = lifetime
    if lifetime == S.LIFETIME_CONTAINED:
        env[S.DESCENDANT_GUARD_ENV] = S.AUTHORITY_MODE
    else:
        assert transfer is not None
        env[S.TRANSFER_TOKEN_ENV] = transfer.token
        env[S.TRANSFER_RECEIVER_ENV] = transfer.receiver
    return env


def normalize_child_env(
    raw: Mapping[str, str] | None,
) -> tuple[dict[str, str], str]:
    env = dict(os.environ if raw is None else raw)
    lifetime = (env.get(S.LIFETIME_ENV) or S.LIFETIME_CONTAINED).strip()
    if lifetime not in S.ALLOWED_LIFETIMES:
        raise S.ProcessAuthorityError(
            f"unsupported descendant process lifetime: {lifetime!r}"
        )

    token = (env.get(S.TRANSFER_TOKEN_ENV) or "").strip()
    receiver = (env.get(S.TRANSFER_RECEIVER_ENV) or "").strip()
    S.strip_authority_envelope(env)
    env[S.LIFETIME_ENV] = lifetime
    if lifetime == S.LIFETIME_CONTAINED:
        if token or receiver:
            raise S.ProcessAuthorityError(
                "contained descendant carried a transfer capability"
            )
        S.strip_transfer_envelope(env)
        env[S.DESCENDANT_GUARD_ENV] = S.AUTHORITY_MODE
        return env, lifetime

    env.pop(S.DESCENDANT_GUARD_ENV, None)
    if not token or not receiver:
        raise S.ProcessAuthorityError(
            f"{lifetime} descendant requires a one-shot transfer grant"
        )
    _pending_for(S.ProcessTransferGrant(token, receiver))
    return env, lifetime


def _read_transfer_ack(
    fd: int,
    child_pid: int,
    grant: S.ProcessTransferGrant,
) -> int:
    ready, _, _ = select.select([fd], [], [], S.TRANSFER_ACK_SECONDS)
    if not ready:
        raise S.ProcessAuthorityError(
            f"receiving owner {grant.receiver!r} did not acknowledge process transfer"
        )
    payload = os.read(fd, 4097)
    if not payload or len(payload) > 4096:
        raise S.ProcessAuthorityError(
            "process transfer acknowledgement is empty or oversized"
        )
    try:
        decoded = json.loads(payload.decode("utf-8").strip())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S.ProcessAuthorityError(
            "process transfer acknowledgement is malformed"
        ) from exc

    expected = {
        "protocol": S.TRANSFER_PROTOCOL,
        "token": grant.token,
        "receiver": grant.receiver,
        "owner_pid": child_pid,
        "owner_sid": child_pid,
        "owner_pgid": child_pid,
    }
    for key, value in expected.items():
        if decoded.get(key) != value:
            raise S.ProcessAuthorityError(
                "process transfer acknowledgement did not match the retained "
                "receiving owner"
            )
    error = decoded.get("error")
    if error:
        detail = error.get("detail") if isinstance(error, dict) else str(error)
        raise S.ProcessAuthorityError(
            f"receiving owner could not exec transferred target: {detail}"
        )
    scope_id = decoded.get("scope_id")
    if not isinstance(scope_id, int) or scope_id <= 0 or scope_id == child_pid:
        raise S.ProcessAuthorityError(
            "process transfer acknowledgement carried an invalid owned scope"
        )
    if decoded.get("target_sid") != child_pid or decoded.get("target_pgid") != scope_id:
        raise S.ProcessAuthorityError(
            "process transfer acknowledgement did not prove the target's exact "
            "receiving scope"
        )
    return scope_id


def _abort_unaccepted_transfer(child: subprocess.Popen[Any]) -> None:
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


def _coerce_exec_argv(raw: Any) -> list[str]:
    if isinstance(raw, (str, bytes, os.PathLike)):
        return [os.fsdecode(raw)]
    try:
        values = list(raw)
    except TypeError as exc:
        raise S.ProcessAuthorityError(
            "transferred Popen args must be a path or argv sequence"
        ) from exc
    if not values:
        raise S.ProcessAuthorityError("transferred Popen args cannot be empty")
    return [os.fsdecode(value) for value in values]


def _transfer_target_spec(
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
        raise S.ProcessAuthorityError("transferred Popen launch is missing args")

    target_argv = _coerce_exec_argv(raw_args)
    requested_executable = kwargs.pop("executable", None)
    if kwargs.get("shell"):
        shell_executable = os.fsdecode(requested_executable or "/bin/sh")
        target_argv = [shell_executable, "-c", *target_argv]
        target_executable = shell_executable
        kwargs["shell"] = False
    else:
        target_executable = os.fsdecode(requested_executable or target_argv[0])

    return (
        tuple(rewritten_args),
        kwargs,
        raw_args,
        target_argv,
        target_executable,
    )


def launch_transferred_popen(
    child: subprocess.Popen[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    env: dict[str, str],
) -> None:
    token = (env.get(S.TRANSFER_TOKEN_ENV) or "").strip()
    receiver = (env.get(S.TRANSFER_RECEIVER_ENV) or "").strip()
    if not kwargs.get("start_new_session"):
        revoke_transfer(token)
        raise S.ProcessAuthorityError(
            "transferred child must request start_new_session=True"
        )
    if kwargs.get("process_group") is not None:
        revoke_transfer(token)
        raise S.ProcessAuthorityError(
            "transferred child cannot combine a process_group override"
        )
    if kwargs.get("preexec_fn") is not None:
        revoke_transfer(token)
        raise S.ProcessAuthorityError(
            "transferred child cannot use opaque preexec_fn code"
        )

    grant = _claim_transfer(token, receiver)
    rewritten_args, kwargs, original_args, target_argv, target_executable = (
        _transfer_target_spec(args, kwargs)
    )
    read_fd, write_fd = os.pipe()
    prior_pass_fds = tuple(kwargs.get("pass_fds") or ())
    kwargs["pass_fds"] = tuple(dict.fromkeys((*prior_pass_fds, write_fd)))
    kwargs["close_fds"] = True

    S.strip_transfer_envelope(env)
    env.pop(S.DESCENDANT_GUARD_ENV, None)
    env.pop(S.LIFETIME_ENV, None)
    env[S.TRANSFER_RECEIPT_ENV] = f"{S.TRANSFER_PROTOCOL}:{receiver}"
    kwargs["env"] = env

    wrapper_spec = {
        "protocol": S.TRANSFER_PROTOCOL,
        "token": grant.token,
        "receiver": grant.receiver,
        "ack_fd": write_fd,
        "sender_pid": os.getpid(),
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
        "hermes_cli.posix_transfer_owner",
        encoded_spec,
        "--",
        *target_argv,
    ]
    if rewritten_args:
        rewritten_args = (wrapper_argv, *rewritten_args[1:])
    else:
        kwargs["args"] = wrapper_argv

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
                    "receiving owner has no force-control signal on this POSIX platform"
                )
            original_send_signal(force_control)
            return
        allowed = {signal.SIGTERM, signal.SIGINT}
        allowed.update(
            value for value in (S.SIGHUP, S.SIGUSR2) if value is not None
        )
        if sig not in allowed:
            raise S.ProcessAuthorityError(
                f"signal {sig!r} bypasses transferred process authority"
            )
        original_send_signal(sig)

    child.send_signal = types.MethodType(send_authority_signal, child)
    os.close(write_fd)
    try:
        _read_transfer_ack(read_fd, int(child.pid), grant)
        child.args = original_args
    except BaseException:
        _abort_unaccepted_transfer(child)
        raise
    finally:
        os.close(read_fd)
