"""Durable, out-of-cgroup verification for updater-owned gateway restarts.

The updater can run inside the gateway service it must restart.  It therefore
cannot wait for that service's cgroup to disappear and return without risking
its own teardown.  This module is launched as a transient systemd service in a
separate cgroup.  It owns the final SIGUSR1, verifies a changed systemd process
and start generation plus an in-process readiness acknowledgement, then writes
the terminal update result atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from utils import atomic_json_write, atomic_write_text

OWNER_RESTART_PENDING_FILE = ".update_owner_restart_pending.json"
OWNER_RESTART_ACK_FILE = ".update_owner_restart_ack.json"
OWNER_RESTART_RESULT_FILE = ".update_owner_restart_result.json"
_OWNER_RESTART_LOCK_FILE = ".update_owner_restart_verifier.lock"
_UPDATE_EXIT_CODE_FILE = ".update_exit_code"
_UPDATE_OUTPUT_FILE = ".update_output.txt"
_REQUEST_VERSION = 2
_RESULT_VERSION = 1
_NONCE_RE = re.compile(r"[0-9a-f]{32}")
_RESTART_NO_EXIT_REASON = "owner exited and Restart=no disables automatic restart"


def _marker(home: Path, name: str) -> Path:
    return Path(home) / name


def _scope_command(scope: str) -> list[str]:
    if scope == "user":
        return ["systemctl", "--user", "--no-ask-password"]
    if scope == "system":
        return ["systemctl", "--no-ask-password"]
    raise ValueError(f"unsupported systemd scope: {scope!r}")


_SYSTEMD_STATE_ARGS = [
    "--property=ActiveState",
    "--property=SubState",
    "--property=MainPID",
    "--property=ExecMainStartTimestampMonotonic",
    "--property=ActiveEnterTimestampMonotonic",
    "--property=Restart",
]


def _run_systemd_state_query(scope: str, service: str) -> subprocess.CompletedProcess[str]:
    command = _scope_command(scope) + ["show", service, *_SYSTEMD_STATE_ARGS]
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 5,
    }
    if scope == "user" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        return subprocess.run(command, **run_kwargs)
    if not hasattr(os, "geteuid"):
        raise OSError("system systemd scope requires a prompt-free privilege path")

    # Match the updater's existing noninteractive policy: prefer a blanket
    # passwordless capability, but still try the exact read-only command for a
    # targeted sudoers rule.  ``-n`` and ``--no-ask-password`` prohibit both
    # sudo and polkit prompts.
    subprocess.run(
        ["sudo", "-n", "true"],
        capture_output=True,
        timeout=5,
    )
    return subprocess.run(["sudo", "-n", *command], **run_kwargs)


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _valid_service_name(service: Any) -> bool:
    return (
        isinstance(service, str)
        and service.startswith("hermes-gateway")
        and "/" not in service
        and "\\" not in service
        and not service.endswith(".service")
    )


def _validate_nonce(nonce: Any) -> str:
    if not isinstance(nonce, str) or _NONCE_RE.fullmatch(nonce) is None:
        raise ValueError("invalid owner restart nonce")
    return nonce


def _request_generation(request: dict[str, Any]) -> tuple[str, int]:
    key = request.get("generation_key")
    if key not in {"exec_start", "active_enter"}:
        raise ValueError("invalid owner restart generation key")
    value = _coerce_int(request.get("old_state", {}).get(key))
    if value <= 0:
        raise ValueError("missing owner restart start generation")
    return str(key), value


def _validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("version") != _REQUEST_VERSION:
        raise ValueError("unsupported owner restart request")
    _validate_nonce(payload.get("nonce"))
    if payload.get("scope") not in {"user", "system"}:
        raise ValueError("invalid owner restart scope")
    if not _valid_service_name(payload.get("service")):
        raise ValueError("invalid owner restart service")
    old_state = payload.get("old_state")
    if not isinstance(old_state, dict):
        raise ValueError("missing owner restart state")
    if (
        old_state.get("active_state") != "active"
        or old_state.get("sub_state") != "running"
        or _coerce_int(old_state.get("main_pid")) <= 0
    ):
        raise ValueError("owner service was not active and running")
    _request_generation(payload)
    if _coerce_int(payload.get("requested_at_ns")) <= 0:
        raise ValueError("invalid owner restart request timestamp")
    if _coerce_int(payload.get("deadline_ns")) <= _coerce_int(
        payload.get("requested_at_ns")
    ):
        raise ValueError("invalid owner restart deadline")
    if payload.get("final_exit_code") not in {0, 1}:
        raise ValueError("invalid deferred update exit code")
    return payload


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_owner_restart_request(
    home: Path,
    *,
    scope: str,
    service: str,
    old_state: dict[str, Any],
    final_exit_code: int,
    timeout_seconds: float,
    nonce: str,
) -> dict[str, Any]:
    """Atomically stage one owner restart request and clear stale IPC state."""
    home = Path(home)
    _scope_command(scope)
    if not _valid_service_name(service):
        raise ValueError("invalid owner restart service")
    nonce = _validate_nonce(nonce)
    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0 or timeout_seconds > 1800:
        raise ValueError("owner restart timeout must be in (0, 1800]")

    normalized_state = {
        "active_state": str(old_state.get("active_state") or ""),
        "sub_state": str(old_state.get("sub_state") or ""),
        "main_pid": _coerce_int(old_state.get("main_pid")),
        "exec_start": _coerce_int(old_state.get("exec_start")),
        "active_enter": _coerce_int(old_state.get("active_enter")),
        "restart": str(old_state.get("restart") or ""),
    }
    generation_key = (
        "exec_start" if normalized_state["exec_start"] > 0 else "active_enter"
    )
    requested_at_ns = time.time_ns()
    request = {
        "version": _REQUEST_VERSION,
        "nonce": nonce,
        "scope": scope,
        "service": service,
        "old_state": normalized_state,
        "generation_key": generation_key,
        "final_exit_code": int(final_exit_code),
        "requested_at_ns": requested_at_ns,
        "deadline_ns": requested_at_ns + int(timeout_seconds * 1_000_000_000),
        "timeout_seconds": timeout_seconds,
    }
    _validate_request(request)

    for name in (
        OWNER_RESTART_ACK_FILE,
        OWNER_RESTART_RESULT_FILE,
        _OWNER_RESTART_LOCK_FILE,
    ):
        try:
            _marker(home, name).unlink(missing_ok=True)
        except OSError:
            pass
    atomic_json_write(
        _marker(home, OWNER_RESTART_PENDING_FILE), request, indent=2, mode=0o600
    )
    return request


def read_systemd_service_state(scope: str, service: str) -> dict[str, Any]:
    """Read one systemd unit's lifecycle identity in a single bounded query."""
    shown = _run_systemd_state_query(scope, service)
    if shown.returncode != 0:
        raise OSError((shown.stderr or "systemctl show failed").strip())
    props: dict[str, str] = {}
    for line in (shown.stdout or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            props[key.strip()] = value.strip()
    return {
        "active_state": props.get("ActiveState", ""),
        "sub_state": props.get("SubState", ""),
        "main_pid": _coerce_int(props.get("MainPID")),
        "exec_start": _coerce_int(props.get("ExecMainStartTimestampMonotonic")),
        "active_enter": _coerce_int(
            props.get("ActiveEnterTimestampMonotonic")
        ),
        "restart": props.get("Restart", ""),
    }


def current_systemd_gateway_service() -> str | None:
    """Return the exact gateway service component containing this process."""
    try:
        cgroup_text = Path("/proc/self/cgroup").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    for line in cgroup_text.splitlines():
        cgroup_path = line.split(":", 2)[-1]
        for component in reversed(cgroup_path.split("/")):
            if component.startswith("hermes-gateway") and component.endswith(
                ".service"
            ):
                return component.removesuffix(".service")
    return None


def acknowledge_owner_restart_ready(
    home: Path,
    *,
    current_service: str | None = None,
    current_pid: int | None = None,
    now_ns: int | None = None,
) -> bool:
    """Acknowledge readiness only from the new, matching gateway owner.

    Call this after the gateway has connected/configured its adapters and marked
    its runtime state running.  The external verifier still independently checks
    systemd ActiveState, MainPID, and start generation before trusting this ack.
    """
    home = Path(home)
    pending_path = _marker(home, OWNER_RESTART_PENDING_FILE)
    if not pending_path.exists():
        return False
    try:
        request = _validate_request(_read_json(pending_path))
        current_service = current_service or current_systemd_gateway_service()
        current_pid = int(current_pid if current_pid is not None else os.getpid())
        now_ns = int(now_ns if now_ns is not None else time.time_ns())
        if current_service != request["service"]:
            return False
        if current_pid <= 0 or current_pid == request["old_state"]["main_pid"]:
            return False
        if now_ns < request["requested_at_ns"] or now_ns > request["deadline_ns"]:
            return False
        atomic_json_write(
            _marker(home, OWNER_RESTART_ACK_FILE),
            {
                "version": 1,
                "nonce": request["nonce"],
                "service": current_service,
                "pid": current_pid,
                "acknowledged_at_ns": now_ns,
            },
            indent=2,
            mode=0o600,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _matching_ready_ack(
    home: Path, request: dict[str, Any], state: dict[str, Any]
) -> bool:
    try:
        ack = _read_json(_marker(home, OWNER_RESTART_ACK_FILE))
        return bool(
            isinstance(ack, dict)
            and ack.get("version") == 1
            and ack.get("nonce") == request["nonce"]
            and ack.get("service") == request["service"]
            and _coerce_int(ack.get("pid")) == _coerce_int(state.get("main_pid"))
            and request["requested_at_ns"]
            <= _coerce_int(ack.get("acknowledged_at_ns"))
            <= request["deadline_ns"]
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _transition_verified(
    home: Path, request: dict[str, Any], state: dict[str, Any]
) -> bool:
    key, old_generation = _request_generation(request)
    return bool(
        state.get("active_state") == "active"
        and state.get("sub_state") == "running"
        and _coerce_int(state.get("main_pid")) > 0
        and _coerce_int(state.get("main_pid"))
        != _coerce_int(request["old_state"].get("main_pid"))
        and _coerce_int(state.get(key)) > 0
        and _coerce_int(state.get(key)) != old_generation
        and _matching_ready_ack(home, request, state)
    )


def _append_update_output(home: Path, message: str) -> None:
    path = _marker(home, _UPDATE_OUTPUT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    try:
        if path.exists() and path.stat().st_size > 0:
            prefix = "\n"
    except OSError:
        prefix = "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, f"{prefix}{message.rstrip()}\n".encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _result_message(
    request: dict[str, Any], *, verified: bool, exit_code: int, reason: str
) -> str:
    if not verified and reason == _RESTART_NO_EXIT_REASON:
        command = "systemctl --user" if request["scope"] == "user" else "sudo systemctl"
        return (
            "✗ Update finalization incomplete: updater-owning service "
            f"{request['service']} exited and has Restart=no; Hermes did not auto-start it.\n"
            "  Restart it manually to load the updated code:\n"
            f"    {command} start {request['service']}\n"
            "  Then verify:\n"
            f"    {command} status {request['service']}"
        )
    scope_flag = "--user " if request["scope"] == "user" else ""
    if verified and exit_code == 0:
        return (
            "✓ Update complete! Updater-owning service "
            f"{request['service']} restarted and passed readiness verification."
        )
    if verified:
        return (
            "⚠ Update finalization incomplete before the owner restart; "
            f"{request['service']} itself restarted successfully."
        )
    return (
        "✗ Update finalization incomplete: updater-owning service "
        f"{request['service']} restart verification failed ({reason}).\n"
        f"  Check: systemctl {scope_flag}status {request['service']}\n"
        f"         journalctl {scope_flag}-u {request['service']} --since '10 min ago'"
    )


def _validate_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("version") != _RESULT_VERSION:
        raise ValueError("unsupported owner restart result")
    _validate_nonce(payload.get("nonce"))
    if not _valid_service_name(payload.get("service")):
        raise ValueError("invalid owner restart result service")
    if not isinstance(payload.get("verified"), bool):
        raise ValueError("invalid owner restart verification flag")
    if payload.get("exit_code") not in {0, 1}:
        raise ValueError("invalid owner restart result exit code")
    if payload.get("exit_code") == 0 and payload.get("verified") is not True:
        raise ValueError("unverified owner restart cannot succeed")
    if not isinstance(payload.get("reason"), str) or not isinstance(
        payload.get("message"), str
    ):
        raise ValueError("invalid owner restart result detail")
    if _coerce_int(payload.get("completed_at_ns")) <= 0:
        raise ValueError("invalid owner restart completion timestamp")
    return payload


def read_owner_restart_result_exit_code(home: Path) -> int | None:
    """Return the atomically committed owner result, if it is valid."""
    try:
        result = _validate_result(
            _read_json(_marker(Path(home), OWNER_RESTART_RESULT_FILE))
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return int(result["exit_code"])


def _persist_verifier_result(
    home: Path,
    request: dict[str, Any],
    *,
    verified: bool,
    reason: str,
    observed_state: dict[str, Any],
) -> int:
    exit_code = int(request["final_exit_code"]) if verified else 1
    message = _result_message(
        request, verified=verified, exit_code=exit_code, reason=reason
    )
    result = {
        "version": _RESULT_VERSION,
        "nonce": request["nonce"],
        "service": request["service"],
        "verified": bool(verified),
        "exit_code": exit_code,
        "reason": reason,
        "observed_state": observed_state,
        "completed_at_ns": time.time_ns(),
        "message": message,
    }

    # Prepare human-readable evidence first, then atomically commit the JSON
    # result as the owner path's authoritative terminal record.  The legacy
    # .update_exit_code marker is only a compatibility projection; gateway
    # watchers also consume the JSON result directly if that projection fails.
    try:
        _append_update_output(home, message)
    except OSError:
        pass
    try:
        atomic_json_write(
            _marker(home, OWNER_RESTART_RESULT_FILE),
            result,
            indent=2,
            mode=0o600,
        )
    except (OSError, TypeError, ValueError):
        return 1

    try:
        atomic_write_text(_marker(home, _UPDATE_EXIT_CODE_FILE), str(exit_code))
    except OSError:
        # The atomic JSON result above remains authoritative and recoverable by
        # the restarted gateway even when this compatibility marker is unwritable.
        pass
    return exit_code


def _existing_result(home: Path, nonce: str) -> int | None:
    try:
        result = _validate_result(
            _read_json(_marker(home, OWNER_RESTART_RESULT_FILE))
        )
        if result.get("nonce") == nonce:
            return int(result["exit_code"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def _acquire_verifier_lock(home: Path, nonce: str) -> int | None:
    path = _marker(home, _OWNER_RESTART_LOCK_FILE)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return None
    os.write(fd, nonce.encode("ascii"))
    os.fsync(fd)
    return fd


def verify_owner_restart(
    home: Path, nonce: str, *, poll_interval: float = 0.25
) -> int:
    """Signal and boundedly verify one staged owner restart request."""
    home = Path(home)
    try:
        nonce = _validate_nonce(nonce)
    except ValueError:
        return 1
    existing = _existing_result(home, nonce)
    if existing is not None:
        return existing

    try:
        request = _validate_request(
            _read_json(_marker(home, OWNER_RESTART_PENDING_FILE))
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 1
    if request["nonce"] != nonce:
        return 1

    lock_fd = _acquire_verifier_lock(home, nonce)
    if lock_fd is None:
        existing = _existing_result(home, nonce)
        return existing if existing is not None else 1
    os.close(lock_fd)

    state: dict[str, Any] = {}
    try:
        try:
            state = read_systemd_service_state(request["scope"], request["service"])
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return _persist_verifier_result(
                home,
                request,
                verified=False,
                reason="could not read initial systemd owner state",
                observed_state=state,
            )

        old_pid = _coerce_int(request["old_state"].get("main_pid"))
        restart_no = request["old_state"].get("restart") == "no"
        initial_old_gone = bool(
            _coerce_int(state.get("main_pid")) != old_pid
            and state.get("active_state") not in {"activating", "deactivating"}
        )
        if restart_no and initial_old_gone:
            return _persist_verifier_result(
                home,
                request,
                verified=False,
                reason=_RESTART_NO_EXIT_REASON,
                observed_state=state,
            )
        if _transition_verified(home, request, state):
            return _persist_verifier_result(
                home,
                request,
                verified=True,
                reason="owner transition and readiness acknowledged",
                observed_state=state,
            )
        if (
            state.get("active_state") != "active"
            or state.get("sub_state") != "running"
            or _coerce_int(state.get("main_pid")) != old_pid
        ):
            return _persist_verifier_result(
                home,
                request,
                verified=False,
                reason="owner state changed before verifier armed",
                observed_state=state,
            )

        try:
            os.kill(old_pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            return _persist_verifier_result(
                home,
                request,
                verified=False,
                reason="could not signal old owner MainPID",
                observed_state=state,
            )

        remaining = max(
            0.0, (request["deadline_ns"] - time.time_ns()) / 1_000_000_000
        )
        deadline = time.monotonic() + remaining
        saw_transition_without_ack = False
        saw_failed_state = False
        while time.monotonic() < deadline:
            try:
                state = read_systemd_service_state(
                    request["scope"], request["service"]
                )
            except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
                time.sleep(max(0.001, poll_interval))
                continue

            key, old_generation = _request_generation(request)
            changed_identity = bool(
                _coerce_int(state.get("main_pid")) > 0
                and _coerce_int(state.get("main_pid")) != old_pid
                and _coerce_int(state.get(key)) > 0
                and _coerce_int(state.get(key)) != old_generation
            )
            if (
                state.get("active_state") == "active"
                and state.get("sub_state") == "running"
                and changed_identity
            ):
                saw_transition_without_ack = True
            if state.get("active_state") == "failed":
                saw_failed_state = True

            old_gone = bool(
                _coerce_int(state.get("main_pid")) != old_pid
                and state.get("active_state") not in {"activating", "deactivating"}
            )
            if restart_no and old_gone:
                return _persist_verifier_result(
                    home,
                    request,
                    verified=False,
                    reason=_RESTART_NO_EXIT_REASON,
                    observed_state=state,
                )

            if _transition_verified(home, request, state):
                return _persist_verifier_result(
                    home,
                    request,
                    verified=True,
                    reason="owner transition and readiness acknowledged",
                    observed_state=state,
                )

            time.sleep(max(0.001, poll_interval))

        if saw_transition_without_ack:
            reason = "new owner never wrote a matching readiness acknowledgement"
        elif saw_failed_state:
            reason = "owner entered failed state before restart verification"
        else:
            reason = "owner restart verification timed out without a transition"
        return _persist_verifier_result(
            home,
            request,
            verified=False,
            reason=reason,
            observed_state=state,
        )
    finally:
        try:
            _marker(home, OWNER_RESTART_PENDING_FILE).unlink(missing_ok=True)
            _marker(home, OWNER_RESTART_ACK_FILE).unlink(missing_ok=True)
            _marker(home, _OWNER_RESTART_LOCK_FILE).unlink(missing_ok=True)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--home", required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args(argv)
    return verify_owner_restart(Path(args.home), args.nonce)


if __name__ == "__main__":
    raise SystemExit(main())
