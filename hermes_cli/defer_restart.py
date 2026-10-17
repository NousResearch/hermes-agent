"""``hermes gateway defer-restart`` — request a deferred, drain-first gateway restart.

This is the CLI half of the deferred self-restart protocol (see
``gateway/defer_restart.py`` for the protocol core and invariants). It exists so
an agent can safely ask its own gateway to restart *after* the current turn's
final reply has been delivered — the one restart path the terminal-tool lifecycle
guard legitimately allows from inside the gateway process tree.

Contract: the CLI NEVER waits for the restart itself (that would deadlock the
turn it is running inside). It returns as soon as the request outcome is
persisted, or reports the outcome as unknown.

Exit codes:
    0  request accepted (or an idempotent duplicate of an accepted request)
    2  usage error
    3  no matching gateway target
    4  target has no supervisor-managed restart path (or unsupported platform)
    5  refused by the cross-process cooldown
    6  outcome unknown — signal sent but acceptance could not be proven;
       callers must NOT immediately retry (see spec)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Optional

#: How long the CLI polls for its persisted result before declaring "unknown".
CONFIRM_TIMEOUT_S = 5.0
CONFIRM_POLL_S = 0.25

EXIT_ACCEPTED = 0
EXIT_USAGE = 2
EXIT_NO_TARGET = 3
EXIT_NO_SUPERVISOR = 4
EXIT_COOLDOWN = 5
EXIT_UNKNOWN = 6


def _print(msg: str) -> None:
    print(msg, flush=True)


def _resolve_target_generation() -> Optional[dict[str, Any]]:
    """Verified ``(pid, start_time)`` of the running gateway for this home/profile."""
    from gateway.status import _get_process_start_time, get_running_pid

    pid = get_running_pid()
    if not pid or pid <= 0:
        return None
    start = _get_process_start_time(pid)
    if start is None:
        return None
    return {"pid": pid, "start_time": start}


def _supervisor_precheck(identity: dict[str, str]) -> tuple[bool, str]:
    """Does a supervisor actually own a restart path for THIS target?

    Bound to the resolved service identity (never "check service A, signal
    process B"). Promise is limited to: a trusted supervisor-managed restart path
    exists at acceptance time — not a permanent guarantee.
    """
    kind = identity.get("kind", "")
    service_id = identity.get("service_id", "")
    try:
        if kind == "launchd":
            return _precheck_launchd()
        if kind == "systemd":
            return _precheck_systemd(service_id)
        if kind == "s6":
            return _precheck_s6(service_id)
    except Exception as exc:
        return False, f"supervisor probe failed: {exc}"
    return False, f"unsupported supervisor kind {kind!r}"


def _precheck_launchd() -> tuple[bool, str]:
    """launchd: job registered AND its agent plist opts into KeepAlive."""
    from hermes_cli.gateway import (
        _launchd_service_registered,
        get_launchd_plist_path,
    )

    plist = get_launchd_plist_path()
    label = plist.stem
    if not plist.is_file():
        return False, f"launchd agent plist missing: {plist}"
    text = plist.read_text(encoding="utf-8", errors="replace")
    if "KeepAlive" not in text:
        return False, f"launchd agent {label} has no KeepAlive (exit would not relaunch)"
    try:
        registered = _launchd_service_registered(label)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"launchctl probe failed: {exc}"
    if not registered:
        return False, f"launchd job {label} is not loaded"
    return True, f"launchd job {label} (KeepAlive)"


def _precheck_systemd(service_id: str) -> tuple[bool, str]:
    """systemd: unit exists, restart policy covers exit 75, not start-limited."""
    cmd = ["systemctl", "--user", "show", service_id,
           "--property=FragmentPath,Restart,RestartPreventExitStatus,SuccessExitStatus,ActiveState"]
    try:
        result = subprocess.run(cmd, timeout=10, capture_output=True, text=True, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"systemctl probe failed: {exc}"
    if result.returncode != 0:
        return False, f"systemd unit {service_id} not found"
    props: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    if not props.get("FragmentPath"):
        return False, f"systemd unit {service_id} has no unit file"
    restart = props.get("Restart", "no")
    prevented = props.get("RestartPreventExitStatus", "").split()
    succeeded = props.get("SuccessExitStatus", "").split()
    if restart == "no":
        return False, f"systemd unit {service_id} has Restart=no"
    if "75" in prevented:
        return False, f"systemd unit {service_id} blocks exit 75 via RestartPreventExitStatus"
    if restart == "on-failure" and "75" in succeeded:
        return False, f"systemd unit {service_id} treats exit 75 as success under Restart=on-failure"
    state = props.get("ActiveState", "")
    if state == "failed":
        return False, f"systemd unit {service_id} is failed (start-limited?)"
    return True, f"systemd unit {service_id} (Restart={restart})"


def _precheck_s6(service_id: str) -> tuple[bool, str]:
    """s6: the gateway service slot is under an up-wanting supervise."""
    from hermes_cli.gateway import _current_profile_name
    from hermes_cli.service_manager import get_service_manager

    mgr = get_service_manager()
    # s6 slots are named gateway-<profile> (see _dispatch_via_service_manager_if_s6).
    service = f"gateway-{_current_profile_name()}"
    if service not in list(mgr.list_profile_gateways()):
        return False, f"s6 gateway slot {service} is not registered"
    if not mgr.is_running(service):
        return False, f"s6 service {service} is not running"
    return True, f"s6 service {service}"


def _write_request_file(payload: dict[str, Any]) -> bool:
    from gateway.defer_restart import _atomic_write_json, _request_path

    return _atomic_write_json(_request_path(payload["request_id"]), payload)


def cmd_defer_restart(args) -> None:  # noqa: ARG001 - argparse contract
    """Entry point for ``hermes gateway defer-restart``. Exits via SystemExit."""
    if os.name == "nt" or not hasattr(signal, "SIGUSR1"):
        _print("✗ deferred restart is not supported on this platform (no SIGUSR1 / no supervisor contract)")
        raise SystemExit(EXIT_NO_SUPERVISOR)

    from gateway import defer_restart as dr

    # Fresh start: prune anything expired so a crashed sender's leftovers cannot
    # confuse this request (cleanup never deletes accepted/unknown evidence).
    try:
        dr.cleanup_expired()
    except Exception:
        pass

    identity = dr.compute_service_identity()
    if identity is None:
        _print("✗ no supervisor-managed gateway service here (launchd/systemd/s6) — "
               "a drained gateway would never come back; refusing (exit 4)")
        raise SystemExit(EXIT_NO_SUPERVISOR)

    target = _resolve_target_generation()
    if target is None:
        _print("✗ no running gateway matches this profile/home (exit 3)")
        raise SystemExit(EXIT_NO_TARGET)

    ok, why = _supervisor_precheck(identity)
    if not ok:
        _print(f"✗ supervisor pre-check failed: {why} (exit 4)")
        raise SystemExit(EXIT_NO_SUPERVISOR)

    identity_key = dr.service_identity_key(identity)
    blocking = dr.cooldown_active_for(identity_key)
    if blocking is not None:
        _print(f"✗ deferred-restart cooldown active for this gateway service "
               f"(last accepted request {blocking.get('request_id', '?')}, window {dr.COOLDOWN_WINDOW_S:.0f}s) (exit 5)")
        raise SystemExit(EXIT_COOLDOWN)

    request_id = uuid.uuid4().hex
    payload = {
        "schema_version": dr.PROTOCOL_SCHEMA_VERSION,
        "request_id": request_id,
        "service_identity": identity,
        "target_generation": target,
        "requested_at": time.time(),
        "actor_ref": {"parent_pid": os.getppid(), "pid": os.getpid()},
    }
    if not _write_request_file(payload):
        _print("✗ could not persist the restart request file (exit 6)")
        raise SystemExit(EXIT_UNKNOWN)

    try:
        os.kill(target["pid"], signal.SIGUSR1)  # windows-footgun: guarded by the SIGUSR1 check above
    except ProcessLookupError:
        _print("✗ target gateway vanished before the wake-up signal (exit 3); the request file will expire")
        raise SystemExit(EXIT_NO_TARGET)
    except (PermissionError, OSError) as exc:
        _print(f"✗ could not signal the gateway: {exc} (exit 6)")
        raise SystemExit(EXIT_UNKNOWN)

    # Poll for OUR persisted outcome. Never wait for the restart itself.
    deadline = time.monotonic() + CONFIRM_TIMEOUT_S
    while time.monotonic() < deadline:
        result = dr.read_result(request_id)
        if result is not None:
            outcome = result.get("outcome")
            if outcome in (dr.OUTCOME_ACCEPTED, dr.OUTCOME_DUPLICATE, dr.OUTCOME_SUPERSEDED):
                _print(f"✓ deferred restart {outcome} (request {request_id[:12]}…, "
                       f"gateway will restart after the current turn drains)")
                raise SystemExit(EXIT_ACCEPTED)
            if outcome == dr.OUTCOME_REJECTED_COOLDOWN:
                _print(f"✗ gateway refused: cooldown (exit 5)")
                raise SystemExit(EXIT_COOLDOWN)
            _print(f"✗ gateway refused: {outcome} {result.get('detail', '')} (exit 6)")
            raise SystemExit(EXIT_UNKNOWN)
        time.sleep(CONFIRM_POLL_S)

    _print(f"⚠ request {request_id[:12]}… signalled but no outcome recorded within "
           f"{CONFIRM_TIMEOUT_S:.0f}s — result unknown (exit 6); do NOT immediately retry")
    raise SystemExit(EXIT_UNKNOWN)
