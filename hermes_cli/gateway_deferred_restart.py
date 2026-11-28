"""Fail-closed target selection and ACK validation for ``gateway defer-restart``."""
from __future__ import annotations

import os
from pathlib import Path

from gateway.deferred_restart import REQUEST_TIMEOUT, SUPPORTED_SUPERVISORS, valid_drain_timeout


def _positive_pid(value) -> bool:
    return type(value) is int and value > 0


def classify_ack(result, supervisor: str) -> str:
    """Return accepted/rejected/unknown. Negative metadata depends on the reason."""
    if not isinstance(result, dict) or type(result.get("accepted")) is not bool:
        return "unknown"
    allowed = {"already_stopping": {"pid", "supervisor", "drain_timeout"},
               "unsupported_supervisor": {"pid"},
               "unsupported_drain_timeout": {"pid", "supervisor"}}
    if result["accepted"]:
        if "reason" in result or not {"pid", "supervisor", "drain_timeout"} <= result.keys():
            return "unknown"
    else:
        fields = allowed.get(result.get("reason")) if isinstance(result.get("reason"), str) else None
        if fields is None or result.keys() - {"accepted", "reason"} - fields:
            return "unknown"
    if "pid" in result and not _positive_pid(result["pid"]):
        return "unknown"
    if "supervisor" in result and (not isinstance(result["supervisor"], str)
                                    or result["supervisor"] not in SUPPORTED_SUPERVISORS
                                    or result["supervisor"] != supervisor):
        return "unknown"
    if "drain_timeout" in result and not valid_drain_timeout(result["drain_timeout"]):
        return "unknown"
    return "accepted" if result["accepted"] else "rejected"


def _verify_identity(identity: dict, home: Path) -> dict:
    from gateway.status import _GATEWAY_KIND
    if (type(identity.get("protocol")) is not int or identity["protocol"] != 1 or identity.get("kind") != _GATEWAY_KIND
            or not _positive_pid(identity.get("pid"))
            or not isinstance(identity.get("hermes_home"), str)
            or Path(identity["hermes_home"]).resolve() != home
            or identity.get("state") in ("starting", "conflict")):
        raise ValueError("gateway identity is unconfirmed")
    return identity


def resolve_target():
    from gateway import host_rendezvous
    from gateway.host_attach import launched_by_other_tenant, _served_from_identity
    from gateway.status import get_running_pid_identity_strict
    from hermes_cli import gateway as gw
    from hermes_cli.gateway_runtime_discovery import missing_owner_state, query_identify
    from hermes_constants import get_hermes_home

    home = get_hermes_home().resolve()
    try:
        identity = query_identify(home, timeout=2.0)
    except (FileNotFoundError, ConnectionRefusedError):
        # A failed dial proves nothing about an owner. Retain stale/unreadable
        # artifacts as ambiguity, and distinguish a host's reservation from an own PID.
        for name in ("gateway.pid", "gateway.sock", "gateway.sock.path"):
            if os.path.lexists(home / name):
                raise ValueError("unreachable own gateway")
        state = missing_owner_state(home)
        owner = get_running_pid_identity_strict(home / "gateway.pid")
        if state == "inaccessible" or (state == "starting" and owner is None):
            raise ValueError("ambiguous own gateway")
        if state == "absent" and os.path.lexists(home / "gateway.lock"):
            raise ValueError("stale own gateway record")
    else:
        return home, _verify_identity(identity, home), False

    profile = gw._current_profile_name() or "default"
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    from hermes_cli.gateway_multiplex_mode import host_serves_named_profile
    root = multiplexer_root_for(home)
    if root is not None and host_serves_named_profile(root, home) is False:
        raise ValueError("standalone profile is excluded from the host")
    host = gw.host_multiplexer_serving(profile)
    record = host_rendezvous.read_record(host_rendezvous.ROLE_GATEWAY)
    if (host is None or record is None or not record.home or not host.served_known
            or host.standalone or not host.serves(profile)):
        raise ValueError("no confirmed host serving this profile")
    target = Path(host.home).resolve()
    if (target == home or Path(record.home).resolve() != target or record.pid != host.pid
            or launched_by_other_tenant(target, home)
            or (owner is not None and owner[0] != host.pid)):
        raise ValueError("ambiguous or cross-tenant host")
    identity = _verify_identity(query_identify(target, timeout=2.0), target)
    if identity["pid"] != host.pid or profile not in _served_from_identity(identity):
        raise ValueError("host service relationship changed")
    return target, identity, True


def run_deferred_restart() -> int:
    from gateway.control_socket import launch_supervisor, query_gateway_control
    from utils import is_truthy_value

    if is_truthy_value(os.environ.get("HERMES_CRON_SESSION")):
        print("Deferred restart refused in a marked cron session.")
        return 1
    try:
        home, identity, satellite = resolve_target()
        supervisor = launch_supervisor(identity)
        if os.name != "posix" or supervisor not in SUPPORTED_SUPERVISORS:
            print("Deferred restart refused: unsupported supervisor (POSIX systemd, launchd, external only).")
            return 1
    except (OSError, ValueError, RuntimeError):
        print("Deferred restart refused: gateway ownership could not be confirmed; inspect hermes gateway status.")
        return 1
    result = query_gateway_control(home, "defer-restart", timeout=REQUEST_TIMEOUT)
    outcome = classify_ack(result, supervisor)
    if outcome == "unknown":
        print("UNKNOWN: the gateway may already be processing the request. Inspect hermes gateway status "
              "before deciding whether to retry; no automatic retry was made.")
        return 1
    if outcome == "rejected":
        print(f"Deferred restart not newly accepted: {result['reason']}.")
        return 1
    print(f"Restart accepted by PID {result['pid']} ({result['supervisor']}). "
          "Completion and readiness are not confirmed.")
    print(f"Loaded drain timeout: {result['drain_timeout']}s; this is a budget, not a fixed restart delay.")
    print("All profiles served by this gateway are affected." + (" This profile uses its host multiplexer." if satellite else ""))
    print("Existing active-work wait and delivery recovery apply; platform send completion and exactly-once delivery are not guaranteed.")
    return 0


def cmd_gateway_defer_restart(args) -> None:
    raise SystemExit(run_deferred_restart())
