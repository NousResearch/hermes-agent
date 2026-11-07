"""Gateway diagnostic labels and notes beside the running/not-running verdict."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hermes_cli.config import get_hermes_home

if TYPE_CHECKING:
    from hermes_cli.gateway import GatewayRuntimeSnapshot


def gateway_manager_label(snapshot: GatewayRuntimeSnapshot) -> str:
    """A host's available service manager does not prove it supervises a visible process."""
    if snapshot.has_process_service_mismatch or (
        snapshot.gateway_pids
        and not snapshot.service_installed
        and not snapshot.service_running
        and snapshot.service_scope is not None
    ):
        return "manual"
    # Unmanaged Docker/Termux labels still describe the process's execution environment.
    return snapshot.manager


def _incompatible_owner_verdict() -> str | None:
    """``incompatible: <reason>`` when the gateway answering for this home is one this Hermes's clients
    (TUI, Desktop, chat) refuse, e.g. an older Hermes's daemon left running across an upgrade."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    try:
        observed = discover_gateway_endpoint(get_hermes_home(), timeout=1.5)
    except (OSError, ValueError):  # discovery reports peers as verdicts; only a bad home lands here
        return None
    if observed.state != "incompatible":
        return None
    return f"incompatible: {observed.reason_code or 'unknown'}"


def report_incompatible_owner(pids) -> bool:
    """Print ✗ (with the fix) when the running gateway cannot serve this Hermes; True if printed.
    ``pids`` names the scanned processes for the manual-run line; empty for a service-managed one."""
    stale = _incompatible_owner_verdict()
    if not stale:
        return False
    running = f"Gateway is running (PID: {', '.join(map(str, pids))})" if pids else "The running gateway"
    print(f"✗ {running} but cannot serve this Hermes ({stale})")
    print("  It was started by another Hermes version; TUI, Desktop and chat cannot attach to it.")
    print("  Replace it with this version:")
    print("    hermes gateway restart")
    return True


def print_standalone_by_config() -> None:
    """Default-profile status: name the profiles that opted out of the host multiplexer by config,
    so the served set the host gateway reports is not mistaken for the installed roster."""
    from hermes_cli.profiles import get_active_profile_name, profiles_to_serve
    if (get_active_profile_name() or "default") != "default":
        return
    roster = {name for name, _home in profiles_to_serve(True, include_standalone=True)}
    served = {name for name, _home in profiles_to_serve(True)}
    names = sorted(roster - served - {"default"})
    if names:
        print(f"standalone by config (temporary compatibility shim): {', '.join(names)}")
