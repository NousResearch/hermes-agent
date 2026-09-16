"""``hermes gateway restart`` for a gateway whose supervisor Hermes did not install.

A custom launchd agent / systemd unit / any KeepAlive-style manager running ``gateway run
--external-supervisor`` owns the respawn. The manual fallback in ``_cmd_restart`` (SIGTERM, then a
foreground ``run_gateway`` inside the restart CLI) stamps the CLI's own PID as the gateway, so every
supervisor respawn refuses with "Gateway already running (PID <restart>)" and the gateway stays
down until the restart process is killed (#110637). The gateway must instead exit back to its
supervisor (SIGUSR1 drain), and success is a fresh supervised PID — never the bare exit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# A custom KeepAlive supervisor keeps its own respawn interval (launchd's is ~once per 10s,
# per LAUNCHD_SUPERVISION_VERIFY_TIMEOUT); 15s matches _wait_for_launchd_service_pid's budget.
SUPERVISED_REPLACEMENT_VERIFY_TIMEOUT = 15.0


def gateway_declares_external_supervisor(pid: int, home: Path | None = None) -> bool:
    """True when the running gateway ``pid`` was launched for an external supervisor.

    The supervisor is SELF-declared by the gateway from its launch context: the control socket
    ``identify`` answer (any ``supervisor`` other than ``"manual"`` — a custom systemd unit or
    launchd agent sets INVOCATION_ID / the XPC name, so the gateway answers ``systemd``/``launchd``
    even though ``_installed_service_kind_for`` saw no canonical unit, and the same supervisor owns
    the respawn), OR the ``--external-supervisor`` argv marker read live (same marker
    ``_prepare_profile_gateway_update_restart`` trusts), else the argv the gateway stamped into
    ``gateway_state.json`` when psutil cannot read the live command line.
    """
    if not pid or pid <= 1:
        return False
    from gateway.control_socket import identify_gateway
    from gateway.status import _get_process_hermes_home, read_runtime_status
    from hermes_cli.gateway import _capture_gateway_argv

    home = home or _get_process_hermes_home()
    identity = identify_gateway(home) or {}
    if identity.get("pid") == pid and identity.get("supervisor") not in (None, "", "manual"):
        return True
    argv = _capture_gateway_argv(pid)
    if argv is None:
        record = read_runtime_status(home / "gateway_state.json") or {}
        argv = record.get("argv") if record.get("pid") == pid else None
    return bool(argv) and "--external-supervisor" in argv


def _wait_for_supervised_gateway_replacement(
    old_pid: int, timeout: float | None = None, *, poll_interval: float = 0.5
) -> int | None:
    """Poll the pidfile until the supervisor's replacement gateway registers a fresh PID.

    A graceful SIGUSR1 exit only proves the old process left — an unloaded, broken, or
    stopped-retrying supervisor leaves the gateway down. Custom-supervisor counterpart of
    ``_wait_for_launchd_service_pid``: the label is invisible to launchctl queries, so identity
    comes from ``get_running_pid``'s lock+PID liveness verification and freshness from ``!= old_pid``.
    Returns the fresh PID, or None once ``timeout`` passes.
    """
    from gateway.status import get_running_pid

    if timeout is None:
        timeout = SUPERVISED_REPLACEMENT_VERIFY_TIMEOUT
    deadline = time.monotonic() + max(timeout, 0.5)
    while True:
        pid = get_running_pid()
        if pid is not None and pid > 0 and pid != old_pid:
            return pid
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval)


def restart_externally_supervised_gateway(supervised_pid: int) -> None:
    """Hand ``supervised_pid`` back to its supervisor (SIGUSR1 drain) and report the fresh PID.

    Never falls through to SIGTERM + foreground run on either failure branch: that would
    SIGTERM a KeepAlive-armed process and stamp this CLI's PID, recreating the competing-owner
    wedge (#110637). A broken/unloaded supervisor surfaces as exit 1, not a success printed over
    a dead gateway (the contract ``launchd_restart`` enforces via ``_wait_for_launchd_service_pid``).
    """
    from hermes_cli.gateway import _get_restart_exit_wait_budget, _graceful_restart_via_sigusr1, _print_lines

    wait_budget = _get_restart_exit_wait_budget()
    print(f"→ Restarting externally-supervised gateway (PID {supervised_pid}) — "
          f"draining in-flight runs (up to {wait_budget:.0f}s)...")
    if _graceful_restart_via_sigusr1(supervised_pid, wait_budget):
        replacement_pid = _wait_for_supervised_gateway_replacement(supervised_pid)
        if replacement_pid is not None:
            print()
            print(f"✓ Gateway relaunched by its supervisor (PID {replacement_pid})")
            return
        print("⚠ Supervisor did not relaunch the gateway after its graceful exit")
    else:
        print(f"⚠ Gateway did not exit within {wait_budget:.0f}s of SIGUSR1 (or the signal could not be sent)")
    _print_lines(
        "",
        "✗ Not stopping or foreground-running a supervisor-owned gateway.",
        "  Check the supervisor (it may be unloaded, wedged, or stopped retrying),",
        "  then rerun once it is healthy: hermes gateway restart",
    )
    sys.exit(1)
