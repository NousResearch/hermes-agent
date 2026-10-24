"""Gateway systemd restart readiness: how long the CLI waits for the replacement to reach ``running``.

Extracted from ``hermes_cli/gateway.py``. Bodies read facade helpers through ``_gw()`` (late
binding on ``hermes_cli.gateway``) so the seams tests and callers patch on the facade keep
intercepting the moved code.
"""
from __future__ import annotations

import time


def _gw():
    from hermes_cli import gateway  # late: the facade imports this module
    return gateway


# A multiplexer brings each served profile's adapters up one after another, so a healthy cold boot of a
# large fleet outlasts the static supervisor budget (18 Telegram profiles: ~210s vs 155s, #137995). Every
# startup step the replacement publishes buys at least this long again — sized past the slowest single
# cold-start connect (Telegram's 45s initial cap) plus a profile's config hydration.
_RESTART_BOOT_PROGRESS_GRACE_S = 90.0
# Ceiling on those extensions beyond the base budget: a boot that keeps flapping one platform cannot
# hold the CLI forever.
_RESTART_BOOT_PROGRESS_MAX_EXTENSION_S = 1800.0


def _runtime_boot_progress(state: dict | None, pid: int) -> tuple | None:
    """Fingerprint of the startup steps ``pid`` has published, or None unless ``state`` is that PID's
    still-``starting`` record. Only platform entries this PID wrote count (a previous generation's
    leftovers are not progress), and only their ``state`` (a metrics refresh is not a startup step)."""
    gw = _gw()
    if pid <= 0 or gw._runtime_state_pid(state) != pid or (state or {}).get("gateway_state") != "starting":
        return None
    platforms = state.get("platforms")
    if not isinstance(platforms, dict):
        return ()
    return tuple(sorted(
        (str(key), str(entry.get("state") or ""))
        for key, entry in platforms.items()
        if isinstance(entry, dict) and gw._positive_pid(entry.get("writer_pid")) == pid
    ))


class _BootProgress:
    """Tracks the first replacement's published startup steps. Only that PID earns extensions: a
    later PID means it died, and a crash loop must not keep the CLI waiting."""

    def __init__(self, previous_pid: int | None):
        self.previous_pid = previous_pid
        self.pid = 0
        self.fingerprint: tuple | None = None

    def advanced(self, new_pid: int | None, runtime_pid: int, runtime_state: dict | None) -> bool:
        """Record this poll; True when the tracked replacement published a new startup step."""
        if not new_pid or new_pid == self.previous_pid or runtime_pid != new_pid or self.pid not in (0, new_pid):
            return False
        progress = _runtime_boot_progress(runtime_state, new_pid)
        if progress is None:
            return False
        advanced = bool(self.pid) and progress != self.fingerprint
        self.pid, self.fingerprint = new_pid, progress
        return advanced

    def still_booting(self) -> bool:
        """True when the tracked replacement is alive and its record still says ``starting``."""
        if not self.pid or _runtime_boot_progress(_gw()._read_gateway_runtime_status(), self.pid) is None:
            return False
        from gateway.status import _pid_exists
        return _pid_exists(self.pid)


def _wait_for_systemd_service_restart(
    *,
    system: bool = False,
    previous_pid: int | None = None,
    timeout: float | None = None,
    replacement_observed: list[bool] | None = None,
) -> bool:
    """Wait for the gateway service to become active after a restart handoff. A replacement still
    connecting platforms is booting, not failing: each startup step it publishes extends the
    deadline (capped) instead of declaring it dead mid-boot (#137995)."""
    gw = _gw()
    svc = gw.get_service_name()
    scope_label = gw._service_scope_label(system).capitalize()
    if timeout is None:
        timeout = gw._systemd_restart_wait_timeout(system=system)
    deadline = time.monotonic() + timeout
    progress_ceiling = deadline + _RESTART_BOOT_PROGRESS_MAX_EXTENSION_S
    printed_runtime_wait = printed_boot_progress = False
    boot = _BootProgress(previous_pid)

    while time.monotonic() < deadline:
        props = gw._read_systemd_unit_properties(system=system)
        active_state = props.get("ActiveState", "")
        sub_state = props.get("SubState", "")
        try:
            from gateway.status import get_running_pid
            new_pid = get_running_pid()
        except Exception:
            new_pid = None
        new_pid = new_pid or gw._systemd_main_pid_from_props(props)

        runtime_state = gw._read_gateway_runtime_status()
        runtime_pid = gw._runtime_state_pid(runtime_state)
        if (
            previous_pid is not None
            and replacement_observed is not None
            and not replacement_observed
            and any(p > 0 and p != previous_pid for p in (new_pid or 0, runtime_pid))
        ):
            replacement_observed.append(True)

        if boot.advanced(new_pid, runtime_pid, runtime_state):
            extended = min(time.monotonic() + _RESTART_BOOT_PROGRESS_GRACE_S, progress_ceiling)
            if extended > deadline and not printed_boot_progress:
                print(f"⏳ {scope_label} gateway (PID {new_pid}) is still connecting platforms; "
                      "waiting while startup progresses...")
                printed_boot_progress = True
            deadline = max(deadline, extended)

        if active_state == "active" and new_pid and (previous_pid is None or new_pid != previous_pid):
            if runtime_pid != new_pid:
                runtime_state = gw._read_gateway_runtime_status()
                if runtime_state and gw._runtime_state_pid(runtime_state) != new_pid:
                    runtime_state = None
            gateway_state = (runtime_state or {}).get("gateway_state")
            if gateway_state in ("running", "degraded"):
                print(f"✓ {scope_label} service restarted (PID {new_pid})")
                if gateway_state == "degraded":
                    # Serving, but a configured platform is parked or retrying: a real restart, not a
                    # failure — say so instead of waiting out the timeout and reporting one.
                    print(f"⚠ {scope_label} gateway is DEGRADED — see `hermes gateway status`")
                return True
            if gateway_state == "startup_failed":
                reason = (runtime_state or {}).get("exit_reason") or "startup failed"
                print(
                    f"⚠ {scope_label} service process restarted (PID {new_pid}), but gateway startup failed: {reason}"
                )
                return False
            if not printed_runtime_wait:
                print(f"⏳ {scope_label} service process started (PID {new_pid}); waiting for gateway runtime...")
                printed_runtime_wait = True

        if active_state == "activating" and sub_state == "auto-restart":
            time.sleep(1)
            continue

        if gw._systemd_unit_is_start_limited(props):
            gw._print_systemd_start_limit_wait(system=system)
            return False

        time.sleep(2)

    sudo, _, user_flag = gw._systemd_cli_bits(system)
    if boot.still_booting():
        # Still booting at the ceiling: say so rather than "did not become active", which reads as
        # a dead gateway and invites a second restart that kills this boot mid-way.
        print(
            f"⚠ {scope_label} gateway (PID {boot.pid}) is still starting — left running.\n"
            f"  Watch it come up: {sudo}hermes gateway status\n"
            f"  Check logs:       journalctl {user_flag}-u {svc} -l --since '5 min ago'"
        )
        return False
    print(
        f"⚠ {scope_label} service did not become active within {int(timeout)}s.\n"
        f"  Check status: {sudo}hermes gateway status\n"
        f"  Check logs:   journalctl {user_flag}-u {svc} -l --since '2 min ago'"
    )
    return False


def _systemd_restart_wait_timeout(system: bool = False) -> float:
    """Cover systemd's relaunch delays before applying the runtime wait floor."""
    from gateway.shutdown_forensics import parse_systemd_duration_to_us
    props = _gw()._read_systemd_unit_properties(system=system, properties=("RestartUSec", "TimeoutStartUSec"))
    supervisor_budget = 0.0
    for name in ("RestartUSec", "TimeoutStartUSec"):
        raw = props.get(name, "")
        duration_us = int(raw) if raw.isdigit() else parse_systemd_duration_to_us(raw)
        if duration_us is not None:
            supervisor_budget += duration_us / 1_000_000
    return 60.0 + supervisor_budget
