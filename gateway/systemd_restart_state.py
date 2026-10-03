"""Systemd restart observation, relaunch verification, and recovery state."""
from __future__ import annotations

import subprocess
import time

from gateway import restart, service_identity, systemd_runtime

_CAPTURE_TEXT = dict(
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)


def _parse_kv_pairs(items) -> dict[str, str]:
    """Return stripped KEY=VALUE pairs; later keys win."""
    return {k: v.strip() for k, v in (item.split("=", 1) for item in items if "=" in item)}


def _systemctl_show(properties: tuple[str, ...], *, system: bool) -> dict[str, str]:
    """``systemctl show --property a,b`` for the gateway unit as ``{key: value}``; {} on failure."""
    try:
        result = systemd_runtime.run_systemctl(
            ["show", service_identity.service_name(), "--no-pager", "--property", ",".join(properties)],
            system=systemd_runtime.select_scope(system), timeout=10, **_CAPTURE_TEXT,
        )
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return {}
    return _parse_kv_pairs(result.stdout.splitlines()) if result.returncode == 0 else {}

def _read_systemd_unit_properties(
    system: bool = False,
    properties: tuple[str, ...] = ("ActiveState", "SubState", "Result", "ExecMainStatus", "MainPID"),
) -> dict[str, str]:
    """Return selected ``systemctl show`` properties for the gateway unit."""
    return _systemctl_show(properties, system=system)

def _positive_pid(value) -> int | None:
    """``int(value)`` when it parses and is > 0, else None."""
    try:
        pid = int(value or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None

def _systemd_main_pid_from_props(props: dict[str, str]) -> int | None:
    return _positive_pid(props.get("MainPID", "0") or "0")

def _runtime_state_pid(state: dict | None) -> int:
    """``pid`` recorded in a runtime-status dict; 0 when absent, unparsable, or non-positive."""
    return _positive_pid((state or {}).get("pid", 0)) or 0

def _systemd_main_pid(system: bool = False) -> int | None:
    return _systemd_main_pid_from_props(_read_systemd_unit_properties(system=system))

def _read_gateway_runtime_status() -> dict | None:
    try:
        from gateway.status import read_runtime_status
        state = read_runtime_status()
    except Exception:
        return None
    return state if isinstance(state, dict) else None

def _systemd_cli_bits(system: bool) -> tuple[str, str, str]:
    """``(sudo_prefix, scope_flag, user_flag)`` for printed hints: ``("sudo ", " --system", "")`` in
    system scope, ``("", "", "--user ")`` in user scope."""
    return ("sudo ", " --system", "") if system else ("", "", "--user ")

def _wait_for_systemd_service_restart(
    *,
    system: bool = False,
    previous_pid: int | None = None,
    timeout: float | None = None,
    replacement_observed: list[bool] | None = None,
) -> bool:
    """Wait for the gateway service to become active after a restart handoff."""
    svc = service_identity.service_name()
    scope_label = systemd_runtime.scope_label(system).capitalize()
    if timeout is None:
        timeout = _systemd_restart_wait_timeout(system=system)
    deadline = time.monotonic() + timeout
    printed_runtime_wait = False

    while time.monotonic() < deadline:
        props = _read_systemd_unit_properties(system=system)
        active_state = props.get("ActiveState", "")
        sub_state = props.get("SubState", "")
        try:
            from gateway.status import get_running_pid
            new_pid = get_running_pid()
        except Exception:
            new_pid = None
        new_pid = new_pid or _systemd_main_pid_from_props(props)

        runtime_state = _read_gateway_runtime_status()
        runtime_pid = _runtime_state_pid(runtime_state)
        if (
            previous_pid is not None
            and replacement_observed is not None
            and not replacement_observed
            and any(p > 0 and p != previous_pid for p in (new_pid or 0, runtime_pid))
        ):
            replacement_observed.append(True)

        if active_state == "active" and new_pid and (previous_pid is None or new_pid != previous_pid):
            if runtime_pid != new_pid:
                runtime_state = _read_gateway_runtime_status()
                if runtime_state and _runtime_state_pid(runtime_state) != new_pid:
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

        if _systemd_unit_is_start_limited(props):
            _print_systemd_start_limit_wait(system=system)
            return False

        time.sleep(2)

    sudo, _, user_flag = _systemd_cli_bits(system)
    print(
        f"⚠ {scope_label} service did not become active within {int(timeout)}s.\n"
        f"  Check status: {sudo}hermes gateway status\n"
        f"  Check logs:   journalctl {user_flag}-u {svc} -l --since '2 min ago'"
    )
    return False

def _systemd_restart_wait_timeout(system: bool = False) -> float:
    """Cover systemd's relaunch delays before applying the runtime wait floor."""
    from gateway.shutdown_forensics import parse_systemd_duration_to_us
    props = _read_systemd_unit_properties(system=system, properties=("RestartUSec", "TimeoutStartUSec"))
    supervisor_budget = 0.0
    for name in ("RestartUSec", "TimeoutStartUSec"):
        raw = props.get(name, "")
        duration_us = int(raw) if raw.isdigit() else parse_systemd_duration_to_us(raw)
        if duration_us is not None:
            supervisor_budget += duration_us / 1_000_000
    return 60.0 + supervisor_budget

def _systemd_unit_is_start_limited(props: dict[str, str]) -> bool:
    return "start-limit-hit" in (props.get("Result", "").lower(), props.get("SubState", "").lower())

def _systemd_error_indicates_start_limit(exc: subprocess.CalledProcessError) -> bool:
    parts: list[str] = []
    for attr in ("stderr", "stdout", "output"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(value.decode(errors="replace") if isinstance(value, bytes) else str(value))
    text = "\n".join(parts).lower()
    return "start-limit-hit" in text or "start request repeated too quickly" in text or "start-limit" in text

def _systemd_service_is_start_limited(system: bool = False) -> bool:
    return _systemd_unit_is_start_limited(_read_systemd_unit_properties(system=system))

def _print_systemd_start_limit_wait(system: bool = False) -> None:
    svc = service_identity.service_name()
    scope_label = systemd_runtime.scope_label(system).capitalize()
    sudo, scope_flag, user_flag = _systemd_cli_bits(system)
    print(f"⏳ {scope_label} service is temporarily rate-limited by systemd.")
    print("  systemd is refusing another immediate start after repeated exits.")
    print(f"  Wait for the start-limit window to expire, then run: {sudo}hermes gateway restart{scope_flag}")
    print(f"  Or clear the failed state manually: systemctl {user_flag}reset-failed {svc}")
    print(f"  Check logs: journalctl {user_flag}-u {svc} -l --since '5 min ago'")

def _recover_pending_systemd_restart(system: bool = False, previous_pid: int | None = None) -> bool:
    """Recover a planned service restart that is stuck in systemd state."""
    props = _read_systemd_unit_properties(system=system)
    if not props:
        return False

    try:
        from gateway.status import read_runtime_status
    except Exception:
        return False

    if not (read_runtime_status() or {}).get("restart_requested"):
        return False

    active_state = props.get("ActiveState", "")
    if active_state == "activating" and props.get("SubState", "") == "auto-restart":
        print("⏳ Service restart already pending — waiting for systemd relaunch...")
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    if active_state == "failed" and (
        props.get("ExecMainStatus", "") == str(restart.GATEWAY_SERVICE_RESTART_EXIT_CODE)
        or props.get("Result", "") == "exit-code"
    ):
        svc = service_identity.service_name()
        print(f"↻ Clearing failed state for pending {systemd_runtime.scope_label(system)} service restart...")
        systemd_runtime.run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        systemd_runtime.run_systemctl(["start", svc], system=system, check=False, timeout=90)
        return _wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)

    return False
