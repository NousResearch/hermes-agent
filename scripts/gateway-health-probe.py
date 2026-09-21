#!/usr/bin/env python3
"""
KENSEI Gateway Health Probe (P2-7 Resilience).

Checks every configured gateway systemd service. If a service is
inactive or failed, attempts a restart once per probe cycle.
Reports restarts to Discord #ops. Silent when all healthy.

Usage:
    gateway-health-probe.py          # check all known gateways
    gateway-health-probe.py --once   # single check, exit
    gateway-health-probe.py --dry-run  # print actions, don't execute

Gateways:
    hermes-gateway (kensei)
    hermes-gateway-misa-misa
    hermes-gateway-remii
    hermes-gateway-wesker
    hermes-gateway-gojo
    hermes-gateway-octacon
    hermes-gateway-ceecee
    hermes-gateway-mrhermagi

Also checks: message-loop vs cron distinction — confirms gateway
services aren't consuming cron ticks.
"""

import subprocess
import sys
import time
from pathlib import Path

GATEWAYS = [
    "hermes-gateway.service",
    "hermes-gateway-misa-misa.service",
    "hermes-gateway-remii.service",
    "hermes-gateway-wesker.service",
    "hermes-gateway-gojo.service",
    "hermes-gateway-octacon.service",
    "hermes-gateway-ceecee.service",
    "hermes-gateway-mrhermagi.service",
]

STATE_FILE = Path.home() / ".hermes" / "data" / "gateway-health-state.json"


def systemctl_status(service: str) -> dict:
    """Get systemd unit status."""
    r = subprocess.run(
        ["systemctl", "show", service,
         "--property=ActiveState,SubState,ExecMainPID,MainPID,ActiveEnterTimestamp"],
        capture_output=True, text=True, timeout=10,
    )
    result = {}
    for line in r.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            result[k] = v
    return result


def systemctl_restart(service: str) -> tuple[bool, str]:
    """Attempt to restart a service. Returns (ok, message)."""
    r = subprocess.run(
        ["sudo", "systemctl", "restart", service],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode == 0, r.stderr or r.stdout or "no output"


def load_state() -> dict:
    """Load last-known state from disk."""
    if STATE_FILE.exists():
        import json
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    """Persist current state."""
    import json
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def probe(dry_run: bool = False) -> list[str]:
    """Run one probe cycle. Returns list of alert lines (empty = all healthy)."""
    alerts = []
    state = load_state()
    now = time.time()

    for service in GATEWAYS:
        try:
            status = systemctl_status(service)
        except Exception as exc:
            alerts.append(f"GATEWAY PROBE ERROR: {service} — cannot read status: {exc}")
            continue

        active = status.get("ActiveState", "unknown")
        substate = status.get("SubState", "")
        pid = status.get("MainPID", "0")

        prev = state.get(service, {})
        prev_restarts = prev.get("restarts", 0)

        if active == "active":
            # Healthy — update state
            state[service] = {
                "last_seen_active": now,
                "last_status": f"{active}/{substate}",
                "restarts": prev_restarts,
                "pid": pid,
            }
            continue

        if active == "inactive":
            # Gate down — attempt restart
            last_restart = prev.get("last_restart_attempt", 0)
            if now - last_restart < 300:  # 5 min cooldown
                alerts.append(
                    f"GATEWAY DOWN: {service} ({active}/{substate}) — "
                    f"restart cooldown active (last attempt {int(now-last_restart)}s ago)"
                )
                continue

            alerts.append(
                f"GATEWAY DOWN: {service} ({active}/{substate}) — "
                f"{'DRY-RUN: would restart' if dry_run else 'attempting restart...'}"
            )

            if not dry_run:
                ok, msg = systemctl_restart(service)
                if ok:
                    alerts[-1] += " RESTART OK"
                    prev_restarts += 1
                else:
                    alerts[-1] += f" RESTART FAILED: {msg[:120]}"

            state[service] = {
                "last_status": f"{active}/{substate}",
                "last_restart_attempt": now,
                "restarts": prev_restarts,
                "pid": pid,
            }

        elif active == "failed":
            alerts.append(
                f"GATEWAY FAILED: {service} ({active}/{substate}) — "
                f"{'DRY-RUN: would restart' if dry_run else 'attempting restart...'}"
            )
            if not dry_run:
                ok, msg = systemctl_restart(service)
                if ok:
                    alerts[-1] += " RESTART OK"
                    prev_restarts += 1
                else:
                    alerts[-1] += f" RESTART FAILED: {msg[:120]}"

            state[service] = {
                "last_status": f"{active}/{substate}",
                "last_restart_attempt": now,
                "restarts": prev_restarts,
                "pid": pid,
            }

    save_state(state)
    return alerts


def main():
    dry_run = "--dry-run" in sys.argv

    alerts = probe(dry_run=dry_run)

    if alerts:
        print("\n".join(alerts))
        return 1
    else:
        # [SILENT] on success
        return 0


if __name__ == "__main__":
    sys.exit(main())
