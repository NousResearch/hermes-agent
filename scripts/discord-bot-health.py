#!/usr/bin/env python3
"""Discord Bot Health Monitor - checks all 9 gateway bots for Discord connectivity.

Silent when healthy (exit 0, empty stdout).
Outputs alert text when a bot is down or unreachable.
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# P13 isolation: HERMES derives from HERMES_HOME (env-overridable) so
# local disposable runs never touch /home/kensei/.hermes.
HERMES = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))

# Known gateway service inventory. discover_services() prefers live
# systemctl discovery, but when systemd is unavailable or a unit file is
# not yet installed, this fallback list ensures every gateway bot is
# still checked. kensei-review and quan were previously missed because
# the glob only returns installed unit files and these gateways may not
# have unit files present in every deployment. Keep this list in sync
# with the agent fleet in agents/.
KNOWN_GATEWAY_SERVICES = [
    ("kensei", "hermes-gateway.service"),
    ("kensei-review", "hermes-gateway-kensei-review.service"),
    ("octacon", "hermes-gateway-octacon.service"),
    ("wesker", "hermes-gateway-wesker.service"),
    ("denji", "hermes-gateway-denji.service"),
    ("ceecee", "hermes-gateway-ceecee.service"),
    ("dezzy", "hermes-gateway-dezzy.service"),
    ("gojo", "hermes-gateway-gojo.service"),
    ("remii", "hermes-gateway-remii.service"),
    ("light", "hermes-gateway-light.service"),
    ("misa-misa", "hermes-gateway-misa-misa.service"),
    ("mrhermagi", "hermes-gateway-mrhermagi.service"),
    ("quan", "hermes-gateway-quan.service"),
]


def discover_services():
    """List every installed Hermes gateway service from systemd."""
    try:
        result = subprocess.run(
            [
                "systemctl",
                "list-unit-files",
                "--type=service",
                "--no-legend",
                "hermes-gateway*.service",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return sorted(KNOWN_GATEWAY_SERVICES)

    services = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        service = fields[0]
        if service == "hermes-gateway.service":
            name = "kensei"
        elif service.startswith("hermes-gateway-") and service.endswith(".service"):
            name = service.removeprefix("hermes-gateway-").removesuffix(".service")
        else:
            continue
        services.append((name, service))

    # Merge with the fallback inventory so bots without installed unit
    # files (kensei-review, quan, etc.) are still checked. systemd
    # discovery takes precedence; the inventory fills any gaps.
    discovered = {name: svc for name, svc in services}
    for name, svc in KNOWN_GATEWAY_SERVICES:
        if name not in discovered:
            services.append((name, svc))
    return sorted(services)

def check_service(name, service):
    """Check if a systemd service is active."""
    result = subprocess.run(
        ["systemctl", "is-active", service],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip() == "active"

def check_pid(name, service):
    """Check if the service has a valid main PID."""
    result = subprocess.run(
        ["systemctl", "show", service, "-p", "MainPID", "--value"],
        capture_output=True, text=True, timeout=10
    )
    pid = result.stdout.strip()
    if not pid or pid == "0":
        return False, pid
    # Verify PID exists in /proc
    proc_path = Path(f"/proc/{pid}")
    return proc_path.exists(), pid

def check_gateway_log_errors(name):
    """Check the last 50 lines of gateway log for critical errors."""
    log_path = HERMES / "profiles" / name / "logs" / "gateway.log"
    if name == "kensei":
        log_path = HERMES / "logs" / "gateway.log"
    if not log_path.exists():
        return []
    
    try:
        result = subprocess.run(
            ["tail", "-50", str(log_path)],
            capture_output=True, text=True, timeout=10
        )
        errors = []
        for line in result.stdout.splitlines():
            lower = line.lower()
            if any(kw in lower for kw in ["critical", "fatal", "unhandled exception"]):
                errors.append(line.strip()[:200])
        return errors[-3:]  # Last 3 errors max
    except Exception:
        return sorted(KNOWN_GATEWAY_SERVICES)

def main():
    alerts = []
    
    for name, service in discover_services():
        is_active = check_service(name, service)
        if not is_active:
            alerts.append(f"**{name}** - service NOT active ({service})")
            continue
        
        has_pid, pid = check_pid(name, service)
        if not has_pid:
            alerts.append(f"**{name}** - service active but PID {pid} not found in /proc")
            continue
        
        # Check for recent critical errors in logs
        errors = check_gateway_log_errors(name)
        if errors:
            error_summary = "; ".join(errors[:2])
            alerts.append(f"**{name}** - recent errors: {error_summary}")
    
    if not alerts:
        sys.exit(0)  # Silent when healthy
    
    # Output alert
    ts = datetime.now().strftime("%d/%m/%y %H:%M")
    print(f"**🔴 Discord Bot Health Alert - {ts}**\n")
    for alert in alerts:
        print(f"• {alert}")
    print(f"\n**Action:** Check `systemctl status <service>` for affected bots.")
    sys.exit(0)  # Still exit 0 - we delivered the alert via stdout

if __name__ == "__main__":
    main()
