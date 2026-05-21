"""Harness: dashboard opt-in via HERMES_DASHBOARD.

Today (tini): dashboard starts once when HERMES_DASHBOARD=1; if it crashes
it stays dead. After Phase 2 (s6): dashboard starts once; if it crashes
it is restarted under supervision. The restart-after-crash test lives in
Phase 2 Task 2.5; this file only locks the opt-in surface (which must
not change between tini and s6).
"""
from __future__ import annotations

import subprocess
import time


def _poll(container: str, probe: str, *, deadline_s: float = 30.0,
          interval_s: float = 0.5) -> tuple[bool, str]:
    """Repeatedly run ``probe`` inside the container until it exits 0 or
    ``deadline_s`` elapses. Returns (success, last stdout)."""
    end = time.monotonic() + deadline_s
    last = ""
    while time.monotonic() < end:
        r = subprocess.run(
            ["docker", "exec", container, "sh", "-c", probe],
            capture_output=True, text=True, timeout=10,
        )
        last = r.stdout
        if r.returncode == 0:
            return True, last
        time.sleep(interval_s)
    return False, last


def test_dashboard_not_running_by_default(
    built_image: str, container_name: str,
) -> None:
    """Without HERMES_DASHBOARD, no dashboard process should be running."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "60"],
        check=True, capture_output=True, timeout=30,
    )
    # Give the entrypoint enough time to finish bootstrap; if a dashboard
    # were going to start it'd be visible by now.
    time.sleep(5)
    r = subprocess.run(
        ["docker", "exec", container_name,
         "pgrep", "-f", "hermes dashboard"],
        capture_output=True, text=True, timeout=10,
    )
    # pgrep exits non-zero when no match found
    assert r.returncode != 0, (
        "Dashboard should not be running without HERMES_DASHBOARD"
    )


def test_dashboard_opt_in_starts(
    built_image: str, container_name: str,
) -> None:
    """With HERMES_DASHBOARD=1, a dashboard process should be visible."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    # Poll for the dashboard subprocess to appear — the entrypoint
    # backgrounds it and bootstrap (skills sync etc.) can take a few
    # seconds before the python process actually launches.
    ok, _ = _poll(
        container_name, "pgrep -f 'hermes dashboard'", deadline_s=30.0,
    )
    assert ok, "Dashboard should be running with HERMES_DASHBOARD=1"


def test_dashboard_port_override(
    built_image: str, container_name: str,
) -> None:
    """HERMES_DASHBOARD_PORT changes the dashboard's listen port."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", "-e", "HERMES_DASHBOARD_PORT=9120",
         built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    # The dashboard process appearing in pgrep doesn't mean it's bound
    # to the port yet — uvicorn takes another second or two to come up.
    # The image doesn't ship ss/netstat, so probe /proc/net/tcp directly:
    # port 9120 = 0x23A0, state 0A = LISTEN.
    ok, stdout = _poll(
        container_name,
        "grep -E ' 0+:23A0 .* 0A ' /proc/net/tcp /proc/net/tcp6 "
        "2>/dev/null",
        deadline_s=60.0,
    )
    assert ok, f"Dashboard not listening on port 9120: stdout={stdout!r}"
