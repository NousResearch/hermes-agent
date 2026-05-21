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


def test_dashboard_not_running_by_default(
    built_image: str, container_name: str,
) -> None:
    """Without HERMES_DASHBOARD, no dashboard process should be running."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "30"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(3)
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
         "-e", "HERMES_DASHBOARD=1", built_image, "sleep", "30"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(5)
    r = subprocess.run(
        ["docker", "exec", container_name,
         "pgrep", "-f", "hermes dashboard"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, (
        "Dashboard should be running with HERMES_DASHBOARD=1"
    )


def test_dashboard_port_override(
    built_image: str, container_name: str,
) -> None:
    """HERMES_DASHBOARD_PORT changes the dashboard's listen port."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", "-e", "HERMES_DASHBOARD_PORT=9120",
         built_image, "sleep", "30"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(5)
    r = subprocess.run(
        ["docker", "exec", container_name, "sh", "-c",
         "ss -tlnp 2>/dev/null | grep ':9120' "
         "|| netstat -tln 2>/dev/null | grep ':9120'"],
        capture_output=True, text=True, timeout=10,
    )
    assert "9120" in r.stdout, (
        f"Dashboard not listening on port 9120: stdout={r.stdout!r}"
    )
