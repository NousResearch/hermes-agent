"""Runtime smoke tests for Docker HOME overrides and script behavior.

Build the real image and verify the actual runtime behavior:

  1. dashboard service resets HOME to /opt/data before privilege drop
  2. stage2 hook repairs profiles/ and cron/ ownership on every boot
"""
from __future__ import annotations

from tests.docker.conftest import (
    docker_exec,
    docker_exec_sh,
    poll_container,
    restart_container,
    start_container,
)




def test_dashboard_service_resets_home(
    built_image: str, container_name: str,
) -> None:
    """The dashboard run script must export HOME=/opt/data before dropping
    privileges, so HOME-anchored state (discord lockfile, XDG dirs) doesn't
    try to write to /root (the /init context's HOME).

    Start the container with HERMES_DASHBOARD=1 and verify the running
    dashboard process has HOME=/opt/data in its real environment.

    Since the dashboard requires an auth provider on non-loopback binds,
    we bind to 127.0.0.1 where the auth gate doesn't engage, and check
    the process env.
    """
    start_container(built_image, container_name, "HERMES_DASHBOARD=1", "HERMES_DASHBOARD_HOST=127.0.0.1")

    # Wait for the supervised dashboard process, then read its HOME from
    # /proc/<pid>/environ (the real runtime environment, not the script text).
    ok, out = poll_container(
        container_name,
        'pid=$(pgrep -f "hermes dashboard" | head -1); '
        '[ -n "$pid" ] && tr "\\0" "\\n" < /proc/$pid/environ | grep "^HOME="',
        deadline_s=60.0,
    )
    assert ok, f"dashboard process never started (last probe output: {out!r})"
    assert "HOME=/opt/data" in out, (
        f"dashboard process does not run with HOME=/opt/data: {out!r}"
    )




def test_stage2_repairs_profiles_and_cron_ownership(
    built_image: str, container_name: str,
) -> None:
    """profiles/ and cron/ must both be reclaimed after root-context writes.

    The stage2 hook chowns these dirs to hermes:hermes on every boot.
    We simulate a root-owned file in each, then restart the container
    and verify ownership is repaired.
    """
    start_container(built_image, container_name)

    # Create root-owned files in profiles/ and cron/ to simulate
    # docker exec (root) writes.
    docker_exec(
        container_name, "mkdir", "-p", "/opt/data/profiles/testprof",
        user="root", timeout=5,
    )
    docker_exec(
        container_name, "touch", "/opt/data/profiles/testprof/marker",
        user="root", timeout=5,
    )
    docker_exec(
        container_name, "touch", "/opt/data/cron/root_owned.json",
        user="root", timeout=5,
    )

    # Verify they're root-owned before restart.
    r = docker_exec_sh(
        container_name,
        'stat -c "%U" /opt/data/profiles/testprof/marker '
        '/opt/data/cron/root_owned.json',
        timeout=5,
    )
    assert "root" in r.stdout, (
        f"expected root-owned files before restart, got: {r.stdout!r}"
    )

    # Restart — stage2 hook runs again and repairs ownership.
    restart_container(container_name)

    # Verify files are now owned by hermes.
    r = docker_exec_sh(
        container_name,
        'stat -c "%U" /opt/data/profiles/testprof/marker '
        '/opt/data/cron/root_owned.json',
        timeout=5,
    )
    assert "hermes" in r.stdout, (
        f"expected hermes-owned files after restart, got: {r.stdout!r} — "
        f"stage2 hook did not repair profiles/ and cron/ ownership"
    )