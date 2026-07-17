"""Runtime smoke tests for Docker HOME overrides and script behavior.

Build the real image and verify the actual runtime behavior:

  1. main-wrapper preserves the Docker ``-w`` working directory
  2. dashboard service resets HOME (via HERMES_REAL_HOME) before privilege drop — both the default fallback AND a custom HERMES_REAL_HOME
  3. dashboard does not auto-add ``--insecure`` from a non-loopback bind host
  4. stage2 hook repairs profiles/ and cron/ ownership on every boot
"""
from __future__ import annotations

import subprocess

from tests.docker.conftest import docker_exec, docker_exec_sh, start_container, restart_container


def test_main_wrapper_preserves_docker_workdir(
    built_image: str, container_name: str,
) -> None:
    """The main-wrapper MUST save and restore the original working directory
    so the container starts in the Docker ``-w`` directory, not /opt/data.

    Regression test for #35472. We pass ``-w /tmp`` and a command that
    prints its cwd; the output must be ``/tmp``, proving the wrapper
    restored the cwd after its internal ``cd /opt/data``.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", "-w", "/tmp",
         built_image, "sh", "-c", "pwd"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"container failed: {r.stderr[-1000:]}"
    # The stage2 hook emits boot logs (config migration, skills sync)
    # to stdout before the CMD runs. The actual pwd output is the LAST
    # line of stdout.
    last_line = r.stdout.strip().split("\n")[-1].strip()
    assert last_line == "/tmp", (
        f"expected cwd /tmp, got {last_line!r} — "
        f"main-wrapper did not preserve the Docker -w directory"
    )


def test_dashboard_service_resets_home(
    built_image: str, container_name: str,
) -> None:
    """The dashboard run script must export HOME before dropping
    privileges, so HOME-anchored state (discord lockfile, XDG dirs) doesn't
    try to write to /root (the /init context's HOME).

    Default fallback: when HERMES_REAL_HOME is not set, HOME defaults
    to /opt/data.

    We check this by inspecting the environment of the dashboard service
    process via /proc/<pid>/environ.

    Since the dashboard requires an auth provider on non-loopback binds,
    we bind to 127.0.0.1 where the auth gate doesn't engage, and check
    the process env.
    """
    start_container(built_image, container_name, "HERMES_DASHBOARD=1", "HERMES_DASHBOARD_HOST=127.0.0.1")

    r = docker_exec_sh(
        container_name,
        'pid=$(pgrep -f "hermes dashboard" | head -1); '
        'if [ -n "$pid" ]; then '
        '  tr "\\0" "\\n" < /proc/$pid/environ | grep "^HOME="; '
        'else '
        '  echo "NO_DASHBOARD_PROCESS"; '
        'fi',
        timeout=15,
    )
    assert "HOME=/opt/data" in r.stdout, (
        f"dashboard process does not have HOME=/opt/data: "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )


def test_dashboard_service_resets_home_with_real_home(
    built_image: str, container_name: str,
) -> None:
    """Custom HERMES_REAL_HOME: the dashboard process must have HOME set
    to the configured value, and a privilege-dropped hermes process must
    be able to write under it.

    This exercises the stage2 hook's HERMES_REAL_HOME mkdir+chown path
    and the dashboard run script's HOME export with a non-default value.
    """
    start_container(
        built_image,
        container_name,
        "HERMES_DASHBOARD=1",
        "HERMES_DASHBOARD_HOST=127.0.0.1",
        "HERMES_REAL_HOME=/home/agent",
    )

    r = docker_exec_sh(
        container_name,
        'pid=$(pgrep -f "hermes dashboard" | head -1); '
        'if [ -n "$pid" ]; then '
        '  tr "\\0" "\\n" < /proc/$pid/environ | grep "^HOME="; '
        'else '
        '  echo "NO_DASHBOARD_PROCESS"; '
        'fi',
        timeout=15,
    )
    assert "HOME=/home/agent" in r.stdout, (
        f"dashboard process does not have HOME=/home/agent: "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )
    assert "NO_DASHBOARD_PROCESS" not in r.stdout, (
        f"dashboard process did not start: stdout={r.stdout!r}"
    )

    write_r = docker_exec_sh(
        container_name,
        'export HOME="${HERMES_REAL_HOME:-/opt/data}"; '
        'mkdir -p "$HOME/.config" && touch "$HOME/.config/hermes-real-home-marker" '
        '&& stat -c "%U" "$HOME/.config/hermes-real-home-marker"',
        timeout=10,
    )
    assert write_r.returncode == 0, (
        f"write under $HOME failed: rc={write_r.returncode} "
        f"stdout={write_r.stdout!r} stderr={write_r.stderr!r}"
    )
    assert "hermes" in write_r.stdout, (
        f"marker not hermes-owned: {write_r.stdout!r}"
    )


def test_dashboard_does_not_auto_insecure_from_host(
    built_image: str, container_name: str,
) -> None:
    """The dashboard MUST NOT auto-add ``--insecure`` based on
    HERMES_DASHBOARD_HOST. The auth gate is the authority now.

    The auth gate is the authority on whether non-loopback binds are
    safe; ``--insecure`` must never be auto-derived from the bind host.

    We start the container with a non-loopback bind host and verify
    the dashboard process does NOT receive ``--insecure`` in its
    command line. If the dashboard fails to start (because the auth
    gate correctly blocks an unauthenticated non-loopback bind), that's
    also acceptable — the point is no auto-insecure.
    """
    start_container(built_image, container_name, "HERMES_DASHBOARD=1", "HERMES_DASHBOARD_HOST=0.0.0.0")

    # Check the dashboard process command line for --insecure.
    r = docker_exec_sh(
        container_name,
        'pid=$(pgrep -f "hermes dashboard" | head -1); '
        'if [ -n "$pid" ]; then '
        '  tr "\\0" " " < /proc/$pid/cmdline; '
        'fi',
        timeout=10,
    )
    cmdline = r.stdout.strip()
    # If the process is running, it must NOT have --insecure.
    if cmdline:
        assert "--insecure" not in cmdline, (
            f"dashboard process has --insecure in cmdline (auto-derived "
            f"from host): {cmdline!r}"
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