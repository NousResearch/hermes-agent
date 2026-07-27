"""Runtime smoke tests for Docker stage2 browser executable discovery.

Build the real image and verify the chromium binary is actually
discovered at boot: ``AGENT_BROWSER_EXECUTABLE_PATH`` is set, points to
a real executable, and is a browser binary (not a shared library picked
up by a broad ``find | grep``).
"""
from __future__ import annotations

from tests.docker.conftest import docker_exec_sh, start_container


def test_stage2_discovers_chromium_binary(
    built_image: str, container_name: str,
) -> None:
    """The stage2 hook must discover the Playwright chromium binary and
    export AGENT_BROWSER_EXECUTABLE_PATH so the browser tool can find it.

    The discovery uses filename matching, not a broad ``find | grep``:
    shared libraries (libGLESv2.so etc.) inherit the executable bit from
    Playwright's tarball but must not be picked up. This test verifies the
    discovered binary is a real browser, not a .so.
    """
    start_container(built_image, container_name)

    # AGENT_BROWSER_EXECUTABLE_PATH must be set via s6 container_environment.
    r = docker_exec_sh(
        container_name,
        "cat /run/s6/container_environment/AGENT_BROWSER_EXECUTABLE_PATH",
        timeout=10,
    )
    assert r.returncode == 0, (
        f"AGENT_BROWSER_EXECUTABLE_PATH not set by stage2 hook: {r.stderr}"
    )
    browser_path = r.stdout.strip()
    assert browser_path, "AGENT_BROWSER_EXECUTABLE_PATH is empty"

    # Must be a real file and executable.
    r = docker_exec_sh(
        container_name,
        f'test -x "{browser_path}"',
        timeout=5,
    )
    assert r.returncode == 0, (
        f"discovered browser path is not executable: {browser_path}"
    )

    # Must be a browser binary by basename — NOT a shared library.
    accepted_names = (
        "chrome", "chromium", "chrome-headless-shell",
        "headless_shell", "chromium-browser",
    )
    r = docker_exec_sh(
        container_name,
        f'basename "{browser_path}"',
        timeout=5,
    )
    basename = r.stdout.strip()
    assert basename in accepted_names, (
        f"discovered binary basename {basename!r} is not a recognized "
        f"browser name (accepted: {accepted_names}) — the discovery may "
        f"have picked up a shared library (.so) instead of the real browser"
    )


def test_stage2_browser_path_accessible_to_hermes_user(
    built_image: str, container_name: str,
) -> None:
    """The discovered browser binary must be accessible to the
    unprivileged hermes user (UID 10000), since that's who runs
    agent-browser subprocesses."""
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        'path="$(cat /run/s6/container_environment/AGENT_BROWSER_EXECUTABLE_PATH)" '
        '&& test -r "$path" && test -x "$path"',
        timeout=10,
    )
    assert r.returncode == 0, (
        f"browser binary not readable+executable by hermes user: {r.stderr}"
    )
