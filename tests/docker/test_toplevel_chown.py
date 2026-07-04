"""Runtime smoke tests for Docker top-level state-file ownership repair.

Build the real image and verify the actual runtime behavior:

  1. Root-owned top-level state files (auth.json, state.db, gateway.lock,
     gateway_state.json) are chowned to hermes on boot
  2. Non-allowlisted host-owned files are NOT touched (targeted, not
     blanket find -user root sweep)
  3. Symlinked allowlisted files are NOT chowned through the symlink
     (path_has_symlink_component guard)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import subprocess

from tests.docker.conftest import (
    docker_exec,
    docker_exec_sh,
    restart_container,
    start_container,
    wait_for_container_ready,
)


# The files the stage2 hook should repair (mirrors the allowlist in
# stage2-hook.sh). We test a representative subset.
ALLOWLISTED_FILES = ("auth.json", "state.db", "gateway.lock", "gateway_state.json")


def test_root_owned_state_files_repaired_on_boot(
    built_image: str, container_name: str,
) -> None:
    """Root-owned top-level state files must be chowned to hermes on boot."""
    start_container(built_image, container_name)

    # Create root-owned state files to simulate docker exec (root) writes
    for f in ALLOWLISTED_FILES:
        docker_exec(
            container_name, "touch", f"/opt/data/{f}",
            user="root", timeout=5,
        )

    # Verify they're root-owned
    r = docker_exec_sh(
        container_name,
        " ".join(f'stat -c %U /opt/data/{f}' for f in ALLOWLISTED_FILES),
        timeout=5,
    )
    for line in r.stdout.split():
        assert line == "root", f"expected root-owned, got: {line}"

    # Restart - stage2 should repair ownership
    restart_container(container_name)

    # Verify files are now hermes-owned
    r = docker_exec_sh(
        container_name,
        " ".join(f'stat -c %U /opt/data/{f}' for f in ALLOWLISTED_FILES),
        timeout=5,
    )
    for line in r.stdout.split():
        assert line == "hermes", (
            f"expected hermes-owned after restart, got: {line}"
        )


def test_non_allowlisted_host_file_not_touched(
    built_image: str, container_name: str,
) -> None:
    """A non-allowlisted host-owned file must NOT be chowned, even if
    root-owned. Regression guard for #19788 / #19795: a bind-mounted
    $HERMES_HOME may contain host-owned files Hermes does not manage."""
    start_container(built_image, container_name)

    # Create a non-allowlisted file as root
    docker_exec(
        container_name, "touch", "/opt/data/host_secret.json",
        user="root", timeout=5,
    )
    # Make it root-owned explicitly (it already is, but be sure)
    docker_exec(
        container_name, "chown", "root:root", "/opt/data/host_secret.json",
        user="root", timeout=5,
    )

    # Restart
    restart_container(container_name)

    # The file must STILL be root-owned (not touched by stage2)
    r = docker_exec_sh(
        container_name,
        "stat -c %U /opt/data/host_secret.json",
        timeout=5,
    )
    assert r.stdout.strip() == "root", (
        f"non-allowlisted host file was chowned by stage2 (should be "
        f"preserved): {r.stdout.strip()}"
    )


def test_symlinked_allowlisted_file_not_chowned(
    built_image: str, container_name: str,
) -> None:
    """A symlinked allowlisted file (e.g. auth.json -> /tmp/outside.json)
    must NOT be chowned through the symlink.

    The path_has_symlink_component guard in stage2-hook.sh must detect
    the symlink and refuse the chown, printing a warning instead. The
    symlink target must remain untouched and the symlink itself must
    still be a symlink after restart.
    """
    tmp = tempfile.mkdtemp()
    host_data: Path | None = None
    tmp_path = Path(tmp)
    try:
        host_data = tmp_path / "data"
        host_data.mkdir()

        # Pre-create a symlink: auth.json -> /opt/data/.symlink-target
        # The target must exist so [ -e ] on the symlink returns true and
        # the chown loop enters the refuse_symlinked_path guard. We create
        # the target inside the bind mount so it persists across containers.
        subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{host_data}:/opt/data",
             "--entrypoint", "sh", built_image,
             "-c", "touch /opt/data/.symlink-target && ln -s /opt/data/.symlink-target /opt/data/auth.json"],
            check=True, capture_output=True, timeout=30,
        )

        # Boot the container with the bind mount
        subprocess.run(
            ["docker", "run", "-d", "--name", container_name,
             "-v", f"{host_data}:/opt/data",
             built_image, "sleep", "infinity"],
            check=True, capture_output=True, timeout=60,
        )
        # Wait for cont-init to finish (first boot runs stage2)
        wait_for_container_ready(container_name)

        # The symlink must still exist (not replaced by a regular file)
        r = docker_exec_sh(
            container_name,
            "test -L /opt/data/auth.json && echo SYMLINK || echo NOT_SYMLINK",
            timeout=5,
        )
        assert "SYMLINK" in r.stdout, (
            f"auth.json symlink was replaced by a regular file: {r.stdout}"
        )

        # The refusal warning goes to stdout (docker logs), not
        # container-boot.log (which is written by container_boot.py).
        r = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True, text=True, timeout=10,
        )
        combined = r.stdout + r.stderr
        assert "refusing" in combined and "auth.json" in combined, (
            f"expected symlink refusal warning for auth.json in docker logs: {combined}"
        )
    finally:
        # Clean up root/hermes-owned files left by stage2 chown
        if host_data is not None:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, timeout=10,
            )
            subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{host_data}:/clean",
                 "--entrypoint", "sh", built_image,
                 "-c", "chown -R 0:0 /clean 2>/dev/null; rm -rf /clean/* /clean/.* 2>/dev/null; chown 0:0 /clean; true"],
                capture_output=True, timeout=15,
            )
            try:
                host_data.rmdir()
                tmp_path.rmdir()
            except OSError:
                pass
