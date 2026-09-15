"""Runtime smoke tests for Docker immutable install tree and install-method stamp.

Build the real image and verify at runtime:

  1. /opt/hermes is not writable by the hermes user (immutable install tree)
  2. PYTHONDONTWRITEBYTECODE and HERMES_DISABLE_LAZY_INSTALLS are set
  3. /opt/hermes/.install_method contains "docker" (code-scoped stamp)
  4. $HERMES_HOME/.install_method is NOT stamped as "docker" by stage2
  5. A stale "docker" stamp in $HERMES_HOME is healed (removed) on boot
"""
from __future__ import annotations

import pytest

from tests.docker.conftest import (
    docker_exec,
    docker_exec_sh,
    restart_container,
    start_container,
)


@pytest.mark.parametrize("uid", [10000, 23456])
def test_install_tree_not_writable_by_hermes(
    built_image: str, container_name: str, uid: int,
) -> None:
    """The hermes user must not be able to modify /opt/hermes.

    The install tree (source, venv, TUI bundle, node_modules) must remain
    root-owned and non-writable so an agent session cannot self-modify
    the installation and brick the gateway.
    """
    start_container(built_image, container_name, f"HERMES_UID={uid}", f"HERMES_GID={uid}")
    identity = docker_exec(container_name, "id", "-u")
    assert identity.returncode == 0 and identity.stdout.strip() == str(uid)

    probe = docker_exec(container_name, "/opt/hermes/.venv/bin/python", "-c", """
import os
from pathlib import Path
from hermes_cli.config import detect_install_method
assert os.geteuid() != 0
code = Path('/opt/hermes/.install_method')
home = Path('/opt/data/.install_method')
assert code.read_text().strip() == 'docker'
assert detect_install_method(Path('/opt/hermes')) == 'docker'
assert not home.exists() or home.read_text().strip() != 'docker'
try:
    with code.open('a'):
        pass
except PermissionError:
    pass
else:
    raise AssertionError('runtime user can alter installation method')
assert code.read_text().strip() == 'docker'
for relative in ('pm-runtime/pm-runtime.json', 'tools/facts.json', 'manifest.json'):
    path = Path('/opt/hermes') / relative
    assert path.stat().st_uid == 0, path
    assert path.read_bytes(), path
    assert not os.access(path, os.W_OK), path
    assert path.stat().st_mode & 0o022 == 0, path
for relative in ('.venv/.lock', 'pm-runtime/.lock'):
    path = Path('/opt/hermes') / relative
    assert not path.exists() or not os.access(path, os.W_OK), path
""")
    assert probe.returncode == 0, probe.stdout + probe.stderr

    r = docker_exec_sh(
        container_name,
        # Try to create a file under /opt/hermes as the hermes user
        "touch /opt/hermes/test_write 2>&1 && "
        "echo WRITE_SUCCEEDED || echo WRITE_FAILED",
        timeout=10,
    )
    assert "WRITE_FAILED" in r.stdout, (
        f"hermes user can write to /opt/hermes (install tree not immutable): "
        f"{r.stdout}"
    )

    # Also check a key subdirectory
    r = docker_exec_sh(
        container_name,
        "touch /opt/hermes/.venv/test_write 2>&1 && "
        "echo WRITE_SUCCEEDED || echo WRITE_FAILED",
        timeout=10,
    )
    assert "WRITE_FAILED" in r.stdout, (
        f"hermes user can write to /opt/hermes/.venv: {r.stdout}"
    )


def test_hermes_disable_lazy_installs_and_dont_write_bytecode(
    built_image: str, container_name: str,
) -> None:
    """The container must set PYTHONDONTWRITEBYTECODE and
    HERMES_DISABLE_LAZY_INSTALLS=1 so no .pyc files are written to the
    immutable install tree and no lazy installs attempt to modify it."""
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        'test "$PYTHONDONTWRITEBYTECODE" = "1" && '
        'test "$HERMES_DISABLE_LAZY_INSTALLS" = "1" && '
        'echo ENV_OK || echo ENV_MISSING',
        timeout=10,
    )
    assert "ENV_OK" in r.stdout, (
        f"expected PYTHONDONTWRITEBYTECODE=1 and "
        f"HERMES_DISABLE_LAZY_INSTALLS=1, got: {r.stdout} stderr={r.stderr}"
    )




def test_stale_docker_stamp_in_home_is_healed_on_boot(
    built_image: str, container_name: str,
) -> None:
    """A stale 'docker' stamp left in $HERMES_HOME by an older image
    must be removed on boot so shared homes self-heal."""
    # Start container, write a stale stamp
    start_container(built_image, container_name)

    # Write a stale 'docker' stamp as root
    docker_exec(
        container_name, "sh", "-c",
        "printf 'docker\\n' > /opt/data/.install_method",
        user="root", timeout=5,
    )
    # Verify it exists
    r = docker_exec_sh(container_name, "cat /opt/data/.install_method", timeout=5)
    assert r.stdout.strip() == "docker"

    # Restart - stage2 should heal it
    restart_container(container_name)

    # The stale stamp must be gone
    r = docker_exec_sh(
        container_name,
        "test -f /opt/data/.install_method && "
        "cat /opt/data/.install_method || echo HEALED",
        timeout=10,
    )
    assert "HEALED" in r.stdout or r.stdout.strip() != "docker", (
        f"stale 'docker' stamp in $HERMES_HOME was not healed on boot: "
        f"{r.stdout}"
    )
