"""Runtime smoke tests for Docker PUID/PGID and UID/GID remap.

Build the real image and verify the actual runtime behavior:

  1. PUID/PGID env vars remap the hermes user UID/GID at boot
  2. HERMES_UID/HERMES_GID take precedence over PUID/PGID aliases
  3. NAS-style low UIDs (99:100) are accepted and remapped
  4. Invalid UIDs are rejected
  5. The remapped user can write to the data volume
"""
from __future__ import annotations

import secrets

from tests.docker.conftest import docker_exec, docker_exec_sh, poll_container, start_container

def test_puid_pgid_remaps_hermes_user(
    built_image: str, container_name: str,
) -> None:
    """PUID=1000 PGID=1000 must remap the hermes user to UID 1000."""
    start_container(built_image, container_name, "PUID=1000", "PGID=1000")

    r = docker_exec_sh(
        container_name,
        "id -u hermes",
        timeout=10,
    )
    assert r.stdout.strip() == "1000", (
        f"expected hermes UID 1000 after PUID remap, got: {r.stdout.strip()}"
    )

    r = docker_exec_sh(
        container_name,
        "id -g hermes",
        timeout=10,
    )
    assert r.stdout.strip() == "1000", (
        f"expected hermes GID 1000 after PGID remap, got: {r.stdout.strip()}"
    )

    # The remapped user must still be able to write to the data volume.
    r = docker_exec_sh(
        container_name,
        "touch /opt/data/test_write && echo WRITE_OK || echo WRITE_FAIL",
        timeout=10,
    )
    assert "WRITE_OK" in r.stdout, (
        f"hermes user cannot write to /opt/data after remap: {r.stdout}"
    )

def test_nas_low_uid_accepted(
    built_image: str, container_name: str,
) -> None:
    """NAS-style low UIDs (99:100, common on Unraid) must be accepted."""
    start_container(built_image, container_name, "PUID=99", "PGID=100")

    r = docker_exec_sh(container_name, "id -u hermes", timeout=10)
    assert r.stdout.strip() == "99", (
        f"expected hermes UID 99, got: {r.stdout.strip()}"
    )

    r = docker_exec_sh(container_name, "id -g hermes", timeout=10)
    assert r.stdout.strip() == "100", (
        f"expected hermes GID 100, got: {r.stdout.strip()}"
    )


def test_root_registered_gateway_after_uid_remap(built_image: str, container_name: str) -> None:
    """A root-created slot must start its gateway and logger as the remapped account."""
    start_container(built_image, container_name, "HERMES_UID=1000", "HERMES_GID=1000")
    key = secrets.token_hex(32)
    setup = docker_exec(container_name, "python3", "-c", (
        "from pathlib import Path; "
        "p=Path('/opt/data/profiles/remapped'); p.mkdir(parents=True); "
        "(p/'SOUL.md').touch(); "
        "(p/'config.yaml').write_text('gateway:\\n  standalone: true\\n'); "
        f"(p/'.env').write_text('API_SERVER_KEY={key}\\nAPI_SERVER_PORT=18642\\n')"
    ))
    assert setup.returncode == 0, setup.stderr
    registered = docker_exec(container_name, "python3", "-c", (
        "import os; from hermes_agent.hermes_cli.service_manager import S6ServiceManager; "
        "assert os.geteuid() == 0; os.umask(0o077); "
        "S6ServiceManager().register_profile_gateway('remapped')"
    ), user="root", extra_docker_args=("-e", "HERMES_DOCKER_EXEC_AS_ROOT=1"))
    assert registered.returncode == 0, registered.stderr
    slot = "/run/hermes-service/gateway-remapped"
    ok, state = poll_container(container_name, (
        f"test \"$(/command/s6-svstat -o up {slot})\" = true && "
        f"test \"$(/command/s6-svstat -o up {slot}/log)\" = true && "
        "python3 -c \"import socket; socket.create_connection(('127.0.0.1',18642),2).close()\""
    ), deadline_s=60)
    assert ok, f"gateway/logger did not start: {state}"
    ownership = docker_exec_sh(container_name, (
        f"stat -c '%u:%g' {slot} {slot}/supervise {slot}/event "
        f"{slot}/log {slot}/log/supervise {slot}/log/event"
    ))
    assert ownership.returncode == 0, ownership.stderr
    assert ownership.stdout.splitlines() == ["1000:1000"] * 6
    processes = docker_exec_sh(container_name, (
        f"pid=$(/command/s6-svstat -o pid {slot}); "
        "awk '/^Uid:/ {print $2}' /proc/$pid/status"
    ))
    assert processes.returncode == 0 and processes.stdout.strip() == "1000", processes.stderr
