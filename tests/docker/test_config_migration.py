"""Runtime smoke test for Docker config-schema migration on boot.

Build the real image and verify: a config.yaml present in $RABBIT_HOME
is migrated by docker_config_migrate.py on boot, running as the rabbit
user.
"""
from __future__ import annotations

from tests.docker.conftest import docker_exec_sh, start_container


def test_config_migration_runs_on_boot(
    built_image: str, container_name: str,
) -> None:
    """A config.yaml in $RABBIT_HOME must be migrated on boot by
    docker_config_migrate.py, running as the rabbit user."""
    # Start container
    start_container(built_image, container_name)

    # Verify config.yaml exists (should be seeded by stage2 if not present)
    r = docker_exec_sh(
        container_name,
        "test -f /opt/data/config.yaml && echo EXISTS || echo MISSING",
        timeout=10,
    )
    assert "EXISTS" in r.stdout, (
        f"config.yaml not found in $RABBIT_HOME: {r.stdout}"
    )

    # Verify config.yaml is owned by rabbit (migration ran as rabbit)
    r = docker_exec_sh(
        container_name,
        'stat -c "%U" /opt/data/config.yaml',
        timeout=10,
    )
    assert r.stdout.strip() == "rabbit", (
        f"config.yaml not owned by rabbit (migration may have run as root): "
        f"{r.stdout.strip()}"
    )


