"""Runtime tests for Docker group_add supplementary-group preservation.

Boots the real image with ``--group-add`` and verifies the hermes runtime
user retains the granted GID after the s6-setuidgid privilege drop, which
rebuilds supplementary groups from /etc/group (regression test for #75627).
"""

from __future__ import annotations

import subprocess

from tests.docker.conftest import docker_exec_sh, wait_for_container_ready


def test_group_add_gid_preserved_for_hermes(
    built_image: str,
    container_name: str,
) -> None:
    """A GID granted via --group-add with no /etc/group entry must survive
    the privilege drop: hermes must list it among its supplementary groups."""
    r = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--group-add",
            "1001",
            built_image,
            "sleep",
            "infinity",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert r.returncode == 0, f"docker run --group-add failed: {r.stderr[-1000:]}"
    wait_for_container_ready(container_name)

    # id -G uses the same initgroups()/getgrouplist() path as the
    # s6-setuidgid privilege drop: if the hook did its job, 1001 is now a
    # real membership of hermes instead of a dropped grant.
    r = docker_exec_sh(container_name, "id -G hermes", timeout=10)
    assert r.returncode == 0, f"id -G hermes failed: {r.stderr[-500:]}"
    gids = r.stdout.split()
    assert "1001" in gids, (
        "granted GID 1001 was dropped for hermes by the privilege drop, "
        f"groups now: {gids}"
    )
