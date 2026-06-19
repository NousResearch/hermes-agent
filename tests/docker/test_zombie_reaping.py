"""Harness: PID 1 must reap orphaned zombie processes.

tini (current PID 1) reaps zombies via its built-in subreaper behavior.
s6-overlay's ``/init`` (Phase 2 PID 1) does the same. This invariant is
required for long-running containers spawning subprocesses (subagents,
dashboard, dynamic gateways) — otherwise the process table fills with
defunct entries and eventually exhausts the kernel PID space.

Every ``docker exec`` here runs as the unprivileged ``hermes`` user
(via :func:`docker_exec_sh` in conftest); see the conftest module
docstring.
"""
from __future__ import annotations

import subprocess
import time

from tests.docker.conftest import docker_exec, docker_exec_sh


def test_orphan_zombies_reaped(
    built_image: str, container_name: str,
) -> None:
    """Spawn an orphan child that exits immediately. PID 1 must reap it."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "60"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(2)

    # `( ( sleep 0.1 & ) & ); sleep 1` creates a grandchild detached from
    # the original docker exec session — it becomes an orphan reparented
    # to PID 1 in the container. When it exits, PID 1 must reap it.
    docker_exec_sh(
        container_name, "( ( sleep 0.1 & ) & ); sleep 1", timeout=10,
    )
    time.sleep(1)

    r = docker_exec(container_name, "ps", "axo", "stat,pid,comm")
    zombies = [
        line for line in r.stdout.split("\n")
        if line.strip().startswith("Z")
    ]
    assert not zombies, f"Zombies not reaped by PID 1: {zombies}"
