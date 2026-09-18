"""Dashboard docker-backend probe must follow the same runtime resolution as the agent.

Regression: Desktop showed "Docker CLI not found" on podman-only machines even
when the terminal backend was already running containers via HERMES_DOCKER_BINARY
or a PATH podman. The probe looked only for a `docker` binary and called
`docker info --format {{.ServerVersion}}` (a Docker-only field).
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli.web_routers import tools as tools_mod
from tools.environments import docker as docker_mod


@pytest.fixture(autouse=True)
def _reset_docker_cache():
    docker_mod._docker_executable = None
    yield
    docker_mod._docker_executable = None


def _hide_host_runtimes(monkeypatch):
    """Force find_docker() off the host PATH so the probe cannot luck into a real docker."""
    monkeypatch.delenv("HERMES_DOCKER_BINARY", raising=False)
    monkeypatch.setattr(docker_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(docker_mod, "_DOCKER_SEARCH_PATHS", [])
    monkeypatch.setattr(tools_mod.shutil, "which", lambda name: None)


def test_probe_ready_when_only_podman_is_configured(tmp_path, monkeypatch):
    """HERMES_DOCKER_BINARY=podman + working `version` is ready; never `docker info`."""
    _hide_host_runtimes(monkeypatch)
    fake = tmp_path / "podman"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("HERMES_DOCKER_BINARY", str(fake))

    captured: list[list[str]] = []

    def _run(argv, **kwargs):
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="podman version 5.0.0\n", stderr="")

    monkeypatch.setattr(tools_mod.subprocess, "run", _run)

    status, detail = tools_mod._probe_docker_backend({})
    assert status == "ready"
    assert detail == ""
    assert captured == [[str(fake), "version"]]


def test_probe_needs_setup_when_docker_and_podman_are_absent(monkeypatch):
    _hide_host_runtimes(monkeypatch)

    status, detail = tools_mod._probe_docker_backend({})
    assert status == "needs_setup"
    assert "not found" in detail.lower()
    assert "podman" in detail.lower()
