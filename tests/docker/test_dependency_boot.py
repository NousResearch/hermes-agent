"""Fail-closed boot contracts for the published image, on disposable volumes."""
from __future__ import annotations

import subprocess
import uuid

import pytest


def _docker(*args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


@pytest.fixture
def volume():
    name = f"hermes-dependency-boot-{uuid.uuid4().hex}"
    _docker("volume", "create", name).check_returncode()
    try:
        yield name
    finally:
        _docker("volume", "rm", name, timeout=30).check_returncode()


def _boot(image: str, volume: str, *args: str) -> subprocess.CompletedProcess[str]:
    return _docker("run", "--rm", "-v", f"{volume}:/opt/data", *args, image, "--version")


def test_offline_selected_plugin_aborts_without_leaking_index_credentials(built_image: str, volume: str):
    marker = "secret-index-credential-537"
    setup = _docker(
        "run", "--rm", "--entrypoint", "sh", "-v", f"{volume}:/opt/data", built_image,
        "-c", "mkdir -p /opt/data/plugins/broken; "
        "printf 'memory:\\n  provider: broken\\n' > /opt/data/config.yaml; "
        "printf '[project]\\nname = \"broken\"\\nversion = \"1.0.0\"\\n"
        "dependencies = [\"impossible-hermes-fixture-package-537==999999\"]\\n' "
        "> /opt/data/plugins/broken/pyproject.toml; "
        "chown -R hermes:hermes /opt/data",
    )
    assert setup.returncode == 0, setup.stderr
    boot = _boot(built_image, volume, "--network", "none", "-e",
                 f"UV_INDEX_URL=https://user:{marker}@index.invalid/simple")
    logs = boot.stdout + boot.stderr
    assert boot.returncode != 0
    assert "dependency refresh failed; refusing startup" in logs
    assert "fatal: stopping the container" in logs
    assert "Hermes Agent v" not in logs
    assert marker not in logs
    assert "Traceback (most recent call last)" not in logs
    assert "No solution found" not in logs


def test_read_only_data_volume_refuses_boot(built_image: str, volume: str):
    boot = _docker("run", "--rm", "-v", f"{volume}:/opt/data:ro", built_image, "--version")
    logs = boot.stdout + boot.stderr
    assert boot.returncode != 0
    assert "fatal: stopping the container" in logs
    assert "Hermes Agent v" not in logs


def test_remapped_uid_and_warm_boot_still_work(built_image: str, volume: str):
    for expected in ("base", "base"):
        boot = _boot(built_image, volume, "-e", "HERMES_UID=11001", "-e", "HERMES_GID=11001")
        logs = boot.stdout + boot.stderr
        assert boot.returncode == 0, logs[-2000:]
        assert "Hermes Agent v" in logs
        assert f"dependency environment: {expected}" in logs
