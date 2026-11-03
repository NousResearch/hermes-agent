"""Verify the Docker image bakes a runnable full Chromium executable.

The image must ship no separate headless-shell package or store entry.
"""
from __future__ import annotations

import json

from tests.docker.conftest import docker_exec, docker_exec_sh, start_container


def test_stage2_discovers_chromium_binary(
    built_image: str, container_name: str,
) -> None:
    """The build bakes the full-browser path, not a shell or .so, and Hermes resolves it.

    Exercise it as the runtime user: an executable bit alone does not prove
    Chromium can load its shared libraries.
    """
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        "cat /etc/hermes/chromium-executable-path",
        timeout=10,
    )
    assert r.returncode == 0, f"no baked Chromium path: {r.stderr}"
    browser_path = r.stdout.strip()
    assert browser_path, "/etc/hermes/chromium-executable-path is empty"

    # Hermes's own resolver (the launcher's binary) lands on the same build.
    r = docker_exec(
        container_name,
        "/opt/hermes/.venv/bin/python", "-c",
        "from hermes_cli.browser_runtime import chromium_executable; "
        "print(chromium_executable(allow_override=False))",
        timeout=30,
    )
    assert r.returncode == 0, f"chromium_executable() failed: {r.stderr}"
    assert r.stdout.strip() == browser_path, r.stdout

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
    accepted_names = ("chrome", "chromium", "chromium-browser")
    r = docker_exec_sh(
        container_name,
        f'basename "{browser_path}"',
        timeout=5,
    )
    basename = r.stdout.strip()
    assert basename in accepted_names, (
        f"discovered binary basename {basename!r} is not a recognized "
        f"browser name (accepted: {accepted_names}) — the discovery may "
        f"have picked up a shell or shared library instead of full Chromium"
    )

    r = docker_exec(
        container_name,
        "python3", "-c",
        "import json; from pathlib import Path; "
        "root = Path('/opt/hermes/tools'); "
        "packages = json.loads((root / 'facts.json').read_text())['packages']; "
        "print(json.dumps({'packages': list(packages), 'shell_entries': "
        "[p.name for p in root.glob('*headless*shell*')]}))",
        timeout=10,
    )
    assert r.returncode == 0, f"cannot inspect installed browser packages: {r.stderr}"
    inventory = json.loads(r.stdout)
    assert "chromium" in inventory["packages"], inventory
    assert "chromium-headless-shell" not in inventory["packages"], inventory
    assert not inventory["shell_entries"], inventory

    r = docker_exec(
        container_name,
        browser_path,
        "--version",
        timeout=10,
    )
    assert r.returncode == 0, f"full Chromium executable failed: {r.stderr}"
    assert "Chrome" in r.stdout or "Chromium" in r.stdout, r.stdout
