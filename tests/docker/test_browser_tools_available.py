"""Verify the image offers the built-in browser_* tools out of the box.

The readiness check only looks for an installed agent-browser CLI and never
installs one, so the image must stage it (#135732).
"""
from __future__ import annotations

import json

from tests.docker.conftest import docker_exec, docker_exec_sh, start_container


def test_builtin_browser_tools_ready(
    built_image: str, container_name: str,
) -> None:
    """With ``browser.backend: off``, check_browser_requirements() must pass
    as the runtime user, using the CLI staged under /opt/hermes/tools."""
    start_container(built_image, container_name)

    r = docker_exec_sh(
        container_name,
        "hermes config set browser.backend off",
        timeout=30,
    )
    assert r.returncode == 0, f"cannot set browser.backend: {r.stderr}"

    r = docker_exec(
        container_name,
        "/opt/hermes/.venv/bin/python", "-c",
        "import json; "
        "from tools.browser_tool_install import "
        "_find_agent_browser, check_browser_requirements; "
        "print(json.dumps({'ready': check_browser_requirements(), "
        "'cli': _find_agent_browser(validate=False)}))",
        timeout=30,
    )
    assert r.returncode == 0, f"readiness probe failed: {r.stderr}"
    probe = json.loads(r.stdout.strip().splitlines()[-1])
    assert probe["ready"], probe
    assert probe["cli"].startswith("/opt/hermes/tools/"), probe

    r = docker_exec(container_name, probe["cli"], "--version", timeout=30)
    assert r.returncode == 0, f"agent-browser does not run: {r.stderr}"
