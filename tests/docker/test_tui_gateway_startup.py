"""Ink's real bootstrap starts its s6 owner without sharing the terminal."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.docker.conftest import docker_exec, poll_container
from tests.docker.test_interactive_logging import (
    _CATCHALL,
    _STDOUT,
    _WARNING,
    _emit_supervised_output,
    _start_console,
)


@pytest.mark.parametrize("profile", ["default", "multiplexed", "standalone"])
def test_tui_bootstrap_starts_the_correct_supervised_owner(
    built_image: str,
    container_name: str,
    profile: str,
) -> None:
    _start_console(built_image, container_name, tty=True, env=())
    home = "/opt/data" if profile == "default" else "/opt/data/profiles/worker"
    if profile != "default":
        config = "gateway:\n  standalone: true\n" if profile == "standalone" else "{}\n"
        setup = docker_exec(
            container_name,
            "python",
            "-c",
            (
                "from pathlib import Path; "
                "from hermes_cli.service_manager import S6ServiceManager; "
                f"home = Path({home!r}); home.mkdir(parents=True); "
                f"(home / 'config.yaml').write_text({config!r}); "
                "S6ServiceManager().register_profile_gateway('worker', start_now=False)"
            ),
        )
        assert setup.returncode == 0, setup.stderr

    # Run the very helper Ink launches, including its authenticated ticket grant.
    # Keep the token in the container and report only safe identity fields.
    probe = docker_exec(
        container_name,
        "python",
        "-c",
        (
            "import json, subprocess; "
            "r = subprocess.run(['python', '/opt/hermes/ui-tui/scripts/gateway_bootstrap.py', '--start'], "
            "capture_output=True, text=True, timeout=45); "
            "assert r.returncode == 0, r.stderr; "
            "grant = json.loads(r.stdout); "
            "assert grant['url'] and grant['protocols'] and grant['instance_id']; "
            "print(json.dumps({'profile_id': grant['profile_id']}))"
        ),
        timeout=60,
        extra_docker_args=("-e", f"HERMES_HOME={home}"),
    )
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout)["profile_id"] == home

    owner = "worker" if profile == "standalone" else "default"
    identity = docker_exec(
        container_name,
        "python",
        "-c",
        (
            "from hermes_cli.service_manager import S6ServiceManager; "
            "m = S6ServiceManager(); "
            f"assert m.is_running('gateway-{owner}'); "
            + (
                "assert not m.is_running('gateway-worker')"
                if profile == "multiplexed"
                else ""
            )
        ),
    )
    assert identity.returncode == 0, identity.stderr
    _emit_supervised_output(container_name)
    retained, output = poll_container(
        container_name, f"grep -F {_WARNING} {_CATCHALL}", user="root"
    )
    assert retained, output
    logs = subprocess.run(
        ["docker", "logs", container_name],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert _STDOUT not in logs.stdout + logs.stderr
    assert _WARNING not in logs.stdout + logs.stderr


def test_s6_discovery_refuses_another_install_root(
    built_image: str, container_name: str
) -> None:
    _start_console(built_image, container_name, tty=True, env=())
    result = docker_exec(
        container_name,
        "python",
        "-c",
        (
            "import time; from pathlib import Path; "
            "from hermes_cli.gateway_runtime import ensure_gateway_runtime; "
            "receipt = ensure_gateway_runtime(Path('/tmp/foreign-home'), timeout=5); "
            "assert receipt.state == 'inaccessible', receipt.state; "
            "assert receipt.reason_code == 'profile_mismatch', receipt.reason_code"
        ),
    )
    assert result.returncode == 0, result.stderr
