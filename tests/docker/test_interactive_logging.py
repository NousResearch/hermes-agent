"""Keep supervised output away from an interactive Docker CMD's terminal."""

from __future__ import annotations

import subprocess

import pytest

from tests.docker.conftest import docker_exec, poll_container, wait_for_docker_logs

_FOREGROUND = "CONSOLE_PROBE_FOREGROUND"
_STDOUT = "CONSOLE_PROBE_SUPERVISED_STDOUT"
_WARNING = "CONSOLE_PROBE_SUPERVISED_WARNING"
_PROFILE = "console-probe"
_CATCHALL = "/run/uncaught-logs/current"


def _start_console(image: str, name: str, *, tty: bool, env: tuple[str, ...]) -> None:
    args = ["docker", "run", "-d", "--name", name]
    if tty:
        args.append("-t")
    for value in env:
        args.extend(["-e", value])
    probe = (
        "if [ -t 0 ] && [ -t 1 ]; then tty=yes; else tty=no; fi; "
        f'printf "{_FOREGROUND} tty=%s\\n" "$tty"; exec sleep infinity'
    )
    subprocess.run(
        [*args, image, "sh", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = wait_for_docker_logs(name, _FOREGROUND)
    assert f"{_FOREGROUND} tty={'yes' if tty else 'no'}" in output


def _emit_supervised_output(container: str) -> None:
    # Exercise the production service/log scripts and privilege drop. Only the
    # executable is synthetic, so no provider or messaging connection is needed.
    script = (
        f"#!/bin/sh\nprintf '%s\\n' '{_STDOUT}'\n"
        f"printf '%s\\n' '{_WARNING}' >&2\nexec sleep infinity\n"
    )
    code = (
        "from pathlib import Path; "
        "from hermes_cli.service_manager import S6ServiceManager; "
        "directory = Path('/opt/data/console-probe-bin'); "
        "directory.mkdir(); "
        "executable = directory / 'hermes'; "
        f"executable.write_text({script!r}, encoding='utf-8'); "
        "executable.chmod(0o755); "
        "manager = S6ServiceManager(); "
        f"manager.register_profile_gateway('{_PROFILE}', "
        "extra_env={'PATH': str(directory) + ':/command:/usr/bin:/bin'})"
    )
    result = docker_exec(container, "python", "-c", code)
    assert result.returncode == 0, result.stderr
    ready, output = poll_container(
        container,
        f"grep -F {_STDOUT} /opt/data/logs/gateways/{_PROFILE}/current",
    )
    assert ready, f"Supervised stdout was not retained: {output}"


@pytest.mark.parametrize(
    ("tty", "env", "isolated"),
    [
        (True, (), True),
        (False, (), False),
        (True, ("S6_LOGGING=0",), False),
        (True, ("S6_LOGGING=1",), False),
    ],
    ids=["interactive", "headless", "explicit-console-logging", "explicit-tee-logging"],
)
def test_supervised_output_respects_terminal_ownership(
    built_image: str,
    container_name: str,
    tty: bool,
    env: tuple[str, ...],
    isolated: bool,
) -> None:
    _start_console(built_image, container_name, tty=tty, env=env)
    _emit_supervised_output(container_name)
    if isolated:
        logs = subprocess.run(
            ["docker", "logs", container_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        console = logs.stdout + logs.stderr
        assert _STDOUT not in console, console
        assert _WARNING not in console, console
        retained, output = poll_container(
            container_name,
            f"grep -F {_WARNING} {_CATCHALL}",
            user="root",
        )
        assert retained, f"Supervised warning was not retained: {output}"
    else:
        console = wait_for_docker_logs(container_name, _WARNING)
        assert _STDOUT in console, console


def test_explicit_logging_preserves_cmd_terminal(
    built_image: str,
    container_name: str,
) -> None:
    _start_console(built_image, container_name, tty=True, env=("S6_LOGGING=2",))
    _emit_supervised_output(container_name)
    retained, output = poll_container(
        container_name,
        f"grep -F {_WARNING} {_CATCHALL}",
        user="root",
    )
    assert retained, output


def test_explicit_cmd_logging_is_not_overridden(
    built_image: str,
    container_name: str,
) -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "-dt",
            "--name",
            container_name,
            "-e",
            "S6_LOGGING=2",
            "-e",
            "S6_CMD_USE_TERMINAL=0",
            built_image,
            "sh",
            "-c",
            f"echo {_FOREGROUND}; exec sleep infinity",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    retained, output = poll_container(
        container_name,
        f"grep -F {_FOREGROUND} {_CATCHALL}",
        user="root",
    )
    assert retained, output
