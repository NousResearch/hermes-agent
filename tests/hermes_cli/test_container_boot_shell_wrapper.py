"""Role detection through a compose ``command:`` shell wrapper.

Regression for the split-brain boot described in ``fork-maintenance``: both containers share the
bind-mounted ``$HERMES_HOME``, so a dashboard container that reconciles registers
``gateway-default`` with autostart, wins the cross-profile host lock, and the real gateway
container then exits 75 with *"Another gateway already owns this host"* — 8642/8789 never bound.

``container_boot`` decided the container's role from its own argv. Compose files wrap the
invocation as ``sh -lc '<bootstrap>; exec hermes <sub>'``, so the role detector saw ``sh`` and read
every container as "not the dashboard".

These are pure functions over argv tokens, so they run on every host — the s6 registration path
they protect is Linux-only and stays in ``test_container_boot.py``.
"""
from __future__ import annotations

import pytest

from hermes_cli import container_boot


def _s6_argv(*command: str) -> tuple[str, ...]:
    """The argv shape the image really produces: rc.init → main-wrapper → the CMD tokens."""
    return (
        "/bin/sh", "-e", "/run/s6/basedir/scripts/rc.init", "top",
        "/opt/hermes/docker/main-wrapper.sh", *command,
    )


def test_a_dashboard_wrapped_in_a_shell_command_still_reads_as_the_dashboard() -> None:
    argv = _s6_argv(
        "sh", "-lc",
        "(python3 -m pip show edge-tts >/dev/null 2>&1) || python3 -m pip install -q edge-tts; "
        "exec /opt/hermes/.venv/bin/hermes dashboard --host 0.0.0.0 --port 9119 --no-open",
    )

    assert container_boot._is_dashboard_container(argv) is True
    assert container_boot._strip_container_argv_prefix(argv) == [
        "dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open",
    ]


def test_a_gateway_wrapped_in_a_shell_command_still_reads_as_gateway_run() -> None:
    """Before s6 the gateway ran as the container command and the reconciler seeds the run intent
    from that. Reading the wrapper instead of the exec target loses the seed, so a fresh volume
    registers the slot DOWN and the gateway container comes up dark."""
    argv = _s6_argv("sh", "-lc", "exec /opt/hermes/.venv/bin/hermes gateway run")

    assert container_boot._is_dashboard_container(argv) is False
    assert container_boot._is_legacy_gateway_run_request(argv) is True
    assert container_boot._strip_container_argv_prefix(argv) == ["gateway", "run"]


def test_a_shell_command_that_never_runs_hermes_is_not_a_hermes_command() -> None:
    """Only the exec target counts. A bootstrap that merely mentions a hermes *path* must not be
    rewritten into a subcommand, or every shell CMD would inherit someone else's role."""
    argv = _s6_argv("sh", "-lc", "echo /opt/hermes/.venv/bin/hermes; sleep infinity")

    assert container_boot._is_dashboard_container(argv) is False
    assert container_boot._is_legacy_gateway_run_request(argv) is False


def test_the_last_hermes_token_wins_when_no_exec_is_present() -> None:
    """A bootstrap may build a path then invoke it; without ``exec`` the final hermes token is the
    command that runs."""
    argv = _s6_argv("sh", "-lc", "hermes --version; /opt/hermes/.venv/bin/hermes dashboard --port 9119")

    assert container_boot._strip_container_argv_prefix(argv) == [
        "dashboard", "--port", "9119",
    ]


def test_a_shell_without_a_c_flag_keeps_its_arguments() -> None:
    """``sh -e script.sh`` has no inline command, so there is nothing to unwrap."""
    argv = _s6_argv("sh", "-e", "hermes dashboard --port 9119")

    assert container_boot._strip_container_argv_prefix(argv) == [
        "sh", "-e", "hermes dashboard --port 9119",
    ]


def test_a_direct_subcommand_is_unaffected_by_the_shell_unwrap() -> None:
    """Control: the documented form (the subcommand passed directly, no shell) is unchanged."""
    dashboard = _s6_argv("dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open")

    assert container_boot._is_dashboard_container(dashboard) is True
    assert container_boot._strip_container_argv_prefix(dashboard) == [
        "dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open",
    ]


@pytest.mark.parametrize("shell", ["sh", "bash", "dash", "ash", "zsh"])
def test_every_supported_shell_wrapper_is_unwrapped(shell: str) -> None:
    argv = _s6_argv(shell, "-lc", "exec /opt/hermes/.venv/bin/hermes gateway run")

    assert container_boot._strip_container_argv_prefix(argv) == ["gateway", "run"]
