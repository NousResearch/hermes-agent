"""Tests for the strict gateway command-line matcher.

Regression guard for the Windows ``hermes gateway restart`` silent-outage bug:
the previous loose substring match (``"... gateway" in cmdline``) false-matched
``gateway status``/``dashboard`` siblings and unrelated processes such as
``python -m tui_gateway``, which let ``restart()`` race a still-draining old
process and ``status``/``start`` report false positives.
"""

from __future__ import annotations

import pytest

from gateway.status import (
    looks_like_gateway_command_line as matches,
    looks_like_gateway_runtime_command_line as matches_runtime,
)


ACCEPT = [
    "pythonw.exe -m hermes_cli.main gateway run",
    r"C:\Users\me\hermes\venv\Scripts\pythonw.exe -m hermes_cli.main gateway run",
    "python -m hermes_cli.main --profile work gateway run",
    "python -m hermes_cli.main gateway run --replace",
    "python -m hermes_cli/main.py gateway run",
    "python gateway/run.py",
    "hermes-gateway.exe",
    "hermes gateway",          # bare `hermes gateway` defaults to run
    "hermes gateway run",
    # profile selector AFTER the `gateway` token (argv is profile-position
    # agnostic — _apply_profile_override strips --profile/-p anywhere)
    "hermes gateway --profile work run",
    "python -m hermes_cli.main gateway -p work run",
    "hermes gateway --profile=work run",
    # a profile literally NAMED "gateway"
    "hermes -p gateway gateway run",
    "python -m hermes_cli.main --profile gateway gateway run",
    # quoted Windows paths with spaces (shlex-aware tokenization)
    r'"C:\Program Files\Hermes\hermes-gateway.exe"',
    r'"C:\Program Files\Hermes\gateway\run.py" run',
    r'"C:\Program Files\Py\pythonw.exe" -m hermes_cli.main gateway run',
]

WINDOWS_REDIRECTOR_SHIM_RUN = (
    r"C:\Users\me\hermes\venv\Scripts\python.exe -I -c "
    "import sys, runpy; "
    r"sys.argv = ['C:\\Users\\me\\hermes\\venv\\Scripts\\hermes.exe', "
    "'-p', 'worker', 'gateway', 'run']; "
    "runpy.run_module('hermes_cli.main', run_name='__main__')"
)

REJECT = [
    "python -m tui_gateway",                              # unrelated module
    "python -m hermes_cli.main gateway status",           # other subcommand
    "python -m hermes_cli.main gateway restart",
    "python -m hermes_cli.main gateway stop",
    "python -m hermes_cli.main --profile x dashboard",    # non-gateway subcommand
    "some random python -m mygateway thing",
    "",
    None,
]


@pytest.mark.parametrize("cmd", ACCEPT)
def test_accepts_real_gateway_run(cmd):
    assert matches(cmd) is True


def test_accepts_windows_redirector_python_c_sys_argv_gateway_run():
    assert matches(WINDOWS_REDIRECTOR_SHIM_RUN) is True
    assert matches_runtime(WINDOWS_REDIRECTOR_SHIM_RUN) is True


@pytest.mark.parametrize("cmd", REJECT)
def test_rejects_non_gateway_run_for_strict_matcher(cmd):
    assert matches(cmd) is False


@pytest.mark.parametrize("subcommand", ["status", "stop"])
def test_rejects_non_runtime_gateway_subcommands_inside_python_c_sys_argv(subcommand):
    command = WINDOWS_REDIRECTOR_SHIM_RUN.replace("'run'", f"'{subcommand}'")

    assert matches(command) is False
    assert matches_runtime(command) is False


def test_rejects_embedded_restart_as_strict_gateway_run():
    command = WINDOWS_REDIRECTOR_SHIM_RUN.replace("'run'", "'restart'")

    assert matches(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "python.exe -I -c import sys; sys.argv = ['hermes.exe', 'gateway', 'run']",
        (
            "not-python sys.argv = ['hermes.exe', 'gateway', 'run']; "
            "runpy.run_module('hermes_cli.main', run_name='__main__')"
        ),
        (
            "bash -c import sys, runpy; "
            "sys.argv = ['hermes.exe', 'gateway', 'run']; "
            "runpy.run_module('hermes_cli.main', run_name='__main__')"
        ),
        (
            "python.exe -I -c import sys, runpy; "
            "sys.argv = ['hermes.exe', 'gateway', 'run'; "
            "runpy.run_module('hermes_cli.main', run_name='__main__')"
        ),
        (
            "python.exe -I -c import sys, runpy; "
            "sys.argv = ['hermes.exe', 'gateway', 'run']; "
            "print('hermes_cli.main')"
        ),
    ],
)
def test_rejects_non_launcher_or_malformed_embedded_sys_argv(command):
    assert matches(command) is False
    assert matches_runtime(command) is False


