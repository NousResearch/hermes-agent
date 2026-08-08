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
    "python3.11 -m hermes_cli.main gateway run",
    "pythonw3.11.exe -m hermes_cli.main gateway run",
    "python -m hermes_cli/main.py gateway run",
    "python hermes_cli/main.py gateway run",
    "/opt/hermes/venv/bin/python /opt/hermes/venv/bin/hermes gateway run",
    "python gateway/run.py",
    "hermes-gateway.exe",
    "hermes gateway",          # bare `hermes gateway` defaults to run
    "hermes gateway run",
    "HERMES_HOME=/opt/data/profiles/work hermes gateway run",
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

REJECT = [
    "python -m tui_gateway",                              # unrelated module
    "python -m hermes_cli.main gateway status",           # other subcommand
    "python -m hermes_cli.main gateway restart",
    "python -m hermes_cli.main gateway stop",
    "python -m hermes_cli.main --profile x dashboard",    # non-gateway subcommand
    "some random python -m mygateway thing",
    # Entry-point-like text in ordinary script arguments is not process
    # identity. These holders must not be exempted from the update guard.
    "python.exe myscript.py hermes_cli.main gateway run",
    "python.exe -c print hermes_cli.main gateway run",
    "python.exe worker.py --note hermes_cli.main gateway run",
    r"python.exe worker.py C:\\tools\\hermes.exe gateway run",
    "python.exe worker.py gateway/run.py",
    "hermes chat gateway run",
    "python -m hermes_cli.main chat gateway run",
    "python hermes_cli/main.py dashboard gateway run",
    "python /opt/hermes/venv/bin/not-hermes gateway run",
    "",
    None,
]


@pytest.mark.parametrize("cmd", ACCEPT)
def test_accepts_real_gateway_run(cmd):
    assert matches(cmd) is True


@pytest.mark.parametrize("cmd", REJECT)
def test_rejects_non_gateway_commands(cmd):
    assert matches(cmd) is False


@pytest.mark.parametrize(
    "cmd",
    [
        "hermes gateway restart",
        "python -m hermes_cli.main gateway restart",
        "/opt/hermes/venv/bin/python /opt/hermes/venv/bin/hermes gateway restart",
    ],
)
def test_runtime_matcher_accepts_restart_without_calling_it_run(cmd):
    assert matches_runtime(cmd) is True
    assert matches(cmd) is False


@pytest.mark.parametrize(
    "cmd",
    [
        "hermes gateway status",
        "hermes gateway stop",
        "hermes dashboard gateway restart",
    ],
)
def test_runtime_matcher_rejects_non_runtime_commands(cmd):
    assert matches_runtime(cmd) is False


