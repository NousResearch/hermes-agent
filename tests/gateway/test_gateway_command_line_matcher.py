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
    looks_like_pausable_hermes_process as matches_pausable,
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


# ---------------------------------------------------------------------------
# looks_like_pausable_hermes_process — the updater-stoppable backend matcher
#
# The Windows update preflight exempts processes the updater itself can stop:
# `gateway run` chains (paused by _pause_windows_gateways_for_update) and
# headless `serve` / `dashboard` backends (reaped by
# _kill_stale_dashboard_processes). A secondary profile's `serve` backend
# must be exempted; an operator REPL or stray script must not.
# ---------------------------------------------------------------------------

PAUSABLE_ACCEPT = [
    # every gateway-run chain accepted by the strict matcher
    "pythonw.exe -m hermes_cli.main gateway run",
    r"C:\Users\me\hermes\venv\Scripts\pythonw.exe -m hermes_cli.main gateway run",
    "python -m hermes_cli.main --profile work gateway run",
    "python -m hermes_cli.main gateway run --replace",
    "python -m hermes_cli/main.py gateway run",
    "python gateway/run.py",
    "hermes-gateway.exe",
    "hermes gateway",
    "hermes gateway run",
    "hermes gateway --profile work run",
    "python -m hermes_cli.main gateway -p work run",
    "hermes gateway --profile=work run",
    "hermes -p gateway gateway run",
    "python -m hermes_cli.main --profile gateway gateway run",
    r'"C:\Program Files\Hermes\hermes-gateway.exe"',
    r'"C:\Program Files\Hermes\gateway\run.py" run',
    r'"C:\Program Files\Py\pythonw.exe" -m hermes_cli.main gateway run',
    # serve / dashboard backends — the #81774 gap
    "python.exe -m hermes_cli.main serve --host 127.0.0.1 --port 8756",
    r"C:\Users\me\hermes\venv\Scripts\python.exe -m hermes_cli.main serve",
    "python -m hermes_cli.main --profile work serve",
    "python -m hermes_cli.main -p work serve --host 0.0.0.0",
    "python -m hermes_cli.main --profile=work serve",
    "python -m hermes_cli.main serve --profile work",
    "python -m hermes_cli.main dashboard --port 9119",
    "python -m hermes_cli.main --profile work dashboard",
    "python -m hermes_cli/main.py serve --host 127.0.0.1",
    # profile literally NAMED "serve"/"dashboard" must not shadow the token
    "python -m hermes_cli.main --profile serve serve",
    "python -m hermes_cli.main --profile dashboard dashboard",
    # case variations survive
    "PYTHON.EXE -m HERMES_CLI.MAIN SERVE --HOST 127.0.0.1",
]

PAUSABLE_REJECT = [
    # other hermes_cli.main subcommands are not long-lived backends
    "python -m hermes_cli.main gateway stop",
    "python -m hermes_cli.main gateway status",
    "python -m hermes_cli.main gateway install",
    "python -m hermes_cli.main chat",
    "python -m hermes_cli.main config",
    # nested tokens are NOT the dispatched subcommand
    "python -m hermes_cli.main mcp serve",
    "python -m hermes_cli.main webhook serve",
    # not a hermes_cli.main invocation
    "python -m tui_gateway",
    "python -m mygateway serve",
    "python myscript.py serve",
    "python -m some_other_cli serve --host 127.0.0.1",
    # operator REPL / stray script
    "python.exe",
    "python.exe -i",
    r"C:\x\venv\Scripts\python.exe",
    "",
    None,
]


@pytest.mark.parametrize("cmd", PAUSABLE_ACCEPT)
def test_accepts_pausable_hermes_backends(cmd):
    assert matches_pausable(cmd) is True


@pytest.mark.parametrize("cmd", PAUSABLE_REJECT)
def test_rejects_non_pausable(cmd):
    assert matches_pausable(cmd) is False


