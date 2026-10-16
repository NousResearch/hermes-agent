"""The retired scanner's historical import and desktop CLI contracts."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli._scan_venv_blockers import _is_pausable_gateway


@pytest.mark.parametrize(
    "cmdline",
    [
        # venv-side launcher, exactly as the scheduled task spawns it
        r"C:\Users\u\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
        " -m hermes_cli.main gateway run --replace",
        # uv-side worker re-running the same argv (quoted exe, double space)
        r'"C:\Users\u\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe"'
        "  -m hermes_cli.main gateway run --replace",
        # profile-scoped gateway
        "python.exe -m hermes_cli.main --profile work gateway run",
        # A profile named gateway must not shadow the subcommand token.
        "python.exe -m hermes_cli.main --profile gateway gateway run",
        "python.exe -m hermes_cli.main -p gateway gateway run",
        # bare gateway defaults to run
        "python.exe -m hermes_cli.main gateway",
        "PYTHON.EXE -m hermes_cli.main GATEWAY RUN",
    ],
)
def test_is_pausable_gateway_accepts_gateway_run_chains(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is True


@pytest.mark.parametrize(
    "cmdline",
    [
        # Desktop backends are not messaging gateways.
        "python.exe -m hermes_cli.main serve --host 127.0.0.1 --port 8756",
        "python.exe -m hermes_cli.main gateway stop",
        "python.exe -m hermes_cli.main gateway status",
        "python.exe -m hermes_cli.main gateway install",
        "python.exe",
        "python.exe myscript.py gateway run",
        "",
    ],
)
def test_is_pausable_gateway_rejects_everything_else(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is False


def test_is_pausable_gateway_import_failure_fails_closed(monkeypatch):
    """An old updater can import this helper on a partially replaced tree."""
    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "gateway.status":
            raise ImportError("gateway unavailable during checkout swap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert _is_pausable_gateway("python -m hermes_cli.main gateway run") is False


def _run_legacy_cli(tmp_path, *args):
    # -S omits site-packages, including psutil: this entry point must work on
    # a half-updated tree, without inspecting or terminating live processes.
    return subprocess.run(
        [sys.executable, "-S", "-m", "hermes_cli._scan_venv_blockers", *args],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_legacy_cli_is_dependency_free_and_informational(tmp_path):
    result = _run_legacy_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    # These three fields are consumed by historical Desktop's strict parser.
    assert data["ok"] is True
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["retired"] is True
    assert data["message"]
    assert result.stderr == ""


@pytest.mark.parametrize("identity", [("123", "1722798000.25"), (), ("invalid", "nan")])
def test_legacy_termination_is_refused(tmp_path, identity):
    result = _run_legacy_cli(tmp_path, "--terminate-safe", *identity)
    # Historical Desktop treats exit 0 as proof the PID was stopped, without
    # parsing stdout. Never acknowledge a termination that did not happen.
    assert result.returncode != 0
    data = json.loads(result.stdout)
    assert data["ok"] is False
    assert "retired" in data["error"].lower()
    assert result.stderr == ""