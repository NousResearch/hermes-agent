"""Tests for hermes_cli/_scan_venv_blockers.py.

Tests call the real production functions (``main``, ``_redact_sensitive_cmdline``).
The detector is patched directly so no real process table interaction occurs.
"""

from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.redact as redact_module
from hermes_cli._scan_venv_blockers import (
    _is_pausable_gateway,
    _redact_sensitive_cmdline,
    main,
)


# ---------------------------------------------------------------------------
# main() — stdout, stderr, exit code (with patched detector)
# ---------------------------------------------------------------------------


def _psutil_fake() -> dict:
    """Return a sys.modules dict entry that makes psutil appear available."""
    return {"psutil": types.SimpleNamespace(Process=lambda *a: MagicMock())}


def _psutil_fake_with_argv(argv_by_pid: dict) -> dict:
    """psutil fake whose Process(pid).cmdline() returns a live argv list.

    Simulates the real psutil re-read the scanner performs on the captured
    (possibly truncated) cmdline before the gateway exemption runs.
    """

    class _Proc:
        def __init__(self, pid: int) -> None:
            self._pid = pid

        def cmdline(self):
            return argv_by_pid.get(self._pid)

    return {"psutil": types.SimpleNamespace(Process=_Proc)}






# ---------------------------------------------------------------------------
# _redact_sensitive_cmdline
# ---------------------------------------------------------------------------


def test_redact_long_flag_value_space_separated() -> None:
    """--token SECRET must preserve --token and emit --token <redacted>."""
    raw = "python.exe -m hermes_cli.main serve --token ghp_abc123 --host 10.0.0.1"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe -m hermes_cli.main serve --token <redacted>"
    assert "ghp_abc123" not in result




def test_redact_sensitive_text_failure_returns_fully_redacted() -> None:
    """When agent.redact.redact_sensitive_text raises, the entire result
    must equal '<redacted>' so PID and name still provide diagnostics."""
    with patch.object(
        redact_module,
        "redact_sensitive_text",
        side_effect=RuntimeError("no redactor"),
    ):
        result = _redact_sensitive_cmdline("python.exe --token abc123")

    assert result == "<redacted>"


def test_redact_session_key() -> None:
    """--session-key <identifier> must redact the value and everything after."""
    raw = "python.exe -m tui_gateway.slash_worker --session-key 20260712-abcdef --model test"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe -m tui_gateway.slash_worker --session-key <redacted>"


def test_redact_normal_host_port_profile_remain() -> None:
    raw = "python.exe -m hermes_cli.main serve --host 10.0.0.1 --port 9119 --profile work"
    result = _redact_sensitive_cmdline(raw)
    assert "10.0.0.1" in result
    assert "9119" in result
    assert "work" in result


def test_redact_no_sensitive_flags_is_noop() -> None:
    raw = "python.exe -m hermes_cli.main serve --host 127.0.0.1"
    assert _redact_sensitive_cmdline(raw) == raw


def test_redact_empty_string() -> None:
    assert _redact_sensitive_cmdline("") == ""


def test_redact_short_flags_not_redacted() -> None:
    """Short flags -t (toolset), -p (profile), -k are NOT redacted."""
    raw = "python.exe -m hermes_cli.main serve -t web -p default -k somearg"
    result = _redact_sensitive_cmdline(raw)
    assert result == raw  # short flags pass through unchanged


# ---------------------------------------------------------------------------
# _is_pausable_gateway — the gateway exemption
#
# `hermes-setup` always invokes `hermes update --yes --gateway`, whose
# `_pause_windows_gateways_for_update()` stops running gateways itself. The
# Desktop preflight must therefore not report gateway launcher/worker chains
# as blockers — doing so aborts the handoff before the component that can
# handle them ever runs.
# ---------------------------------------------------------------------------


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
        # a profile literally NAMED "gateway" — the profile value must not
        # shadow the subcommand token (the hand-rolled matcher regressed this)
        "python.exe -m hermes_cli.main --profile gateway gateway run",
        "python.exe -m hermes_cli.main -p gateway gateway run",
        # bare `gateway` defaults to `run` (mirrors the canonical matcher)
        "python.exe -m hermes_cli.main gateway",
        # case variations survive
        "PYTHON.EXE -m hermes_cli.main GATEWAY RUN",
    ],
)
def test_is_pausable_gateway_accepts_gateway_run_chains(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is True


@pytest.mark.parametrize(
    "cmdline",
    [
        # desktop backend: no pause machinery downstream, must keep blocking
        "python.exe -m hermes_cli.main serve --host 127.0.0.1 --port 8756",
        # other gateway subcommands are not running gateways
        "python.exe -m hermes_cli.main gateway stop",
        "python.exe -m hermes_cli.main gateway status",
        "python.exe -m hermes_cli.main gateway install",
        # operator REPL / stray script
        "python.exe",
        "python.exe myscript.py gateway run",  # not a hermes_cli.main invocation
        "",
    ],
)
def test_is_pausable_gateway_rejects_everything_else(cmdline: str) -> None:
    assert _is_pausable_gateway(cmdline) is False


def _run_main_with_detector(monkeypatch, capsys, matches):
    """Run main() with the process detector patched to return *matches*."""
    for name, mod in _psutil_fake().items():
        monkeypatch.setitem(sys.modules, name, mod)
    import hermes_cli.main as cli_main

    monkeypatch.setattr(cli_main, "_detect_venv_python_processes", lambda: matches)
    with pytest.raises(SystemExit) as excinfo:
        main()
    out = capsys.readouterr().out
    return excinfo.value.code, json.loads(out)


def test_main_exempts_gateway_chain_but_keeps_other_holders(monkeypatch, capsys):
    """A gateway launcher/worker pair alone must scan clear; a non-gateway
    holder alongside it must still block (and be the only reported PID)."""
    gateway_launcher = (
        12,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main gateway run --replace",
    )
    gateway_worker = (
        34,
        "python.exe",
        r'"C:\u\uv\python\python.exe"  -m hermes_cli.main gateway run --replace',
    )
    stray_repl = (56, "python.exe", r"C:\x\venv\Scripts\python.exe")

    # Gateway chain only → clear
    code, data = _run_main_with_detector(
        monkeypatch, capsys, [gateway_launcher, gateway_worker]
    )
    assert code == 0
    assert data["ok"] is True
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["pausable_gateways"] == 2

    # Gateway chain + stray REPL → blocked, reporting only the REPL
    code, data = _run_main_with_detector(
        monkeypatch, capsys, [gateway_launcher, gateway_worker, stray_repl]
    )
    assert code == 0
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [56]
    assert data["pausable_gateways"] == 2


def test_main_desktop_serve_backend_still_blocks(monkeypatch, capsys):
    """The desktop's own `serve` backend has no downstream pause — it must
    keep blocking exactly as before the exemption."""
    serve = (
        78,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1",
    )
    code, data = _run_main_with_detector(monkeypatch, capsys, [serve])
    assert code == 0
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [78]
    assert data["pausable_gateways"] == 0


def test_main_rereads_truncated_argv_before_gateway_exemption(monkeypatch, capsys):
    """A uv-side gateway worker whose captured cmdline was truncated at 120
    chars (the managed-CPython path alone exceeds the budget) must still be
    exempted: the scanner re-reads the live argv via psutil before running
    the gateway matcher. Regression for the Desktop ``venv-blocked`` dead-end
    where the reported holder is a gateway the updater would pause itself.
    """
    worker_pid = 3401
    truncated_worker = (
        worker_pid,
        "python.exe",
        # The 120-char prefix captured by _detect_venv_python_processes;
        # "gateway run" lies beyond the truncation point.
        r"E:\hermes\hermes-agent\.hermes-runtime\python\generation-1785786039-27004-e71ff1af\cpython-3.11-windows-x86_64-none\pyth",
    )
    full_argv = [
        r"E:\hermes\hermes-agent\.hermes-runtime\python\generation-1785786039-27004-e71ff1af\cpython-3.11-windows-x86_64-none\python.exe",
        "-m",
        "hermes_cli.main",
        "gateway",
        "run",
    ]

    for name, mod in _psutil_fake_with_argv({worker_pid: full_argv}).items():
        monkeypatch.setitem(sys.modules, name, mod)
    import hermes_cli.main as cli_main

    monkeypatch.setattr(cli_main, "_detect_venv_python_processes", lambda: [truncated_worker])

    with pytest.raises(SystemExit) as excinfo:
        main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert excinfo.value.code == 0
    assert data["ok"] is True
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["pausable_gateways"] == 1


def test_main_truncated_gateway_worker_without_live_argv_falls_back(
    monkeypatch, capsys
):
    """When psutil cannot re-read the live argv (dead PID, access denied),
    the scanner falls back to the captured cmdline — a truncated gateway
    worker then still counts as a blocker, preserving pre-fix behavior.
    """
    truncated_worker = (
        3402,
        "python.exe",
        r"E:\hermes\hermes-agent\.hermes-runtime\python\generation-1785786039-27004-e71ff1af\cpython-3.11-windows-x86_64-none\pyth",
    )

    # Default fake: Process(pid).cmdline() returns a MagicMock, so the join
    # fails and the captured (truncated) cmdline is used.
    for name, mod in _psutil_fake().items():
        monkeypatch.setitem(sys.modules, name, mod)
    import hermes_cli.main as cli_main

    monkeypatch.setattr(cli_main, "_detect_venv_python_processes", lambda: [truncated_worker])

    with pytest.raises(SystemExit) as excinfo:
        main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert excinfo.value.code == 0
    assert data["ok"] is True
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [3402]
    assert data["pausable_gateways"] == 0