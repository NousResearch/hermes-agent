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
    _is_non_hermes_venv_holder,
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


# ---------------------------------------------------------------------------
# _is_non_hermes_venv_holder — the non-Hermes module exemption (#77422)
# ---------------------------------------------------------------------------
#
# A `python -m http.server 8077` (or any stdlib/user module) run from the
# venv interpreter holds the python.exe but does NOT import Hermes' native
# .pyd extensions — the actual reason the update preflight blocks venv
# holders. Reporting it as a blocker gives a misleading "Close other Hermes
# windows" error for a process that has nothing to do with Hermes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmdline",
    [
        # the exact issue case: operator's stdlib http.server on the venv python
        r"C:\Users\u\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -m http.server 8077",
        # another stdlib module
        r"C:\x\venv\Scripts\python.exe -m json.tool",
        # a user one-off module
        r"C:\x\venv\Scripts\python.exe -m my_script",
        # no Hermes marker anywhere in the venv-python cmdline
        r"C:\x\venv\Scripts\python.exe -m uvicorn app:main",
    ],
)
def test_is_non_hermes_venv_holder_accepts_stdlib_user_modules(cmdline: str) -> None:
    assert _is_non_hermes_venv_holder(cmdline) is True


@pytest.mark.parametrize(
    "cmdline",
    [
        # real Hermes invocations stay blockers
        "python.exe -m hermes_cli.main serve --host 127.0.0.1",
        "python.exe -m hermes_cli.main gateway run --replace",
        "python.exe -m hermes.main",
        # bare interpreter = REPL, conservative: keep blocking
        r"C:\x\venv\Scripts\python.exe",
        r"C:\x\venv\Scripts\python.exe -i",
    ],
)
def test_is_non_hermes_venv_holder_rejects_hermes_and_repl(cmdline: str) -> None:
    assert _is_non_hermes_venv_holder(cmdline) is False


def test_main_exempts_non_hermes_module_holders(monkeypatch, capsys):
    """A venv-python `-m http.server` must NOT block the update; a real
    Hermes serve backend alongside it still does."""
    http_server = (
        91,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m http.server 8077",
    )
    serve = (
        78,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main serve --host 127.0.0.1",
    )
    # http.server alone → clear
    code, data = _run_main_with_detector(monkeypatch, capsys, [http_server])
    assert code == 0
    assert data["blocked"] is False
    assert data["processes"] == []
    assert data["non_hermes_holders"] == 1

    # http.server + real Hermes serve backend → blocked, only the serve PID
    code, data = _run_main_with_detector(monkeypatch, capsys, [http_server, serve])
    assert code == 0
    assert data["blocked"] is True
    assert [p["pid"] for p in data["processes"]] == [78]
    assert data["non_hermes_holders"] == 1