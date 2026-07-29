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
    _redact_sensitive_cmdline,
    main,
)


# ---------------------------------------------------------------------------
# main() — stdout, stderr, exit code (with patched detector)
# ---------------------------------------------------------------------------


def _psutil_fake() -> dict:
    """Return a sys.modules dict entry that makes psutil appear available."""
    return {"psutil": types.SimpleNamespace(Process=lambda *a: MagicMock())}


def test_main_no_holders_prints_clear_json(tmp_path: Path, capsys) -> None:
    from hermes_cli import main as cli_main

    fake_detect = MagicMock(return_value=[])
    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.object(cli_main, "_detect_venv_python_processes", fake_detect), patch.dict(
        sys.modules, _psutil_fake()
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {"ok": True, "blocked": False, "processes": []}


def test_main_holders_prints_blocked_json(tmp_path: Path, capsys) -> None:
    from hermes_cli import main as cli_main

    fake_detect = MagicMock(
        return_value=[(101, "python.exe", "python.exe -m hermes_cli.main serve --host 10.0.0.1")]
    )
    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.object(cli_main, "_detect_venv_python_processes", fake_detect), patch.dict(
        sys.modules, _psutil_fake()
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ok"] is True
    assert data["blocked"] is True
    assert len(data["processes"]) == 1
    p = data["processes"][0]
    assert p["pid"] == 101
    assert p["name"] == "python.exe"
    assert "serve" in p["cmdline"]


def test_main_detector_exception_exits_nonzero(tmp_path: Path, capsys) -> None:
    from hermes_cli import main as cli_main

    with patch.object(
        cli_main, "_detect_venv_python_processes", side_effect=RuntimeError("boom")
    ), patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.dict(sys.modules, _psutil_fake()):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code != 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {"ok": False, "blocked": False, "processes": []}
    assert "boom" in captured.err


def test_main_psutil_unavailable_exits_nonzero(tmp_path: Path, capsys) -> None:
    from hermes_cli import main as cli_main

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.dict(sys.modules, {"psutil": None}):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code != 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {"ok": False, "blocked": False, "processes": []}


def test_main_import_hermes_cli_main_fails(tmp_path: Path, capsys) -> None:
    """When the import of hermes_cli.main raises, main() must produce one
    parseable ok=false JSON on stdout, the diagnostic on stderr, and exit
    non-zero."""
    from hermes_cli import main as cli_main

    real_import = builtins.__import__

    def selective_import(name, *args, **kwargs):
        if name == "hermes_cli.main":
            raise ImportError("detector import failed")
        return real_import(name, *args, **kwargs)

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.dict(sys.modules, _psutil_fake()), patch.object(
        builtins, "__import__", selective_import
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code != 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data == {"ok": False, "blocked": False, "processes": []}
    assert "detector import failed" in captured.err


# ---------------------------------------------------------------------------
# _redact_sensitive_cmdline
# ---------------------------------------------------------------------------


def test_redact_long_flag_value_space_separated() -> None:
    """--token SECRET must preserve --token and emit --token <redacted>."""
    raw = "python.exe -m hermes_cli.main serve --token ghp_abc123 --host 10.0.0.1"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe -m hermes_cli.main serve --token <redacted>"
    assert "ghp_abc123" not in result


def test_redact_long_flag_equals_form() -> None:
    """--api-key=SECRET must preserve --api-key= and emit --api-key=<redacted>."""
    raw = "python.exe --api-key=sk-1234567890abcdef serve"
    result = _redact_sensitive_cmdline(raw)
    assert result == "python.exe --api-key=<redacted>"
    assert "sk-1234567890abcdef" not in result


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