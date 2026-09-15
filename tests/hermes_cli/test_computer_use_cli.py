"""CLI coverage for the public Computer Use command surface."""

import subprocess
import sys
from importlib import import_module
from unittest.mock import Mock

import pytest

from tools.computer_use import cua_backend_driver


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "computer-use", *args],
        capture_output=True, text=True, timeout=30,
    )


def _invoke(monkeypatch, *args: str) -> int:
    cli_main = import_module("hermes_cli.main")
    monkeypatch.setattr(sys, "argv", ["hermes", "computer-use", *args])
    monkeypatch.setattr(cli_main, "_prepare_agent_startup", lambda _args: None)
    try:
        cli_main.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_computer_use_help_omits_browser_approve():
    result = _run("--help")
    assert result.returncode == 0
    assert "browser-approve" not in result.stdout
    assert "doctor" in result.stdout
    assert "permissions" in result.stdout


def test_computer_use_rejects_removed_browser_approve_command():
    result = _run("browser-approve", "--pid", "123")
    assert result.returncode == 2
    assert "invalid choice: 'browser-approve'" in result.stderr


def test_computer_use_status_reports_pm_without_polling_vendor(monkeypatch, capsys, tmp_path):
    from hermes_cli import tools_config_cua as cua

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: str(tmp_path / "cua-driver"))
    monkeypatch.setattr(cua, "_cua_driver_contract_status", lambda _binary=None: {"ready": True, "version": "0.20.0"})
    assert _invoke(monkeypatch, "status") == 0
    output = capsys.readouterr().out
    assert "Hermes PM" in output
    assert "latest" not in output


def test_computer_use_status_returns_nonzero_when_driver_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: None)
    assert _invoke(monkeypatch, "status") == 1
    assert "cua-driver: not installed" in capsys.readouterr().out


@pytest.mark.parametrize("override", [False, True])
def test_computer_use_status_reports_unusable_driver(monkeypatch, capsys, tmp_path, override):
    from hermes_cli import tools_config_cua as cua

    driver = str(tmp_path / "cua-driver")
    if override:
        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver)
    else:
        monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: driver)
    monkeypatch.setattr(cua, "_cua_driver_contract_status", lambda _binary=None: {
        "ready": False, "reason": "manifest is invalid",
    })
    assert _invoke(monkeypatch, "status") == 1
    output = capsys.readouterr().out
    assert "Repair required" in output
    if override:
        assert "custom binary from HERMES_CUA_DRIVER_CMD" in output
        assert "unset the override" in output
    else:
        assert "Run: hermes computer-use install" in output


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("upgrade", [False, True])
def test_computer_use_install_propagates_setup_result(monkeypatch, ready, upgrade):
    from hermes_cli import tools_config_cua as cua

    install = Mock(return_value=ready)
    monkeypatch.setattr(cua, "install_cua_driver", install)
    args = ("install", "--upgrade") if upgrade else ("install",)
    assert _invoke(monkeypatch, *args) == (0 if ready else 1)
    install.assert_called_once_with(upgrade=upgrade)