"""CLI coverage for the public Computer Use command surface."""

import subprocess
import sys
from importlib import import_module
from unittest.mock import Mock

import pytest

from tools.computer_use import cua_backend_driver




def _invoke(monkeypatch, *args: str) -> int:
    cli_main = import_module("hermes_cli.main")
    monkeypatch.setattr(sys, "argv", ["hermes", "computer-use", *args])
    monkeypatch.setattr(cli_main, "_prepare_agent_startup", lambda _args: None)
    try:
        cli_main.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0






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


def test_permissions_status_names_the_stale_tcc_row_for_the_missing_grant(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A grant the daemon reports missing while System Settings shows it ON is a stale TCC row (trycua/cua#3170);
    the status output must name the reset for exactly the missing service, never for one that is granted."""
    from tools.computer_use import permissions

    status = {"platform": "darwin", "platform_supported": True, "installed": True, "version": "cua-driver 0.28.2",
              "ready": False, "can_grant": True, "checks": [], "source": None, "error": None,
              "accessibility": False, "screen_recording": True, "screen_recording_capturable": True}
    monkeypatch.setattr(permissions, "computer_use_status", lambda driver_cmd=None: status)

    assert _invoke(monkeypatch, "permissions", "status") == 1
    out = capsys.readouterr().out
    assert "tccutil reset Accessibility com.trycua.driver" in out
    assert "ScreenCapture" not in out
    assert "hermes computer-use permissions grant" in out
