"""CLI coverage for the public Computer Use command surface."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from unittest.mock import Mock

import pytest

from tools.computer_use import cua_backend_driver


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "computer-use", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _invoke(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """Run the in-process CLI and normalize its process-style exit status."""
    cli_main = import_module("hermes_cli.main")
    monkeypatch.setattr(sys, "argv", ["hermes", "computer-use", *args])
    monkeypatch.setattr(cli_main, "_prepare_agent_startup", lambda _args: None)
    try:
        cli_main.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_computer_use_help_omits_browser_approve() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "browser-approve" not in result.stdout
    assert "doctor" in result.stdout
    assert "permissions" in result.stdout


def test_computer_use_rejects_removed_browser_approve_command() -> None:
    result = _run("browser-approve", "--pid", "123")

    assert result.returncode == 2
    assert "invalid choice: 'browser-approve'" in result.stderr


def test_computer_use_status_returns_zero_for_compatible_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import tools_config
    import hermes_cli.tools_config_cua as tools_config_cua

    driver = r"C:\Users\tester\.local\bin\cua-driver.exe"
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: driver)
    monkeypatch.setattr(
        tools_config,
        "_cua_driver_contract_status",
        lambda _binary=None: {"ready": True},
    )
    monkeypatch.setattr(
        tools_config_cua,
        "_cua_driver_contract_status",
        lambda _binary=None: {"ready": True},
    )
    monkeypatch.setattr(
        cua_backend_driver,
        "cua_driver_update_check",
        lambda: {"update_available": False},
    )

    assert _invoke(monkeypatch, "status") == 0


def test_computer_use_status_returns_nonzero_when_driver_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: None)

    assert _invoke(monkeypatch, "status") == 1
    assert "cua-driver: not installed" in capsys.readouterr().out


def test_computer_use_status_returns_nonzero_for_incompatible_standard_driver(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli import tools_config
    import hermes_cli.tools_config_cua as tools_config_cua

    driver = r"C:\Users\tester\.local\bin\cua-driver.exe"
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: driver)
    monkeypatch.setattr(
        tools_config,
        "_cua_driver_contract_status",
        lambda _binary=None: {
            "ready": False,
            "reason": "required runtime features are missing",
        },
    )
    monkeypatch.setattr(
        tools_config_cua,
        "_cua_driver_contract_status",
        lambda _binary=None: {
            "ready": False,
            "reason": "required runtime features are missing",
        },
    )

    assert _invoke(monkeypatch, "status") == 1
    output = capsys.readouterr().out
    assert "Repair required" in output
    assert "Run: hermes computer-use install" in output


def test_computer_use_status_returns_nonzero_for_incompatible_custom_driver(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli import tools_config
    import hermes_cli.tools_config_cua as tools_config_cua

    driver = r"C:\custom\cmd.exe"
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: driver)
    monkeypatch.setattr(
        tools_config,
        "_cua_driver_contract_status",
        lambda _binary=None: {"ready": False, "reason": "manifest is invalid"},
    )
    monkeypatch.setattr(
        tools_config_cua,
        "_cua_driver_contract_status",
        lambda _binary=None: {"ready": False, "reason": "manifest is invalid"},
    )

    assert _invoke(monkeypatch, "status") == 1
    output = capsys.readouterr().out
    assert "custom binary from HERMES_CUA_DRIVER_CMD" in output
    assert "unset the override" in output


@pytest.mark.parametrize(("ready", "expected"), [(True, 0), (False, 1)])
def test_computer_use_install_checks_resulting_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
    ready: bool,
    expected: int,
) -> None:
    from hermes_cli import tools_config
    import hermes_cli.tools_config_cua as tools_config_cua

    install = Mock(return_value=True)
    monkeypatch.setattr(tools_config, "install_cua_driver", install)
    monkeypatch.setattr(
        tools_config,
        "_cua_driver_contract_status",
        lambda: {"ready": ready},
    )
    monkeypatch.setattr(
        tools_config_cua,
        "_cua_driver_contract_status",
        lambda: {"ready": ready},
    )

    assert _invoke(monkeypatch, "install") == expected
    install.assert_called_once_with(upgrade=False)


def test_computer_use_install_returns_nonzero_for_unrepairable_custom_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import tools_config
    import hermes_cli.tools_config_cua as tools_config_cua

    driver = r"C:\custom\cmd.exe"
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver)
    install = Mock(return_value=False)
    contract = Mock(side_effect=AssertionError("failed install must short-circuit"))
    monkeypatch.setattr(tools_config, "install_cua_driver", install)
    monkeypatch.setattr(tools_config, "_cua_driver_contract_status", contract)
    monkeypatch.setattr(tools_config_cua, "_cua_driver_contract_status", contract)

    assert _invoke(monkeypatch, "install") == 1
    install.assert_called_once_with(upgrade=False)
    contract.assert_not_called()
