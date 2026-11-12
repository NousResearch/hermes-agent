"""Updates reconcile owned CUA pins without bypassing native host setup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import pm
from hermes_cli import tools_config_cua as setup
from hermes_cli import update_cmd_maint as update


@pytest.fixture
def refresh(monkeypatch):
    monkeypatch.setattr(update, "_load_updates_cfg", lambda: {"refresh_cua_driver": True})
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setenv("PATH", "")
    installed = Mock(return_value=SimpleNamespace())
    ensure = Mock()
    monkeypatch.setattr(pm, "installed_package", installed)
    monkeypatch.setattr(pm, "ensure", ensure)
    return installed, ensure


@pytest.mark.parametrize("disabled,installed,override", [
    (True, True, ""),
    (False, False, ""),
    (False, True, "custom-cua"),
])
def test_cua_refresh_skips_disabled_missing_or_external_driver(
    refresh, monkeypatch, disabled, installed, override,
):
    lookup, ensure = refresh
    monkeypatch.setattr(update, "_load_updates_cfg", lambda: {"refresh_cua_driver": not disabled})
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", override)
    lookup.return_value = SimpleNamespace() if installed else None
    host_setup = Mock(side_effect=AssertionError("unexpected host setup"))
    monkeypatch.setattr(setup, "install_cua_driver", host_setup)

    update._refresh_cua_driver_after_update()

    ensure.assert_not_called()
    host_setup.assert_not_called()


@pytest.mark.platforms("linux")
def test_linux_refresh_reconciles_installed_pin(refresh):
    lookup, ensure = refresh
    update._refresh_cua_driver_after_update()
    lookup.assert_called_once_with("cua-driver", allow_outdated=True)
    ensure.assert_called_once_with("cua-driver", explicit=True)


@pytest.mark.platforms("windows")
def test_windows_refresh_defers_pin_and_uac_to_explicit_setup(refresh, monkeypatch, capsys):
    """Keep the task's versioned executable selected until interactive re-registration."""
    _, ensure = refresh
    host_setup = Mock(side_effect=AssertionError("unattended UAC"))
    monkeypatch.setattr(setup, "install_cua_driver", host_setup)

    update._refresh_cua_driver_after_update()

    ensure.assert_not_called()
    host_setup.assert_not_called()
    output = capsys.readouterr().out
    assert "deferred" in output.lower()
    assert "UAC" in output
    assert "hermes computer-use install --upgrade" in output


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("registration_exit", [0, 1])
def test_macos_refresh_registers_post_ensure_signed_app(
    refresh, monkeypatch, tmp_path, capsys, registration_exit,
):
    """Exercise native setup dispatch; no real LaunchServices or codesign writes."""
    from tools.computer_use import cua_backend_daemon as daemon

    lookup, ensure = refresh
    old = tmp_path / "old-cua-driver"
    app = tmp_path / "new-pin" / "CuaDriver.app"
    binary = app / "Contents" / "MacOS" / "cua-driver"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    lookup.return_value = SimpleNamespace(binary=old)

    def select_new(*args, **kwargs):
        lookup.return_value = SimpleNamespace(binary=binary)

    ensure.side_effect = select_new
    contract = Mock(return_value={"ready": True})
    monkeypatch.setattr(setup, "_cua_driver_contract_status", contract)
    validate = Mock()
    monkeypatch.setattr(daemon, "_validate_cua_driver_app_signature", validate)
    register = Mock(return_value=SimpleNamespace(returncode=registration_exit, stdout="", stderr=""))
    monkeypatch.setattr(setup, "_run_text", register)

    update._refresh_cua_driver_after_update()

    ensure.assert_called_once_with("cua-driver", explicit=True)
    contract.assert_called_once_with(str(binary))
    validate.assert_called_once_with(str(app))
    assert register.call_args.args[0][-2:] == ["-f", str(app)]
    assert ("app registration failed" in capsys.readouterr().out) == bool(registration_exit)
