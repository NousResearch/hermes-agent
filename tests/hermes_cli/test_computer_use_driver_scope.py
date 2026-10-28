"""CUA setup/status must agree with runtime selection about external ownership."""

from unittest.mock import Mock

import pytest

from agent import secret_scope


@pytest.fixture
def scoped_driver(tmp_path, monkeypatch):
    binary = tmp_path / "fake-driver.exe"
    binary.write_bytes(b"unused: resolution only\n")
    binary.chmod(0o755)
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    token = secret_scope.set_secret_scope(
        {"HERMES_CUA_DRIVER_CMD": str(binary)}, profile_home=str(tmp_path / "routed-profile"))
    try:
        yield str(binary)
    finally:
        secret_scope.reset_secret_scope(token)


def test_setup_preserves_profile_override(scoped_driver, monkeypatch, capsys):
    import pm
    from hermes_cli import tools_config_cua

    acquire = Mock()
    monkeypatch.setattr(pm, "ensure", acquire)
    monkeypatch.setattr(tools_config_cua, "_cua_driver_contract_status", lambda binary: {
        "ready": False, "reason": "invalid test manifest"})

    assert not tools_config_cua.install_cua_driver(show_installer_progress=False)
    acquire.assert_not_called()
    assert "Update the binary selected by HERMES_CUA_DRIVER_CMD" in capsys.readouterr().out
    assert tools_config_cua._cua_driver_cmd() == scoped_driver


def test_status_identifies_profile_override(scoped_driver, monkeypatch, capsys):
    from hermes_cli import tools_config_cua
    from hermes_cli.subcommands import computer_use

    monkeypatch.setattr(tools_config_cua, "_cua_driver_contract_status", lambda binary: {
        "ready": False, "reason": "invalid test manifest"})
    assert computer_use._cu_status(None) == 1
    output = capsys.readouterr().out
    assert scoped_driver in output
    assert "custom binary from HERMES_CUA_DRIVER_CMD" in output
    assert "Update the binary selected by HERMES_CUA_DRIVER_CMD" in output


def test_update_skips_profile_override(scoped_driver, monkeypatch):
    import pm
    from hermes_cli import update_cmd_maint

    installed = Mock(return_value=None)
    monkeypatch.setattr(pm, "installed_package", installed)
    monkeypatch.setattr(update_cmd_maint, "_load_updates_cfg", dict)
    update_cmd_maint._refresh_cua_driver_after_update()
    installed.assert_not_called()
