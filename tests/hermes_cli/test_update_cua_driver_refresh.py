"""Computer Use refresh behavior after a confirmed Hermes update."""

from unittest.mock import Mock

import pytest

from hermes_cli import update_cmd_maint


@pytest.mark.linux_only
class TestCuaDriverRefreshAfterUpdate:
    def test_canonical_local_bin_resolution_does_not_require_path(self, tmp_path, monkeypatch):
        from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd

        driver = tmp_path / ".local" / "bin" / "cua-driver"
        driver.parent.mkdir(parents=True)
        driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        driver.chmod(0o755)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PATH", "")
        monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)

        assert resolve_cua_driver_cmd() == str(driver)

    def test_enabled_resolved_driver_refreshes_with_confirmed_update_only(self, monkeypatch):
        from hermes_cli import tools_config
        from tools.computer_use import cua_backend_driver

        install = Mock()
        monkeypatch.setattr(update_cmd_maint, "_load_updates_cfg", lambda: {})
        monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: "/tmp/cua-driver")
        monkeypatch.setattr(tools_config, "install_cua_driver", install)

        update_cmd_maint._refresh_cua_driver_after_update()

        install.assert_called_once_with(
            upgrade=True,
            require_confirmed_update=True,
            show_installer_progress=False,
        )

    def test_disabled_refresh_does_not_resolve_or_install(self, monkeypatch):
        from hermes_cli import tools_config
        from tools.computer_use import cua_backend_driver

        resolve = Mock(return_value="/tmp/cua-driver")
        install = Mock()
        monkeypatch.setattr(update_cmd_maint, "_load_updates_cfg", lambda: {"refresh_cua_driver": False})
        monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", resolve)
        monkeypatch.setattr(tools_config, "install_cua_driver", install)

        update_cmd_maint._refresh_cua_driver_after_update()

        resolve.assert_not_called()
        install.assert_not_called()

    def test_unresolved_driver_does_not_install(self, monkeypatch):
        from hermes_cli import tools_config
        from tools.computer_use import cua_backend_driver

        install = Mock()
        monkeypatch.setattr(update_cmd_maint, "_load_updates_cfg", lambda: {})
        monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: None)
        monkeypatch.setattr(tools_config, "install_cua_driver", install)

        update_cmd_maint._refresh_cua_driver_after_update()

        install.assert_not_called()
