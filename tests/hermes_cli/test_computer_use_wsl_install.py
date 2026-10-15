"""The selected Windows target must not run the Linux installer from WSL."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import tools_config_cua as install
from tools.computer_use import cua_backend_driver as driver


@pytest.fixture
def windows_target(monkeypatch):
    monkeypatch.setattr(install.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr('hermes_constants.is_wsl', lambda: True)
    monkeypatch.setattr(driver, 'computer_use_target', lambda: 'windows')
    monkeypatch.delenv('HERMES_CUA_DRIVER_CMD', raising=False)


def test_missing_windows_target_uses_windows_installer(windows_target, monkeypatch):
    monkeypatch.setattr(install, '_resolved_cua_driver_cmd', lambda: None)
    monkeypatch.setattr(install.shutil, 'which', lambda name: '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe' if name == 'powershell.exe' else None)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout='', stderr=''))
    monkeypatch.setattr(install, '_run_text', run)
    monkeypatch.setattr(install, '_cua_driver_contract_status', lambda *a: {'ready': True})
    monkeypatch.setattr(install, '_run_cua_driver_installer', Mock(side_effect=AssertionError('POSIX installer used')))
    assert install.install_cua_driver()
    argv = run.call_args.args[0]
    assert argv[0].endswith('powershell.exe')
    assert install._CUA_INSTALL_PS1_URL in argv[-1]
    assert '-NoAutoStart' not in argv[-1], 'standard mode needs the Windows daemon'
    assert run.call_args.kwargs['stdin'] == install.subprocess.DEVNULL


def test_unattended_wsl_update_never_downloads(windows_target, monkeypatch):
    monkeypatch.setattr(install, '_resolved_cua_driver_cmd', lambda: '/mnt/c/cua-driver.exe')
    monkeypatch.setattr(install, '_cua_driver_contract_status', lambda *a: {'ready': True})
    monkeypatch.setattr(install, '_run_text', Mock(side_effect=AssertionError('installer started')))
    assert install.install_cua_driver(upgrade=True, require_confirmed_update=True)


def test_wsl_override_conflict_does_not_install_elsewhere(windows_target, monkeypatch):
    monkeypatch.setenv('HERMES_CUA_DRIVER_CMD', '/usr/bin/linux-driver')
    monkeypatch.setattr(install, '_resolved_cua_driver_cmd', lambda: None)
    monkeypatch.setattr(driver, 'computer_use_target_error', lambda: 'override conflicts with windows target')
    monkeypatch.setattr(install, '_run_text', Mock(side_effect=AssertionError('installer started')))
    assert not install.install_cua_driver()


def test_wsl_installer_success_requires_selected_runtime_contract(windows_target, monkeypatch):
    monkeypatch.setattr(install, '_resolved_cua_driver_cmd', lambda: None)
    monkeypatch.setattr(install.shutil, 'which', lambda name: '/mnt/c/powershell.exe')
    monkeypatch.setattr(install, '_run_text', Mock(return_value=SimpleNamespace(returncode=0, stdout='', stderr='')))
    monkeypatch.setattr(install, '_cua_driver_contract_status', lambda *a: {'ready': False, 'reason': 'binary missing'})
    assert not install.install_cua_driver()
