"""The selected Windows target must not run the Linux installer from WSL."""
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import tools_config_cua as install
from tools.computer_use import cua_backend_driver as driver


pytestmark = pytest.mark.linux_only


@pytest.fixture
def windows_target(monkeypatch):
    monkeypatch.setattr('hermes_constants.is_wsl', lambda: True)
    monkeypatch.setattr(driver, 'computer_use_target', lambda: 'windows')
    monkeypatch.delenv('HERMES_CUA_DRIVER_CMD', raising=False)


def test_missing_windows_target_uses_windows_installer(windows_target, monkeypatch):
    monkeypatch.setattr(install, '_resolved_cua_driver_cmd', Mock(side_effect=[None, '/mnt/c/cua-driver.exe']))
    monkeypatch.setattr(install, '_cua_driver_autostart_registered_windows', lambda: True)
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
    monkeypatch.setattr(install, '_cua_driver_autostart_registered_windows', lambda: True)
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


@pytest.mark.parametrize("task_after_repair", [True, False])
@pytest.mark.parametrize("fresh_install", [False, True])
def test_wsl_readiness_and_install_require_host_autostart(
    windows_target, monkeypatch, tmp_path, task_after_repair, fresh_install
):
    """Status, compatible-install repair and fresh installation use one host contract."""
    exe = tmp_path / "cua-driver.exe"
    exe.write_text("test driver")
    exe.chmod(0o755)
    native_path = r"D:\Users\Alice O'Brien\Cua\cua-driver.exe"
    state = {"installed": not fresh_install, "task": False}
    calls = []

    def which(name):
        if name in ("powershell.exe", "wslpath"):
            return name
        return str(exe) if state["installed"] and name == str(exe) else None

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "schtasks.exe":
            return SimpleNamespace(returncode=0 if state["task"] else 1)
        if argv[0] == "wslpath":
            assert argv == ["wslpath", "-w", str(exe.resolve())]
            return SimpleNamespace(returncode=0, stdout=native_path, stderr="")
        assert argv[0] == "powershell.exe", argv
        script = argv[-1]
        if install._CUA_INSTALL_PS1_URL in script:
            assert fresh_install
            state["installed"] = True
        else:
            assert "-FilePath $exe" in script
            assert "-ArgumentList @('autostart','enable')" in script
            assert install._ps_single_quote(native_path) in script
            assert str(exe) not in script
            state["task"] = task_after_repair
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(install.shutil, "which", which)
    monkeypatch.setattr(install.subprocess, "run", run)
    monkeypatch.setattr(install, "_resolved_cua_driver_cmd", lambda: str(exe) if state["installed"] else None)
    monkeypatch.setattr(install, "_cua_driver_contract_status", lambda *args: {"ready": state["installed"]})
    assert not install._cua_driver_install_ready()
    assert install.install_cua_driver() is task_after_repair
    assert install._cua_driver_install_ready() is task_after_repair
    assert sum(install._CUA_INSTALL_PS1_URL in argv[-1] for argv, _ in calls) == int(fresh_install)
    # A Linux guest backend does not depend on a Windows scheduled task.
    monkeypatch.setattr(install, "_resolved_cua_driver_cmd", lambda: "/usr/bin/cua-driver")
    assert install._cua_driver_install_ready()


@pytest.mark.parametrize("failure", ["missing-task", "probe-timeout", "repair-denied", "repair-timeout"])
def test_wsl_autostart_failure_never_reports_success_or_elevates_unattended(
    windows_target, monkeypatch, tmp_path, failure
):
    exe = tmp_path / "cua-driver.exe"
    exe.write_text("driver")
    exe.chmod(0o755)
    calls = []
    monkeypatch.setattr(install, "_resolved_cua_driver_cmd", lambda: str(exe))
    monkeypatch.setattr(install, "_cua_driver_contract_status", lambda *args: {"ready": True})
    monkeypatch.setattr(install.shutil, "which", lambda name: str(exe) if name == str(exe) else name)

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "schtasks.exe":
            if failure == "probe-timeout":
                raise subprocess.TimeoutExpired(argv, 10)
            return SimpleNamespace(returncode=1)
        if argv[0] == "wslpath":
            return SimpleNamespace(returncode=0, stdout=r"C:\Cua\cua-driver.exe", stderr="")
        assert argv[0] == "powershell.exe"
        assert install._CUA_INSTALL_PS1_URL not in argv[-1], "compatible driver must not be downloaded again"
        if failure == "repair-timeout":
            raise subprocess.TimeoutExpired(argv, 300)
        return SimpleNamespace(returncode=1, stdout="", stderr="UAC denied")

    monkeypatch.setattr(install.subprocess, "run", run)
    assert not install.install_cua_driver(upgrade=True, require_confirmed_update=True)
    assert all(argv[0] == "schtasks.exe" for argv in calls)
    assert not install.install_cua_driver()
