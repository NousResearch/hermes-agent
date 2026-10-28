"""WSL setup stages the Windows pin and preserves opt-in autostart."""
import os
from types import SimpleNamespace

import pytest

from hermes_cli import tools_config_cua as install
from hermes_cli.config import save_config


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("autostart", [False, True])
def test_wsl_setup_stages_windows_pin_without_installing_the_guest(tmp_path, monkeypatch, autostart):
    monkeypatch.setattr("hermes_constants.is_wsl", lambda: True)
    save_config({"computer_use": {"target": "windows", "autostart": autostart}})
    executable = tmp_path / "cua-driver.exe"
    executable.write_text("Windows binary fixture")
    executable.chmod(0o644)
    exe = str(executable)
    windows_exe = r"C:\Users\Alice O'Neil\cua-driver.exe"
    acquired = []
    state = {"task": False}
    monkeypatch.setattr(install, "_resolved_cua_driver_cmd", lambda: exe)
    monkeypatch.setattr(install, "_cua_driver_contract_status", lambda *a: {"ready": True, "binary": exe})
    monkeypatch.setattr("pm.client.stage_only", lambda name, target: acquired.append((name, target)) or tmp_path)
    monkeypatch.setattr("pm.ensure", lambda *a, **kw: pytest.fail("must not install the Linux guest"))
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    which = install.shutil.which
    monkeypatch.setattr(install.shutil, "which", lambda name: name if name in {exe, "powershell.exe"} else which(name))
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        if argv[0] == "wslpath":
            return SimpleNamespace(returncode=0, stdout=windows_exe)
        if argv[0] == "schtasks.exe":
            xml = f"<Task><Actions><Exec><Command>{windows_exe}</Command></Exec></Actions></Task>".encode()
            return SimpleNamespace(returncode=0 if state["task"] else 1, stdout=xml)
        assert argv[0] == "powershell.exe"
        assert "@('autostart','enable')" in argv[-1]
        assert "Alice O''Neil" in argv[-1]
        assert "irm " not in argv[-1]
        state["task"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(install.subprocess, "run", run)
    assert install.install_cua_driver(show_installer_progress=False)
    assert acquired == [("cua-driver", "win32-x64")]
    assert os.access(exe, os.X_OK)
    assert install._cua_driver_install_ready()
    assert any(argv[0] == "powershell.exe" for argv in commands) is autostart
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", exe)
    acquired.clear()
    assert install.install_cua_driver(show_installer_progress=False)
    assert acquired == []
