from __future__ import annotations

import subprocess
from types import SimpleNamespace

import hermes_cli.main as hermes_main
import hermes_cli.setup as setup_mod
import hermes_cli.tools_config as tools_config
import tools.lazy_deps as lazy_deps


def test_lazy_deps_uv_install_hides_console(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(lazy_deps.shutil, "which", lambda name: "C:/uv.exe" if name == "uv" else None)
    monkeypatch.setattr(lazy_deps, "windows_hide_flags", lambda: 0x1234)
    monkeypatch.setattr(lazy_deps.subprocess, "run", fake_run)

    result = lazy_deps._venv_pip_install(("example>=1,<2",))

    assert result.success is True
    assert calls[0][0] == ["C:/uv.exe", "pip", "install", "example>=1,<2"]
    assert calls[0][1]["creationflags"] == 0x1234


def test_tools_config_uv_install_hides_console(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(tools_config.shutil, "which", lambda name: "C:/uv.exe" if name == "uv" else None)
    monkeypatch.setattr(tools_config, "windows_hide_flags", lambda: 0x5678)
    monkeypatch.setattr(tools_config.subprocess, "run", fake_run)

    result = tools_config._pip_install(["example>=1,<2"])

    assert result.returncode == 0
    assert calls[0][0] == ["C:/uv.exe", "pip", "install", "example>=1,<2"]
    assert calls[0][1]["creationflags"] == 0x5678


def test_run_install_with_heartbeat_hides_console(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hermes_main, "windows_hide_flags", lambda: 0x9ABC)
    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main._run_install_with_heartbeat(
        ["C:/uv.exe", "pip", "install", "-e", "."],
        heartbeat_interval_seconds=999,
    )

    assert calls[0][0] == ["C:/uv.exe", "pip", "install", "-e", "."]
    assert calls[0][1]["creationflags"] == 0x9ABC


def test_setup_uv_install_hides_console(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(setup_mod, "windows_hide_flags", lambda: 0xDEF0)
    monkeypatch.setattr(setup_mod.subprocess, "run", fake_run)

    result = setup_mod._run_uv_pip_install_for_current_python("C:/uv.exe", "modal")

    assert result.returncode == 0
    assert calls[0][0] == ["C:/uv.exe", "pip", "install", "--python", setup_mod.sys.executable, "modal"]
    assert calls[0][1]["creationflags"] == 0xDEF0
