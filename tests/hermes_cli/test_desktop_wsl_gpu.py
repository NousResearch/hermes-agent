"""``hermes gui`` under WSLg selects the installed Mesa D3D12 driver before Electron spawns
its GPU process (#106117) — and never overrides an explicit Mesa choice or fires off-WSL."""

import argparse
from pathlib import Path

import hermes_constants
from hermes_cli import main_desktop


def _launch_env(monkeypatch, tmp_path, *, wsl: bool, dxg: bool, driver: bool) -> dict:
    """Run the real launcher env builder against a fake WSL host laid out under ``tmp_path``."""
    dxg_path = tmp_path / "dxg"
    driver_path = tmp_path / "d3d12_dri.so"
    dxg_path.unlink(missing_ok=True)
    driver_path.unlink(missing_ok=True)
    if dxg:
        dxg_path.touch()
    if driver:
        driver_path.write_bytes(b"\x7fELF")
    monkeypatch.setattr(hermes_constants, "_wsl_detected", wsl)
    monkeypatch.setattr(main_desktop, "_WSL_DXG_DEVICE", dxg_path)
    monkeypatch.setattr(main_desktop, "_WSL_D3D12_DRIVERS", (tmp_path / "missing_dri.so", driver_path))
    monkeypatch.setattr(main_desktop, "_desktop_launch_options", lambda: ([], "auto", "auto", "auto"))
    monkeypatch.setattr(main_desktop, "_detect_linux_password_store", lambda: None)
    env, _flags = main_desktop._desktop_launch_env(argparse.Namespace(cwd=str(tmp_path)))
    return env


def test_wslg_with_dxg_and_installed_d3d12_driver_selects_gpu_mesa_backend(monkeypatch, tmp_path):
    for var in main_desktop._MESA_DRIVER_OVERRIDES:
        monkeypatch.delenv(var, raising=False)

    env = _launch_env(monkeypatch, tmp_path, wsl=True, dxg=True, driver=True)

    assert env["GALLIUM_DRIVER"] == "d3d12"
    assert Path(env["HERMES_DESKTOP_CWD"]) == tmp_path.resolve()  # the rest of the env still builds


def test_explicit_mesa_choice_and_non_wsl_hosts_are_left_alone(monkeypatch, tmp_path):
    for var in main_desktop._MESA_DRIVER_OVERRIDES:
        monkeypatch.delenv(var, raising=False)

    # WSL + GPU present, but the user pinned software rendering: their choice is authoritative.
    monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "1")
    env = _launch_env(monkeypatch, tmp_path, wsl=True, dxg=True, driver=True)
    assert "GALLIUM_DRIVER" not in env and env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE")

    # Any one leg missing → untouched: plain Linux with a dxg-like node and driver installed,
    # WSL without /dev/dxg, WSL with /dev/dxg but no Mesa d3d12 driver on disk.
    for wsl, dxg, driver in ((False, True, True), (True, False, True), (True, True, False)):
        env = _launch_env(monkeypatch, tmp_path, wsl=wsl, dxg=dxg, driver=driver)
        assert "GALLIUM_DRIVER" not in env, (wsl, dxg, driver)
