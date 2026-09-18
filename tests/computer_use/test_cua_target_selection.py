"""Computer-use target selection is profile-scoped and fails closed across WSL host/guest boundaries."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml


def _target_config(target: str):
    return patch("hermes_cli.config.load_config", return_value={"computer_use": {"target": target}})


def test_default_target_is_auto_and_config_default_is_registered():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.computer_use.cua_backend_driver import computer_use_target

    with patch("hermes_cli.config.load_config", return_value={}):
        assert computer_use_target() == "auto"
    assert DEFAULT_CONFIG["computer_use"]["target"] == "auto"


def test_explicit_target_rejects_conflicting_authoritative_override(tmp_path, monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    linux_driver = tmp_path / "cua-driver"
    linux_driver.write_text("#!/bin/sh\n", encoding="utf-8")
    linux_driver.chmod(0o755)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(linux_driver))

    with _target_config("windows"):
        target, platform, command, error = driver._resolve_cua_driver_selection(runtime_host=("linux", True))
    assert (target, platform, command) == ("windows", "linux", str(linux_driver))
    assert "HERMES_CUA_DRIVER_CMD" in (error or "") and "linux" in (error or "").lower()


def test_linux_target_rejects_windows_executable_through_path_symlink(tmp_path, monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    exe = tmp_path / "cua-driver.exe"
    exe.write_text("binary", encoding="utf-8")
    exe.chmod(0o755)
    link = tmp_path / "cua-driver"
    link.symlink_to(exe)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(link))

    with _target_config("linux"):
        assert driver.resolve_cua_driver_cmd() is None
        assert "windows" in (driver.computer_use_target_error() or "").lower()


def test_wsl_windows_target_uses_windows_path_and_never_linux_fallback(monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(driver, "_wsl_windows_install_paths", lambda: ["/mnt/c/Users/u/AppData/Local/Programs/Cua/cua-driver/bin/cua-driver.exe"])
    monkeypatch.setattr(driver.shutil, "which", lambda value: "/usr/bin/cua-driver" if value == "cua-driver" else None)

    with _target_config("windows"):
        _, platform, command, error = driver._resolve_cua_driver_selection(runtime_host=("linux", True))
    assert platform is None and command is None
    error = error or ""
    assert "Windows" in error and "not found" in error


def test_wsl_windows_target_prefers_exe_on_path(monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(driver, "_wsl_windows_install_paths", lambda: [])
    monkeypatch.setattr(driver.shutil, "which", lambda value: "/mnt/c/tools/cua-driver.exe" if value == "cua-driver.exe" else None)

    with _target_config("windows"):
        assert driver._resolve_cua_driver_selection(runtime_host=("linux", True)) == (
            "windows", "windows", "/mnt/c/tools/cua-driver.exe", None)


def test_wsl_windows_install_discovery_is_fixed_sanitized_and_cached(monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout='{"LocalAppData":"C:\\\\Users\\\\Alice\\\\AppData\\\\Local",'
                   '"UserProfile":"C:\\\\Users\\\\Alice"}',
        )

    monkeypatch.setattr(driver.shutil, "which", lambda value: "/mnt/c/Windows/powershell.exe" if value == "powershell.exe" else None)
    monkeypatch.setattr(driver.subprocess, "run", run)
    monkeypatch.setattr(driver, "_wsl_windows_path_to_posix", lambda path: "converted:" + path)
    monkeypatch.setattr(driver, "_cb", lambda: SimpleNamespace(sanitized_cua_driver_env=lambda: {"PATH": "safe"}))
    driver._cached_wsl_windows_install_paths.cache_clear()

    paths = driver._wsl_windows_install_paths()
    assert paths == [
        r"converted:C:\Users\Alice\AppData\Local\Programs\Cua\cua-driver\bin\cua-driver.exe",
        r"converted:C:\Users\Alice\.local\bin\cua-driver.exe",
    ]
    assert driver._wsl_windows_install_paths() == paths
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[1:4] == ["-NoLogo", "-NoProfile", "-NonInteractive"]
    assert argv[4] == "-Command" and "GetFolderPath" in argv[5]
    assert kwargs["timeout"] == 5.0 and kwargs["stdin"] is driver.subprocess.DEVNULL
    assert kwargs["env"] == {"PATH": "safe"}
    driver._cached_wsl_windows_install_paths.cache_clear()


def test_auto_preserves_regular_linux_resolution(monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(driver.shutil, "which", lambda value: "/usr/bin/cua-driver" if value == "cua-driver" else None)

    with _target_config("auto"):
        assert driver._resolve_cua_driver_selection(runtime_host=("linux", False))[:3] == (
            "auto", "linux", "/usr/bin/cua-driver")


@pytest.mark.macos_only
def test_auto_preserves_native_macos_resolution(monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(driver.shutil, "which", lambda value: "/usr/local/bin/cua-driver" if value == "cua-driver" else None)
    with _target_config("auto"):
        assert driver.computer_use_selection_identity()[:2] == ("auto", "macos")


@pytest.mark.windows_only
def test_windows_target_works_on_native_windows(monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(driver.shutil, "which", lambda value: r"C:\Tools\cua-driver.exe" if value == "cua-driver.exe" else None)
    with _target_config("windows"):
        assert driver.computer_use_selection_identity()[:2] == ("windows", "windows")


def test_target_is_profile_scoped_a_b_a(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.computer_use.cua_backend_driver import computer_use_target

    homes = [tmp_path / "a", tmp_path / "b"]
    for home, target in zip(homes, ("linux", "windows")):
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump({"computer_use": {"target": target}}), encoding="utf-8")
    seen = []
    for home in (homes[0], homes[1], homes[0]):
        token = set_hermes_home_override(str(home))
        try:
            seen.append(computer_use_target())
        finally:
            reset_hermes_home_override(token)
    assert seen == ["linux", "windows", "linux"]


def test_status_reports_target_and_resolved_driver_identity(monkeypatch):
    from hermes_constants import is_wsl
    from tools.computer_use import permissions
    from tools.computer_use import cua_backend_driver as driver

    monkeypatch.setattr(driver, "computer_use_selection_identity", lambda: ("windows", "windows", "/mnt/c/cua-driver.exe", None))
    monkeypatch.setattr(permissions, "_doctor", lambda binary: {"ok": True, "checks": []})
    monkeypatch.setattr(permissions, "_run", lambda *a, **k: MagicMock(stdout="0.20.0"))

    status = permissions.computer_use_status()
    assert status["platform"] == sys.platform
    assert {k: status[k] for k in ("target", "is_wsl", "driver_platform", "driver_command", "target_error")} == {
        "target": "windows", "is_wsl": is_wsl(), "driver_platform": "windows",
        "driver_command": "/mnt/c/cua-driver.exe", "target_error": None,
    }


def test_embedded_daemon_uses_named_pipe_for_windows_exe_path_symlink(tmp_path):
    from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon

    exe = tmp_path / "cua-driver.exe"
    exe.write_text("binary", encoding="utf-8")
    link = tmp_path / "cua-driver"
    link.symlink_to(exe)

    daemon = _EmbeddedCuaDaemon(str(link), "unrestricted")
    assert daemon.socket_path.startswith(r"\\.\pipe\hermes-cua-")


@pytest.mark.parametrize("failure", ["timeout", "missing-powershell", "bad-json", "nonzero", "incomplete"])
def test_failed_windows_discovery_recovers_without_restart(monkeypatch, failure):
    from tools.computer_use import cua_backend_driver as driver

    available = [failure != "missing-powershell"]
    monkeypatch.setattr(driver.shutil, "which", lambda name: "powershell.exe" if available[0] else None)
    monkeypatch.setattr(driver, "_wsl_windows_path_to_posix", lambda path: "converted:" + path)
    monkeypatch.setattr(driver, "_cb", lambda: SimpleNamespace(sanitized_cua_driver_env=lambda: {}))
    success = SimpleNamespace(returncode=0, stdout='{"LocalAppData":"C:/Users/A/AppData/Local","UserProfile":"C:/Users/A"}')
    first = {
        "timeout": subprocess.TimeoutExpired("powershell.exe", 5),
        "bad-json": SimpleNamespace(returncode=0, stdout="not JSON"),
        "nonzero": SimpleNamespace(returncode=1, stdout="{}"),
        "incomplete": SimpleNamespace(returncode=0, stdout='{"UserProfile":"C:/Users/A"}'),
    }.get(failure)
    run = MagicMock(side_effect=[first, success] if first is not None else [success])
    monkeypatch.setattr(driver.subprocess, "run", run)
    driver._cached_wsl_windows_install_paths.cache_clear()
    try:
        assert driver._wsl_windows_install_paths() == []
        available[0] = True
        paths = driver._wsl_windows_install_paths()
        assert len(paths) == 2
        calls = run.call_count
        assert driver._wsl_windows_install_paths() == paths
        assert run.call_count == calls
    finally:
        driver._cached_wsl_windows_install_paths.cache_clear()


def test_auto_does_not_probe_windows_installation_or_change_existing_precedence(monkeypatch):
    from tools.computer_use import cua_backend_driver as driver

    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(driver.shutil, "which", lambda name: "/usr/bin/cua-driver" if name == "cua-driver" else None)
    monkeypatch.setattr(driver, "_wsl_windows_install_paths", MagicMock(side_effect=AssertionError("auto must preserve resolution")))
    with _target_config("auto"):
        assert driver._resolve_cua_driver_selection(runtime_host=("linux", True))[:3] == (
            "auto", "linux", "/usr/bin/cua-driver")
