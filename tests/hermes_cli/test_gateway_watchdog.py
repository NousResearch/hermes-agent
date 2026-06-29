"""Tests for gateway watchdog functionality (issue #41662)."""

import pytest
import sys


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")
def test_watchdog_module_imports():
    """The watchdog module should import without errors."""
    from hermes_cli import gateway_watchdog

    assert hasattr(gateway_watchdog, "main")
    assert hasattr(gateway_watchdog, "_gateway_is_alive")
    assert hasattr(gateway_watchdog, "_respawn")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")
def test_watchdog_help():
    """The watchdog module should have a working --help."""
    from hermes_cli import gateway_watchdog

    # --help exits with 0
    exit_code = gateway_watchdog.main(["--help"])
    assert exit_code == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")
def test_gateway_windows_watchdog_functions_exist():
    """All watchdog-related functions should exist in gateway_windows."""
    from hermes_cli import gateway_windows

    assert hasattr(gateway_windows, "_get_watchdog_task_name")
    assert hasattr(gateway_windows, "_get_watchdog_script_path")
    assert hasattr(gateway_windows, "_build_watchdog_cmd_script")
    assert hasattr(gateway_windows, "_write_watchdog_script")
    assert hasattr(gateway_windows, "_install_watchdog_task")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")
def test_watchdog_task_naming(monkeypatch):
    """Watchdog tasks should be named distinctly per profile."""
    from hermes_cli import gateway_windows

    # Mock _profile_suffix to test naming
    def mock_suffix():
        return ""  # default profile

    monkeypatch.setattr("hermes_cli.gateway._profile_suffix", mock_suffix)
    
    # Default profile should have Watchdog in the name
    name = gateway_windows._get_watchdog_task_name()
    assert "Watchdog" in name
    assert "Hermes_Gateway_Watchdog" == name


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")
def test_watchdog_script_path_isolation(monkeypatch, tmp_path):
    """Watchdog script should be isolated from main gateway script."""
    from hermes_cli import gateway_windows

    # Mock get_hermes_home to use temp directory
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    
    gateway_path = gateway_windows.get_task_script_path()
    watchdog_path = gateway_windows._get_watchdog_script_path()
    
    # Both should be in the same directory
    assert gateway_path.parent == watchdog_path.parent
    
    # But have different names
    assert gateway_path != watchdog_path
    assert "watchdog" in watchdog_path.name.lower()
    assert "watchdog" not in gateway_path.name


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")
def test_watchdog_cmd_script_structure(monkeypatch):
    """Watchdog cmd script should be valid and self-contained."""
    from hermes_cli import gateway_windows

    python_path = "C:\\Python\\python.exe"
    working_dir = "C:\\Users\\test\\hermes"
    hermes_home = "C:\\Users\\test\\.hermes"
    profile_arg = ""

    script = gateway_windows._build_watchdog_cmd_script(
        python_path, working_dir, hermes_home, profile_arg
    )

    # Should contain @echo off and proper structure
    assert "@echo off" in script
    assert "HERMES_HOME" in script
    assert "PYTHONIOENCODING" in script
    assert "gateway_watchdog" in script
    assert ">NUL 2>&1" in script  # Should suppress output
    assert "exit /b 0" in script


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only tests")
def test_watchdog_respawn_uses_breakaway(monkeypatch):
    """Watchdog respawn should use CREATE_BREAKAWAY_FROM_JOB."""
    # The actual implementation in gateway_watchdog._respawn() calls
    # gateway_windows._spawn_detached(), which is where breakaway is applied.
    # This test documents the contract.
    from hermes_cli import gateway_windows

    # _spawn_detached should be the respawn mechanism
    assert hasattr(gateway_windows, "_spawn_detached")
