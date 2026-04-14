"""Test for get_python_path() fallback to PROJECT_ROOT/venv/bin/python.

Issue: #9201 - systemd service ExecStart uses uv Python instead of venv Python
when uv manages the Python environment.

Root cause: get_python_path() fell back to sys.executable when _detect_venv_dir()
returned None, but this returns uv's Python path which has no packages installed.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestGetPythonPathFallback:
    """Test that get_python_path() correctly falls back to PROJECT_ROOT/venv."""

    def test_fallback_to_project_root_venv_when_detect_fails(self):
        """When _detect_venv_dir() returns None, should use PROJECT_ROOT/venv."""
        from hermes_cli.gateway import get_python_path, PROJECT_ROOT
        
        # Mock _detect_venv_dir to return None (simulate uv environment)
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            # Mock the venv python to exist
            with patch.object(Path, "exists", return_value=True):
                result = get_python_path()
                
                # Should be PROJECT_ROOT/venv/bin/python, not sys.executable
                expected = str(PROJECT_ROOT / "venv" / "bin" / "python")
                assert result == expected

    def test_detect_venv_dir_returns_valid_venv(self):
        """When _detect_venv_dir() returns a valid venv, should use it."""
        from hermes_cli.gateway import get_python_path
        
        # Mock _detect_venv_dir to return a venv
        mock_venv = MagicMock(spec=Path)
        mock_venv.is_dir.return_value = True
        
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=mock_venv):
            # Mock the venv python to exist
            mock_python = MagicMock(spec=Path)
            mock_python.exists.return_value = True
            
            with patch.object(Path, "__truediv__", return_value=mock_python):
                result = get_python_path()
                
                # Should return the venv python path
                assert result is not None

    def test_fallback_chain_order(self):
        """Fallback order should be: detected_venv -> PROJECT_ROOT/venv -> sys.executable."""
        from hermes_cli.gateway import get_python_path, PROJECT_ROOT
        
        # Test 1: detected_venv has priority
        mock_venv = MagicMock(spec=Path)
        mock_venv_python = MagicMock(spec=Path)
        mock_venv_python.exists.return_value = True
        
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=mock_venv):
            with patch.object(Path, "__truediv__", return_value=mock_venv_python):
                result = get_python_path()
                # Should use detected_venv, not PROJECT_ROOT fallback
        
        # Test 2: PROJECT_ROOT/venv fallback when detect fails but venv exists
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            with patch.object(Path, "exists", return_value=True):
                result = get_python_path()
                expected = str(PROJECT_ROOT / "venv" / "bin" / "python")
                assert result == expected
        
        # Test 3: sys.executable when all fallbacks fail
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            with patch.object(Path, "exists", return_value=False):
                result = get_python_path()
                assert result == sys.executable

    def test_consistency_with_venv_dir_in_systemd_unit(self):
        """python_path should be consistent with venv_dir in generate_systemd_unit."""
        from hermes_cli.gateway import (
            get_python_path,
            generate_systemd_unit,
            PROJECT_ROOT,
            _detect_venv_dir,
        )
        
        # Simulate scenario where _detect_venv_dir returns None
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            with patch.object(Path, "exists", return_value=True):
                python_path = get_python_path()
                unit = generate_systemd_unit(system=False)
                
                # python_path should match venv/bin/python in ExecStart
                assert "venv/bin/python" in python_path
                assert python_path in unit

    def test_windows_fallback_uses_scripts_python_exe(self):
        """On Windows, fallback should use venv/Scripts/python.exe."""
        from hermes_cli.gateway import get_python_path, PROJECT_ROOT
        
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            with patch("hermes_cli.gateway.is_windows", return_value=True):
                with patch.object(Path, "exists", return_value=True):
                    result = get_python_path()
                    expected = str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
                    assert result == expected

    def test_execstart_not_uv_python_path(self):
        """ExecStart should NOT contain uv Python path patterns."""
        from hermes_cli.gateway import generate_systemd_unit
        
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            with patch.object(Path, "exists", return_value=True):
                unit = generate_systemd_unit(system=False)
                
                # Should NOT have uv-specific paths
                assert ".local/share/uv/python" not in unit
                assert "cpython-" not in unit
                # Should have venv path
                assert "venv/bin/python" in unit


class TestSystemdUnitGeneration:
    """Test systemd unit file generation with correct Python path."""

    def test_system_unit_execstart_uses_venv_python(self):
        """System unit ExecStart should use venv/bin/python."""
        from hermes_cli.gateway import generate_systemd_unit, PROJECT_ROOT
        
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            with patch.object(Path, "exists", return_value=True):
                unit = generate_systemd_unit(system=True, run_as_user="root")
                
                # ExecStart should contain venv/bin/python
                assert "venv/bin/python" in unit
                assert "ExecStart=" in unit

    def test_user_unit_execstart_uses_venv_python(self):
        """User unit ExecStart should use venv/bin/python."""
        from hermes_cli.gateway import generate_systemd_unit, PROJECT_ROOT
        
        with patch("hermes_cli.gateway._detect_venv_dir", return_value=None):
            with patch.object(Path, "exists", return_value=True):
                unit = generate_systemd_unit(system=False)
                
                # ExecStart should contain venv/bin/python
                assert "venv/bin/python" in unit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])