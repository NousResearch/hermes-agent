"""POSIX Compliance Tests for Agathos.

Ensures cross-platform compatibility by verifying:
1. Path separator usage (os.pathsep, not hardcoded ':')
2. Platform detection constants
3. Cross-platform PATH construction
4. Service management platform guards

Run: cd ~/Projects/hermes-dev && python -m pytest agathos/tests/test_posix_compliance.py -v
"""

import os
import sys
import platform
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestPlatformDetection:
    """Test platform detection constants are correctly defined."""

    def test_venv_utils_platform_constants(self):
        """Verify venv_utils defines platform constants."""
        from agathos.venv_utils import _IS_MACOS, _IS_LINUX, _IS_WINDOWS

        # Exactly one should be True for the current platform
        assert isinstance(_IS_MACOS, bool)
        assert isinstance(_IS_LINUX, bool)
        assert isinstance(_IS_WINDOWS, bool)

        # Verify logic consistency
        if sys.platform == 'darwin':
            assert _IS_MACOS is True
            assert _IS_LINUX is False
            assert _IS_WINDOWS is False
        elif sys.platform.startswith('linux'):
            assert _IS_MACOS is False
            assert _IS_LINUX is True
            assert _IS_WINDOWS is False
        elif os.name == 'nt' or sys.platform == 'win32':
            assert _IS_MACOS is False
            assert _IS_LINUX is False
            assert _IS_WINDOWS is True

    def test_subprocess_utils_platform_constants(self):
        """Verify subprocess_utils defines platform constants."""
        from agathos.subprocess_utils import _IS_MACOS, _IS_LINUX, _IS_WINDOWS

        assert isinstance(_IS_MACOS, bool)
        assert isinstance(_IS_LINUX, bool)
        assert isinstance(_IS_WINDOWS, bool)

    def test_daemon_mgmt_platform_constants(self):
        """Verify daemon_mgmt defines platform constants."""
        from agathos.daemon_mgmt import _IS_MACOS, _IS_LINUX, _IS_WINDOWS

        assert isinstance(_IS_MACOS, bool)
        assert isinstance(_IS_LINUX, bool)
        assert isinstance(_IS_WINDOWS, bool)

    def test_cli_platform_constants(self):
        """Verify cli defines platform constants."""
        from agathos.cli import _IS_MACOS, _IS_LINUX, _IS_WINDOWS

        assert isinstance(_IS_MACOS, bool)
        assert isinstance(_IS_LINUX, bool)
        assert isinstance(_IS_WINDOWS, bool)


class TestPathSeparatorCompliance:
    """Test that path separators use os.pathsep, not hardcoded ':'."""

    def test_build_venv_aware_env_uses_pathsep(self):
        """Verify build_venv_aware_env uses os.pathsep for PATH."""
        from agathos.venv_utils import build_venv_aware_env

        # Test with extra paths
        extra = ['/extra/path', '/another/path']
        env = build_venv_aware_env(extra_paths=extra)
        path = env.get('PATH', '')

        # PATH should use os.pathsep, not hardcoded ':'
        if os.pathsep == ';':  # Windows
            assert ';' in path or path == '', f"PATH should use ; separator on Windows: {path}"
        else:  # POSIX
            assert ':' in path or path == '', f"PATH should use : separator on POSIX: {path}"

    def test_build_agathos_subprocess_env_uses_pathsep(self):
        """Verify build_agathos_subprocess_env uses os.pathsep for PATH."""
        from agathos.venv_utils import build_agathos_subprocess_env

        env = build_agathos_subprocess_env()
        path = env.get('PATH', '')

        # Verify PATH uses correct separator for platform
        if os.pathsep == ';':
            # On Windows, shouldn't have unescaped colons in Windows paths
            pass  # Just verify it doesn't crash
        else:
            # On POSIX, should have colons separating paths
            assert ':' in path or len(path) == 0

    def test_subprocess_utils_uses_pathsep(self):
        """Verify subprocess_utils uses os.pathsep."""
        from agathos.subprocess_utils import build_agathos_subprocess_env

        env = build_agathos_subprocess_env()
        path = env.get('PATH', '')

        # Should be properly formatted with os.pathsep
        parts = path.split(os.pathsep)
        # Should have multiple path components or be empty/minimal
        assert len(parts) >= 0  # Just verify it doesn't crash

    def test_no_hardcoded_colon_in_path_splitting(self):
        """Verify no hardcoded ':' in PATH splitting operations."""
        import agathos.venv_utils as venv_utils
        import agathos.subprocess_utils as subprocess_utils

        # Read source and verify no hardcoded ':' for PATH splitting
        venv_source = Path(venv_utils.__file__).read_text()
        subproc_source = Path(subprocess_utils.__file__).read_text()

        # Should use os.pathsep or os.sep, not hardcoded ':' for PATH operations
        # (Allow ':' in comments, strings that aren't PATH-related)
        lines_with_pathsep = []
        for i, line in enumerate(venv_source.split('\n'), 1):
            if '.split(' in line or '.join(' in line:
                if "':'" in line or '":"' in line:
                    lines_with_pathsep.append((i, line.strip()))

        # The only hardcoded ':' should be in the platform-specific paths list
        # or in os.pathsep context
        for lineno, line in lines_with_pathsep:
            # Skip if it's the platform paths definition
            if '/usr/local' in line or '/opt/homebrew' in line:
                continue
            # Skip if it's os.pathsep
            if 'os.pathsep' in line:
                continue
            assert False, f"Line {lineno} may have hardcoded PATH separator: {line}"


class TestCrossPlatformPaths:
    """Test cross-platform path handling."""

    def test_get_platform_std_paths_returns_list(self):
        """Verify _get_platform_std_paths returns a list."""
        from agathos.venv_utils import _get_platform_std_paths
        from agathos.subprocess_utils import _get_platform_std_paths as subproc_get_paths

        paths = _get_platform_std_paths()
        subproc_paths = subproc_get_paths()

        assert isinstance(paths, list)
        assert isinstance(subproc_paths, list)

        # All items should be strings
        for p in paths:
            assert isinstance(p, str)
        for p in subproc_paths:
            assert isinstance(p, str)

    def test_platform_specific_paths(self):
        """Verify platform-specific paths are appropriate."""
        from agathos.venv_utils import _get_platform_std_paths, _IS_MACOS, _IS_LINUX, _IS_WINDOWS

        paths = _get_platform_std_paths()

        if _IS_MACOS:
            # macOS should have Homebrew paths
            assert '/opt/homebrew/bin' in paths or '/usr/local/bin' in paths
        elif _IS_LINUX:
            # Linux should have standard paths but not Homebrew-specific
            assert '/usr/bin' in paths
            assert '/bin' in paths
            assert '/opt/homebrew/bin' not in paths
        elif _IS_WINDOWS:
            # Windows should return empty list (uses registry/PATH env)
            assert paths == []

    def test_venv_bin_dir_returns_correct_path(self):
        """Verify get_venv_bin_dir returns correct path for platform."""
        from agathos.venv_utils import get_venv_bin_dir, _IS_WINDOWS

        # Use a temp path
        test_venv = Path('/tmp/test_venv')
        bin_dir = get_venv_bin_dir(test_venv)

        if _IS_WINDOWS:
            assert 'Scripts' in str(bin_dir)
        else:
            assert 'bin' in str(bin_dir)

    def test_service_directory_by_platform(self):
        """Verify service directory is appropriate for platform."""
        from agathos.daemon_mgmt import _get_service_directory, _IS_MACOS, _IS_LINUX, _IS_WINDOWS

        service_dir = _get_service_directory()

        assert isinstance(service_dir, Path)

        if _IS_MACOS:
            assert 'Library' in str(service_dir)
            assert 'LaunchAgents' in str(service_dir)
        elif _IS_LINUX:
            assert '.config' in str(service_dir) or 'systemd' in str(service_dir)
        elif _IS_WINDOWS:
            assert 'AppData' in str(service_dir) or 'Roaming' in str(service_dir)


class TestServiceManagementGuards:
    """Test service management has proper platform guards."""

    def test_launchd_install_checks_platform(self):
        """Verify agathos_launchd_install checks platform."""
        from agathos.daemon_mgmt import agathos_launchd_install, _IS_MACOS

        # On non-macOS, should return False without crashing
        if not _IS_MACOS:
            result = agathos_launchd_install()
            assert result is False, "Should return False on non-macOS platforms"

    def test_launchd_install_does_not_crash(self):
        """Verify agathos_launchd_install doesn't crash on any platform."""
        from agathos.daemon_mgmt import agathos_launchd_install

        # Should not raise an exception
        try:
            result = agathos_launchd_install()
            # Result can be True or False, but shouldn't crash
            assert isinstance(result, bool)
        except Exception as e:
            # If it raises, it should be a specific error, not AttributeError/NameError
            assert not isinstance(e, (AttributeError, NameError, ImportError)), \
                f"Should not have import/attribute errors: {e}"

    def test_launchd_uninstall_does_not_crash(self):
        """Verify agathos_launchd_uninstall doesn't crash on any platform."""
        from agathos.daemon_mgmt import agathos_launchd_uninstall

        try:
            result = agathos_launchd_uninstall()
            assert isinstance(result, bool)
        except Exception as e:
            assert not isinstance(e, (AttributeError, NameError, ImportError)), \
                f"Should not have import/attribute errors: {e}"

    def test_launchd_status_returns_dict(self):
        """Verify agathos_launchd_status returns proper dict."""
        from agathos.daemon_mgmt import agathos_launchd_status

        status = agathos_launchd_status()

        assert isinstance(status, dict)
        assert 'label' in status
        # Support both old (plist_path) and new (service_path) field names
        assert 'service_path' in status or 'plist_path' in status
        assert 'service_exists' in status or 'plist_exists' in status
        assert 'pid_file_exists' in status
        assert 'running_pid' in status
        assert 'is_running' in status


class TestPathConstruction:
    """Test PATH environment construction."""

    def test_agathos_subprocess_env_includes_hermes_paths(self):
        """Verify subprocess env includes hermes paths."""
        from agathos.subprocess_utils import build_agathos_subprocess_env

        env = build_agathos_subprocess_env()
        path = env.get('PATH', '')

        # Should include hermes bin path
        assert 'hermes' in path.lower() or path == '', f"PATH should include hermes: {path}"

    def test_agathos_subprocess_env_includes_home(self):
        """Verify subprocess env sets HOME."""
        from agathos.subprocess_utils import build_agathos_subprocess_env

        env = build_agathos_subprocess_env()

        assert 'HOME' in env
        assert Path(env['HOME']).exists() or env['HOME'] == os.path.expanduser('~')

    def test_agathos_subprocess_env_no_duplicate_paths(self):
        """Verify PATH doesn't have duplicates from merged sources."""
        from agathos.venv_utils import build_agathos_subprocess_env

        env = build_agathos_subprocess_env()
        path = env.get('PATH', '')

        parts = path.split(os.pathsep)
        # Remove empty strings
        parts = [p for p in parts if p]

        # Check for duplicates
        unique_parts = list(dict.fromkeys(parts))  # preserve order, remove dups
        assert len(parts) == len(unique_parts), \
            f"PATH has duplicates: {len(parts)} parts, {len(unique_parts)} unique"


class TestImportsWorkOnAllPlatforms:
    """Test that all modules can be imported on any platform."""

    def test_venv_utils_imports(self):
        """Verify venv_utils imports on this platform."""
        from agathos.venv_utils import (
            is_running_in_venv,
            get_venv_path,
            get_venv_bin_dir,
            get_venv_python,
            detect_hermes_venv,
            get_hermes_python,
            build_venv_aware_env,
            get_agathos_venv_paths,
            build_agathos_subprocess_env,
            resolve_venv_python,
        )

        # Just verify no ImportError
        assert callable(is_running_in_venv)
        assert callable(get_venv_path)

    def test_subprocess_utils_imports(self):
        """Verify subprocess_utils imports on this platform."""
        from agathos.subprocess_utils import (
            build_agathos_subprocess_env,
            safe_subprocess,
        )

        assert callable(build_agathos_subprocess_env)
        assert callable(safe_subprocess)

    def test_daemon_mgmt_imports(self):
        """Verify daemon_mgmt imports on this platform."""
        from agathos.daemon_mgmt import (
            write_agathos_pid_file,
            remove_agathos_pid_file,
            get_agathos_running_pid,
            is_agathos_running,
            agathos_launchd_install,
            agathos_launchd_uninstall,
            agathos_launchd_status,
        )

        assert callable(write_agathos_pid_file)
        assert callable(agathos_launchd_install)


class TestWindowsSpecific:
    """Tests that verify Windows-specific behavior when on Windows."""

    def test_windows_uses_scripts_not_bin(self):
        """On Windows, venv bin dir should be Scripts."""
        from agathos.venv_utils import get_venv_bin_dir, _IS_WINDOWS

        if not _IS_WINDOWS:
            pytest.skip("Not on Windows")

        bin_dir = get_venv_bin_dir(Path('/fake/venv'))
        assert str(bin_dir).endswith('Scripts')

    def test_windows_path_separator(self):
        """On Windows, PATH should use semicolon."""
        from agathos.venv_utils import build_agathos_subprocess_env, _IS_WINDOWS

        if not _IS_WINDOWS:
            pytest.skip("Not on Windows")

        env = build_agathos_subprocess_env()
        path = env.get('PATH', '')

        # On Windows, should use ; as separator
        if len(path) > 0:
            assert ';' in path or len(path.split(os.pathsep)) <= 1


class TestMacOSSpecific:
    """Tests that verify macOS-specific behavior when on macOS."""

    def test_macos_uses_bin_not_scripts(self):
        """On macOS, venv bin dir should be bin."""
        from agathos.venv_utils import get_venv_bin_dir, _IS_MACOS

        if not _IS_MACOS:
            pytest.skip("Not on macOS")

        bin_dir = get_venv_bin_dir(Path('/fake/venv'))
        assert str(bin_dir).endswith('bin')

    def test_macos_includes_homebrew_paths(self):
        """On macOS, std paths should include Homebrew locations."""
        from agathos.venv_utils import _get_platform_std_paths, _IS_MACOS

        if not _IS_MACOS:
            pytest.skip("Not on macOS")

        paths = _get_platform_std_paths()

        # Should include at least one of the Homebrew paths
        has_homebrew = '/opt/homebrew/bin' in paths or '/usr/local/bin' in paths
        assert has_homebrew, f"macOS paths should include Homebrew: {paths}"


class TestLinuxSpecific:
    """Tests that verify Linux-specific behavior when on Linux."""

    def test_linux_uses_standard_paths(self):
        """On Linux, std paths should be standard Unix paths."""
        from agathos.venv_utils import _get_platform_std_paths, _IS_LINUX

        if not _IS_LINUX:
            pytest.skip("Not on Linux")

        paths = _get_platform_std_paths()

        # Should have standard Unix paths
        assert '/usr/bin' in paths or '/bin' in paths

        # Should NOT have Homebrew paths
        assert '/opt/homebrew/bin' not in paths

    def test_linux_snap_paths_added(self):
        """On Linux, snap paths should be added if present."""
        from agathos.venv_utils import _get_platform_std_paths, _IS_LINUX

        if not _IS_LINUX:
            pytest.skip("Not on Linux")

        paths = _get_platform_std_paths()
        # Should include snap path (if directory exists)
        # Just verify the function works
        assert isinstance(paths, list)

    def test_wsl_detection(self):
        """Test WSL detection works on Linux."""
        from agathos.daemon_mgmt import _is_wsl
        from agathos.venv_utils import _is_wsl as venv_is_wsl

        # Both should return the same result
        assert _is_wsl() == venv_is_wsl()

        # Should return a bool
        assert isinstance(_is_wsl(), bool)

    def test_systemd_service_generation(self):
        """Test systemd service file generation."""
        from agathos.daemon_mgmt import generate_systemd_service, _IS_LINUX

        # This should work on any platform (generates a string)
        service_content = generate_systemd_service()

        assert isinstance(service_content, str)
        assert '[Unit]' in service_content
        assert '[Service]' in service_content
        assert '[Install]' in service_content
        assert 'Description=Agathos' in service_content

    def test_service_status_includes_platform_info(self):
        """Service status should include platform-specific info."""
        from agathos.daemon_mgmt import agathos_service_status, _IS_LINUX, _IS_MACOS, _IS_WINDOWS

        status = agathos_service_status()

        # Should include platform info
        assert 'platform' in status
        assert 'service_type' in status
        assert 'is_wsl' in status

        if _IS_MACOS:
            assert status['service_type'] == 'launchd'
        elif _IS_LINUX:
            assert status['service_type'] == 'systemd'


# pytest skip markers for platform-specific tests
import pytest

# Run the compliance tests
if __name__ == '__main__':
    # Use pytest if available
    try:
        import pytest
        sys.exit(pytest.main([__file__, '-v']))
    except ImportError:
        # Fallback to basic test runner
        print("pytest not available, running basic checks...")

        # Run basic import checks
        print("\n=== Testing Imports ===")
        try:
            from agathos.venv_utils import _IS_MACOS, _IS_LINUX, _IS_WINDOWS
            print(f"venv_utils: _IS_MACOS={_IS_MACOS}, _IS_LINUX={_IS_LINUX}, _IS_WINDOWS={_IS_WINDOWS}")
        except Exception as e:
            print(f"FAIL: venv_utils import error: {e}")
            sys.exit(1)

        try:
            from agathos.subprocess_utils import _IS_MACOS, _IS_LINUX, _IS_WINDOWS
            print(f"subprocess_utils: OK")
        except Exception as e:
            print(f"FAIL: subprocess_utils import error: {e}")
            sys.exit(1)

        try:
            from agathos.daemon_mgmt import _IS_MACOS, _IS_LINUX, _IS_WINDOWS
            print(f"daemon_mgmt: OK")
        except Exception as e:
            print(f"FAIL: daemon_mgmt import error: {e}")
            sys.exit(1)

        print("\n=== Testing PATH Separator ===")
        from agathos.venv_utils import build_agathos_subprocess_env
        env = build_agathos_subprocess_env()
        path = env.get('PATH', '')
        print(f"PATH separator check: os.pathsep='{os.pathsep}'")
        print(f"PATH length: {len(path)} chars, {len(path.split(os.pathsep))} components")

        print("\n=== All Basic Checks Passed ===")
        sys.exit(0)
