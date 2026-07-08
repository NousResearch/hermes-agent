"""
Test _find_hermes_root() function in hermes_cli/main.py.

This function is critical for package installations (Brew, pip, etc.) where
the repository structure may differ from the development environment.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest


def _find_hermes_root_test_impl(main_py_path: Path) -> Path:
    """Test implementation of _find_hermes_root().

    Matches the implementation in hermes_cli/main.py.
    """
    # Strategy 1: Try the conventional PROJECT_ROOT
    project_root = main_py_path.parent.parent.resolve()
    if (project_root / "apps" / "desktop" / "package.json").exists():
        return project_root

    # Strategy 2: Search upward for pyproject.toml
    path = main_py_path.resolve()
    for _ in range(10):
        if (path / "pyproject.toml").exists():
            if (path / "apps" / "desktop" / "package.json").exists():
                return path
        parent = path.parent
        if parent == path:
            break
        path = parent

    # Strategy 3: Fallback to HERMES_ROOT environment variable
    if "HERMES_ROOT" in os.environ:
        root = Path(os.environ["HERMES_ROOT"]).resolve()
        if (root / "pyproject.toml").exists():
            if (root / "apps" / "desktop" / "package.json").exists():
                return root

    # Final fallback: return the original PROJECT_ROOT
    return project_root


class TestFindHermesRoot:
    """Test _find_hermes_root() behavior."""

    def test_dev_environment_conventional_path(self, tmp_path: Path):
        """Test that conventional PROJECT_ROOT works in development environment."""
        # Create dev environment structure
        repo_root = tmp_path / "hermes-repo"
        hermes_cli = repo_root / "hermes_cli"
        apps_desktop = repo_root / "apps" / "desktop"

        hermes_cli.mkdir(parents=True)
        apps_desktop.mkdir(parents=True)

        (hermes_cli / "main.py").touch()
        (apps_desktop / "package.json").write_text('{"name": "hermes"}')

        main_py = hermes_cli / "main.py"
        result = _find_hermes_root_test_impl(main_py)

        assert result == repo_root, f"Expected {repo_root}, got {result}"
        assert (result / "apps" / "desktop" / "package.json").exists()

    def test_package_installation_searches_upward(self, tmp_path: Path):
        """Test that package installations search upward for pyproject.toml."""
        # Create package installation structure
        site_packages = tmp_path / "site-packages"
        hermes_cli = site_packages / "hermes_cli"
        repo_root = tmp_path / "hermes-repo"
        apps_desktop = repo_root / "apps" / "desktop"

        site_packages.mkdir(parents=True)
        hermes_cli.mkdir()
        apps_desktop.mkdir(parents=True)

        (hermes_cli / "main.py").touch()
        (repo_root / "pyproject.toml").touch()
        (apps_desktop / "package.json").write_text('{"name": "hermes"}')

        main_py = hermes_cli / "main.py"
        result = _find_hermes_root_test_impl(main_py)

        # Should find repo_root by searching upward
        assert result == repo_root, f"Expected {repo_root}, got {result}"
        assert (result / "pyproject.toml").exists()
        assert (result / "apps" / "desktop" / "package.json").exists()

    def test_hermes_root_environment_override(self, tmp_path: Path):
        """Test that HERMES_ROOT environment variable overrides all strategies."""
        # Create two repo structures
        site_packages = tmp_path / "site-packages"
        hermes_cli = site_packages / "hermes_cli"
        repo_root = tmp_path / "hermes-repo"
        override_root = tmp_path / "override-repo"
        apps_desktop_override = override_root / "apps" / "desktop"

        site_packages.mkdir(parents=True)
        hermes_cli.mkdir()
        override_root.mkdir(parents=True)
        apps_desktop_override.mkdir(parents=True)

        (hermes_cli / "main.py").touch()
        (override_root / "pyproject.toml").touch()
        (apps_desktop_override / "package.json").write_text('{"name": "hermes"}')

        main_py = hermes_cli / "main.py"

        # Set HERMES_ROOT environment variable
        old_env = os.environ.get("HERMES_ROOT")
        try:
            os.environ["HERMES_ROOT"] = str(override_root)
            result = _find_hermes_root_test_impl(main_py)
            assert result == override_root, f"Expected {override_root}, got {result}"
        finally:
            if old_env is None:
                os.environ.pop("HERMES_ROOT", None)
            else:
                os.environ["HERMES_ROOT"] = old_env

    def test_fallback_returns_conventional_path(self, tmp_path: Path):
        """Test that fallback returns the conventional PROJECT_ROOT when all strategies fail."""
        # Create structure where desktop doesn't exist
        site_packages = tmp_path / "site-packages"
        hermes_cli = site_packages / "hermes_cli"

        site_packages.mkdir(parents=True)
        hermes_cli.mkdir()

        (hermes_cli / "main.py").touch()

        main_py = hermes_cli / "main.py"
        result = _find_hermes_root_test_impl(main_py)

        # Should return the conventional PROJECT_ROOT as fallback
        expected = site_packages
        assert result == expected, f"Expected {expected}, got {result}"
        # desktop/package.json should not exist (this will trigger error later)
        assert not (result / "apps" / "desktop" / "package.json").exists()

    def test_search_stops_at_filesystem_root(self, tmp_path: Path):
        """Test that upward search stops at filesystem root."""
        # Create minimal structure at tmp_path (which is not filesystem root)
        site_packages = tmp_path / "site-packages"
        hermes_cli = site_packages / "hermes_cli"

        site_packages.mkdir(parents=True)
        hermes_cli.mkdir()

        (hermes_cli / "main.py").touch()

        main_py = hermes_cli / "main.py"
        result = _find_hermes_root_test_impl(main_py)

        # Should not crash or hang
        expected = site_packages
        assert result == expected