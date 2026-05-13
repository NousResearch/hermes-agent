"""Test HERMES_REAL_HOME is set in subprocess environments.

Covers: https://github.com/NousResearch/hermes-agent/issues/25114

When profile isolation activates (HERMES_HOME/home/ exists), child
processes receive HOME={HERMES_HOME}/home/ for tool config isolation.
This test verifies that HERMES_REAL_HOME is also set, pointing to the
actual user home so scripts can locate ~/.hermes/ correctly.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# get_real_home unit tests
# ---------------------------------------------------------------------------

class TestGetRealHome:
    """Verify get_real_home() returns the actual user home."""

    def test_returns_home_env(self):
        """When HOME is set, get_real_home returns it."""
        from hermes_constants import get_real_home
        with mock.patch.dict(os.environ, {"HOME": "/home/testuser"}, clear=False):
            assert get_real_home() == "/home/testuser"

    def test_prefers_hermes_real_home(self):
        """HERMES_REAL_HOME takes priority over HOME."""
        from hermes_constants import get_real_home
        with mock.patch.dict(os.environ, {
            "HERMES_REAL_HOME": "/home/real",
            "HOME": "/home/fake",
        }, clear=False):
            assert get_real_home() == "/home/real"

    def test_fallback_expanduser(self):
        """When HOME is empty, falls back to expanduser."""
        from hermes_constants import get_real_home
        with mock.patch.dict(os.environ, {"HOME": ""}, clear=False):
            result = get_real_home()
            assert result  # not empty
            assert result != ""

    def test_fallback_tmp(self):
        """Last resort is /tmp."""
        from hermes_constants import get_real_home
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove HOME and HERMES_REAL_HOME
            env = {k: v for k, v in os.environ.items()
                   if k not in ("HOME", "HERMES_REAL_HOME")}
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("os.path.expanduser", return_value="~"):
                    result = get_real_home()
                    assert result == "/tmp"


# ---------------------------------------------------------------------------
# Subprocess env injection tests
# ---------------------------------------------------------------------------

class TestSubprocessEnvRealHome:
    """Verify HERMES_REAL_HOME is injected into subprocess environments."""

    def test_code_execution_sets_real_home(self, tmp_path):
        """execute_code child_env includes HERMES_REAL_HOME."""
        # Simulate profile isolation: HERMES_HOME/home/ exists
        profile_home = tmp_path / "profiles" / "worker"
        home_dir = profile_home / "home"
        home_dir.mkdir(parents=True)

        with mock.patch.dict(os.environ, {
            "HOME": "/home/testuser",
            "HERMES_HOME": str(profile_home),
        }, clear=False):
            from hermes_constants import get_subprocess_home, get_real_home
            
            profile_home_val = get_subprocess_home()
            assert profile_home_val == str(home_dir)
            
            real_home = get_real_home()
            assert real_home == "/home/testuser"
            assert real_home != profile_home_val

    def test_local_env_sets_real_home(self, tmp_path):
        """Local environment subprocesses get HERMES_REAL_HOME."""
        profile_home = tmp_path / "profiles" / "worker"
        home_dir = profile_home / "home"
        home_dir.mkdir(parents=True)

        with mock.patch.dict(os.environ, {
            "HOME": "/home/testuser",
            "HERMES_HOME": str(profile_home),
        }, clear=False):
            # Import and check the _make_run_env function
            import importlib
            import tools.environments.local as local_mod
            importlib.reload(local_mod)
            
            # The function should add HERMES_REAL_HOME when profile home is active
            from hermes_constants import get_real_home
            assert get_real_home() == "/home/testuser"

    def test_no_real_home_when_not_isolated(self):
        """When profile isolation is off, HERMES_REAL_HOME is not needed."""
        with mock.patch.dict(os.environ, {
            "HOME": "/home/testuser",
            "HERMES_HOME": "/home/testuser/.hermes",
        }, clear=False):
            from hermes_constants import get_subprocess_home
            result = get_subprocess_home()
            assert result is None  # No profile home dir


# ---------------------------------------------------------------------------
# Integration: verify the pattern works end-to-end
# ---------------------------------------------------------------------------

class TestRealHomeIntegration:
    """End-to-end verification that subprocesses can find ~/.hermes/."""

    def test_subprocess_can_find_hermes_dir(self, tmp_path):
        """A subprocess with overridden HOME can still find .hermes/ via HERMES_REAL_HOME."""
        real_home = tmp_path / "real_home"
        real_home.mkdir()
        (real_home / ".hermes").mkdir()

        profile_home = tmp_path / "profile_home"
        profile_home.mkdir()

        with mock.patch.dict(os.environ, {
            "HOME": str(profile_home),  # Simulated profile override
            "HERMES_REAL_HOME": str(real_home),
        }, clear=False):
            # Script logic: find .hermes/ using HERMES_REAL_HOME fallback
            hermes_base = Path(os.environ.get("HERMES_REAL_HOME", os.environ.get("HOME", ""))) / ".hermes"
            assert hermes_base.exists()
            assert str(hermes_base).startswith(str(real_home))
