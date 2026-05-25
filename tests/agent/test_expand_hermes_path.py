"""Regression tests for expand_hermes_path subprocess home resolution.

See fix #12260 — skill config paths like ~/my-data were resolving to
/root/my-data instead of ${HERMES_HOME}/home/my-data when profile
isolation is active.
"""

import os
from unittest.mock import patch

import pytest

from hermes_constants import expand_hermes_path


class TestExpandHermesPath:
    """Tests for expand_hermes_path using subprocess home resolution."""

    def test_uses_subprocess_home(self):
        """When get_subprocess_home() returns a profile dir, ~/foo resolves there."""
        with patch("hermes_constants.get_subprocess_home", return_value="/data/hermes/home"):
            result = expand_hermes_path("~/foo")
            assert result == "/data/hermes/home/foo"

    def test_uses_subprocess_home_bare_tilde(self):
        """When get_subprocess_home() returns a profile dir, ~ alone resolves there."""
        with patch("hermes_constants.get_subprocess_home", return_value="/data/hermes/home"):
            result = expand_hermes_path("~")
            assert result == "/data/hermes/home"

    def test_falls_through_without_profile(self):
        """When get_subprocess_home() returns None, ~/foo behaves like os.path.expanduser."""
        with patch("hermes_constants.get_subprocess_home", return_value=None):
            result = expand_hermes_path("~/foo")
            expected = os.path.expanduser("~/foo")
            assert result == expected

    def test_handles_env_vars(self):
        """Environment variables are expanded in the path."""
        env_vars = {"MY_DIR": "/custom/path"}
        with patch.dict(os.environ, env_vars, clear=False):
            with patch("hermes_constants.get_subprocess_home", return_value=None):
                result = expand_hermes_path("$MY_DIR/file")
                assert result == "/custom/path/file"

    def test_ignores_named_user_tilde(self):
        """~otheruser/ does NOT use the profile home — it falls through to expanduser."""
        with patch("hermes_constants.get_subprocess_home", return_value="/data/hermes/home"):
            result = expand_hermes_path("~otheruser/foo")
            # Should NOT be /data/hermes/home/otheruser/foo
            # Should behave like normal expanduser for named users
            expected = os.path.expanduser("~otheruser/foo")
            assert result == expected
            # Explicitly assert it's NOT the profile home prefix
            assert not result.startswith("/data/hermes/home/otheruser")
