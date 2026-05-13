"""Tests for MCP environment variable interpolation warnings."""
import os
from unittest.mock import patch

import pytest


class TestMcpEnvVarWarning:
    """_interpolate_value should warn when env vars resolve to empty strings."""

    def test_warns_on_missing_env_var(self, capsys):
        """Interpolating a missing env var should print a warning to stdout."""
        from hermes_cli.mcp_config import _interpolate_value

        # Ensure the variable is not set
        os.environ.pop("_HERMES_TEST_MISSING_VAR", None)

        result = _interpolate_value("Bearer ${_HERMES_TEST_MISSING_VAR}")

        # Should still return empty for the var (backward compatible)
        assert result == "Bearer "
        # Warning appears on stdout (inline with other CLI diagnostic output)
        captured = capsys.readouterr()
        assert "_HERMES_TEST_MISSING_VAR" in captured.out

    def test_no_warning_when_var_exists(self, capsys):
        """No warning should be emitted when the env var is set."""
        from hermes_cli.mcp_config import _interpolate_value

        os.environ["_HERMES_TEST_SET_VAR"] = "my-key-123"
        try:
            result = _interpolate_value("Bearer ${_HERMES_TEST_SET_VAR}")

            assert result == "Bearer my-key-123"
            captured = capsys.readouterr()
            assert "_HERMES_TEST_SET_VAR" not in captured.out
        finally:
            del os.environ["_HERMES_TEST_SET_VAR"]

    def test_no_warning_when_var_intentionally_empty(self, capsys):
        """An env var explicitly set to empty string must NOT trigger a warning.

        This is the key false-positive guard: a user who intentionally sets
        VAR="" (e.g. to disable auth for a local dev server) should not see
        a spurious warning.  Only a completely absent variable should warn.
        """
        from hermes_cli.mcp_config import _interpolate_value

        os.environ["_HERMES_TEST_EMPTY_VAR"] = ""
        try:
            result = _interpolate_value("Bearer ${_HERMES_TEST_EMPTY_VAR}")

            # Value resolves to the empty string — that's the user's intent
            assert result == "Bearer "
            # No warning should be emitted for an intentionally empty var
            captured = capsys.readouterr()
            assert "_HERMES_TEST_EMPTY_VAR" not in captured.out
        finally:
            del os.environ["_HERMES_TEST_EMPTY_VAR"]

    def test_warning_includes_server_name(self, capsys):
        """Warning message should include the server name when provided."""
        from hermes_cli.mcp_config import _interpolate_value

        os.environ.pop("_HERMES_TEST_MISSING_VAR", None)

        _interpolate_value(
            "Bearer ${_HERMES_TEST_MISSING_VAR}", server_name="my-mcp-server"
        )

        captured = capsys.readouterr()
        assert "_HERMES_TEST_MISSING_VAR" in captured.out
        assert "my-mcp-server" in captured.out

    def test_no_warning_when_no_interpolation_patterns(self, capsys):
        """Plain strings with no ${VAR} patterns must not emit any warnings."""
        from hermes_cli.mcp_config import _interpolate_value

        result = _interpolate_value("Bearer static-token-value")

        assert result == "Bearer static-token-value"
        captured = capsys.readouterr()
        assert captured.out == ""
