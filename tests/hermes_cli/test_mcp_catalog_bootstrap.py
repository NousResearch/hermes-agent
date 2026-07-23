"""Regression tests for MCP catalog subprocess guards."""

import subprocess
from unittest import mock

import pytest

from hermes_cli.mcp_catalog import CatalogError, _run_bootstrap


class TestRunBootstrapTimeout:
    """Test that _run_bootstrap converts TimeoutExpired to CatalogError."""

    def test_timeout_raises_catalog_error(self, tmp_path):
        """When subprocess.run raises TimeoutExpired, _run_bootstrap raises
        CatalogError with 'timed out' in the message."""
        with mock.patch("hermes_cli.mcp_catalog.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="echo test", timeout=300)

            with pytest.raises(CatalogError, match="timed out"):
                _run_bootstrap(tmp_path, ["echo test"])

    def test_nonzero_exit_raises_catalog_error(self, tmp_path):
        """When subprocess.run returns non-zero, _run_bootstrap raises
        CatalogError with the exit code."""
        with mock.patch("hermes_cli.mcp_catalog.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1)

            with pytest.raises(CatalogError, match="exit 1"):
                _run_bootstrap(tmp_path, ["false"])

    def test_success_returns_normally(self, tmp_path):
        """When subprocess.run returns 0, _run_bootstrap completes normally."""
        with mock.patch("hermes_cli.mcp_catalog.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)

            _run_bootstrap(tmp_path, ["true"])
            mock_run.assert_called_once()

    def test_subprocess_kwargs_include_timeout_and_devnull(self, tmp_path):
        """Verify that subprocess.run is called with timeout=300 and
        stdin=subprocess.DEVNULL."""
        with mock.patch("hermes_cli.mcp_catalog.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)

            _run_bootstrap(tmp_path, ["true"])

            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["timeout"] == 300
            assert call_kwargs["stdin"] is subprocess.DEVNULL
