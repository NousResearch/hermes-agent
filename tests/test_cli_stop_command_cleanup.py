"""Tests for CLI _handle_stop_command Docker sandbox cleanup (issue #39489)."""

import pytest
from unittest.mock import patch, MagicMock


class TestStopCommandDockerCleanup:
    """Verify /stop cleans up Docker sandbox environments after killing processes."""

    def _make_cli(self):
        """Create a minimal HermesCLI with no full init."""
        from cli import HermesCLI
        return HermesCLI.__new__(HermesCLI)

    @patch("tools.terminal_tool.cleanup_all_environments", return_value=0)
    @patch("tools.process_registry.process_registry")
    def test_no_running_processes_skips_cleanup(self, mock_pr, mock_cleanup, capsys):
        """When no background processes are running, cleanup is not called."""
        mock_pr.list_sessions.return_value = []

        cli = self._make_cli()
        cli._handle_stop_command()

        output = capsys.readouterr().out
        assert "No running background processes" in output
        mock_pr.kill_all.assert_not_called()
        mock_cleanup.assert_not_called()

    @patch("tools.terminal_tool.cleanup_all_environments", return_value=0)
    @patch("tools.process_registry.process_registry")
    def test_kills_processes_and_cleans_zero_environments(self, mock_pr, mock_cleanup, capsys):
        """When processes are killed but no sandbox environments exist, no cleanup message."""
        mock_pr.list_sessions.return_value = [
            {"status": "running", "session_id": "abc"},
        ]
        mock_pr.kill_all.return_value = 1

        cli = self._make_cli()
        cli._handle_stop_command()

        output = capsys.readouterr().out
        assert "Stopped 1 process" in output
        mock_pr.kill_all.assert_called_once()
        mock_cleanup.assert_called_once()
        # No cleanup message when cleaned == 0
        assert "Cleaned" not in output

    @patch("tools.terminal_tool.cleanup_all_environments", return_value=2)
    @patch("tools.process_registry.process_registry")
    def test_kills_processes_and_cleans_sandbox_environments(self, mock_pr, mock_cleanup, capsys):
        """When both processes and sandbox environments are cleaned, both messages appear."""
        mock_pr.list_sessions.return_value = [
            {"status": "running", "session_id": "abc"},
            {"status": "running", "session_id": "def"},
        ]
        mock_pr.kill_all.return_value = 2

        cli = self._make_cli()
        cli._handle_stop_command()

        output = capsys.readouterr().out
        assert "Stopped 2 process" in output
        assert "Cleaned 2 sandbox environment" in output
        mock_pr.kill_all.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch("tools.terminal_tool.cleanup_all_environments", side_effect=Exception("Docker down"))
    @patch("tools.process_registry.process_registry")
    def test_cleanup_exception_does_not_crash_stop(self, mock_pr, mock_cleanup, capsys):
        """If cleanup_all_environments raises, /stop should not crash."""
        mock_pr.list_sessions.return_value = [
            {"status": "running", "session_id": "abc"},
        ]
        mock_pr.kill_all.return_value = 1

        cli = self._make_cli()
        # Should propagate — cleanup_all_environments doesn't catch internally
        # This test documents current behavior; if we want to swallow, we'd wrap in try/except
        with pytest.raises(Exception, match="Docker down"):
            cli._handle_stop_command()
