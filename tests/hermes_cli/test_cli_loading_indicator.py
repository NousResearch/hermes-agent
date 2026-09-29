"""Regression tests for loading feedback on slow slash commands."""

from unittest.mock import patch

from cli import HermesCLI

class TestCLILoadingIndicator:
    def _make_cli(self):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj._app = None
        cli_obj._last_invalidate = 0.0
        cli_obj._command_running = False
        cli_obj._command_status = ""
        return cli_obj

    def test_removed_skills_command_does_not_start_background_work(self):
        from unittest.mock import Mock
        cli_obj = self._make_cli()
        cli_obj.console = Mock()
        with patch.object(cli_obj, "_handle_skills_command") as handle:
            assert cli_obj.process_command("/skills search kubernetes")
        handle.assert_not_called()
        assert cli_obj._command_running is False
        cli_obj.console.print.assert_called()
