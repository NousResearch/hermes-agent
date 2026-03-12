"""Tests for the `/honcho` slash command in the interactive CLI."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


class TestHonchoCommand:
    def _make_cli(self):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.agent = MagicMock()
        cli_obj.agent.get_honcho_status.return_value = {
            "enabled": True,
            "session_key": "telegram:123",
            "fail_open": True,
            "retry_attempts": 2,
            "tool_events_enabled": True,
            "max_pending_sync": 200,
            "max_pending_tool_events": 400,
            "flush_batch_size": 25,
            "pending_sync": 0,
            "pending_tool_events": 0,
            "last_error": "",
            "telemetry": {"dropped_sync": 0, "dropped_tool_events": 0},
        }
        cli_obj.console = MagicMock()
        return cli_obj

    def test_honcho_status(self, capsys):
        cli_obj = self._make_cli()

        cli_obj.process_command("/honcho")

        output = capsys.readouterr().out
        assert "Honcho Runtime" in output
        assert "telegram:123" in output

    def test_honcho_strict_updates_runtime_and_config(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.set_honcho_runtime_mode.return_value = {
            "fail_open": False,
            "retry_attempts": 2,
            "tool_events_enabled": True,
        }

        with patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/honcho strict")

        cli_obj.agent.set_honcho_runtime_mode.assert_called_once_with(fail_open=False)
        save_mock.assert_called_once_with("honcho.fail_open", False)
        output = capsys.readouterr().out
        assert "strict" in output.lower()

    def test_honcho_retries_updates_runtime_and_config(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.set_honcho_runtime_mode.return_value = {
            "retry_attempts": 5,
            "tool_events_enabled": True,
            "fail_open": True,
        }

        with patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/honcho retries 5")

        cli_obj.agent.set_honcho_runtime_mode.assert_called_once_with(retry_attempts=5)
        save_mock.assert_called_once_with("honcho.retry_attempts", 5)
        output = capsys.readouterr().out
        assert "retries set to 5" in output.lower()

    def test_honcho_tool_events_toggle(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.set_honcho_runtime_mode.return_value = {
            "tool_events_enabled": False,
            "retry_attempts": 2,
            "fail_open": True,
        }

        with patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/honcho tool-events off")

        cli_obj.agent.set_honcho_runtime_mode.assert_called_once_with(tool_events_enabled=False)
        save_mock.assert_called_once_with("honcho.tool_events.enabled", False)
        output = capsys.readouterr().out
        assert "disabled" in output.lower()

    def test_honcho_flush_runs_manual_queue_replay(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.flush_honcho_queues.return_value = {
            "flushed_sync": 2,
            "flushed_tool_events": 3,
            "pending_sync": 1,
            "pending_tool_events": 0,
        }

        cli_obj.process_command("/honcho flush")

        cli_obj.agent.flush_honcho_queues.assert_called_once_with()
        output = capsys.readouterr().out
        assert "flush complete" in output.lower()
        assert "replayed sync=2" in output.lower()
