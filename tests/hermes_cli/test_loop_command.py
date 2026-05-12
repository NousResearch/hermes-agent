"""Tests for /loop CLI command.

Verifies that the CLI handler correctly parses schedules, prompts, and
subcommands, and calls the cronjob tool with the right parameters.
"""

import json
from unittest.mock import patch, MagicMock

import pytest


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_cli():
    """Build a minimal HermesCLI stub with just the loop handler."""
    from cli import HermesCLI
    cli = object.__new__(HermesCLI)
    cli._busy_command = lambda self, ctx: ctx
    cli._slow_command_status = lambda self, cmd: None
    return cli


def _mock_cron_api(success=True, **extra):
    """Return a mock cronjob_tool that returns JSON with given fields."""
    def _cron_tool(**kwargs):
        result = {"success": success, **extra}
        return json.dumps(result)
    return _cron_tool


# ------------------------------------------------------------------
# CLI /loop handler tests
# ------------------------------------------------------------------

class TestLoopCommandCLI:

    def test_dispatch_calls_handler(self):
        """process_command dispatches /loop to _handle_loop_command."""
        from hermes_cli.commands import resolve_command
        cmd_def = resolve_command("/loop")
        assert cmd_def is not None
        assert cmd_def.name == "loop"

    def test_alias_repeat_resolves(self):
        """The 'repeat' alias resolves to the loop command."""
        from hermes_cli.commands import resolve_command
        cmd_def = resolve_command("/repeat")
        assert cmd_def is not None
        assert cmd_def.name == "loop"

    def test_no_args_shows_usage(self, capsys):
        """/loop with no args prints usage banner and lists jobs."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(jobs=[])):
            cli._handle_loop_command("/loop")

        captured = capsys.readouterr()
        assert "(^_^) /loop" in captured.out
        assert "Usage:" in captured.out
        assert "No loop jobs" in captured.out

    def test_create_parses_schedule_and_prompt(self, capsys):
        """/loop 5m check deployment creates a job with schedule=5m prompt=check deployment."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({
                "success": True,
                "job_id": "loop_abc123",
                "schedule": "5m",
                "next_run_at": "2026-05-09T17:00:00Z",
            })

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            cli._handle_loop_command("/loop 5m check deployment")

        captured = capsys.readouterr()
        assert len(calls) == 1
        assert calls[0]["action"] == "create"
        assert calls[0]["schedule"] == "5m"
        assert calls[0]["prompt"] == "check deployment"
        assert calls[0]["name"].startswith("loop:")
        assert calls[0]["deliver"] == "origin"
        assert "Loop job created" in captured.out

    def test_create_every_syntax(self, capsys):
        """/loop every 30m summarize news passes schedule through."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "job_id": "loop_def456", "schedule": "every 30m", "next_run_at": "2026-05-09T17:30:00Z"})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            cli._handle_loop_command('/loop every 30m "summarize news"')

        assert len(calls) == 1
        assert calls[0]["schedule"] == "every 30m"
        assert calls[0]["prompt"] == "summarize news"

    def test_create_empty_prompt_error(self, capsys):
        """/loop 5m with no prompt shows usage error."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api()):
            cli._handle_loop_command("/loop 5m")

        captured = capsys.readouterr()
        assert "Usage:" in captured.out
        assert "Example" in captured.out

    def test_list_subcommand(self, capsys):
        """/loop list filters jobs by name prefix loop:."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        jobs = [
            {"job_id": "j1", "name": "loop: check deployment", "schedule": "5m", "state": "active", "prompt_preview": "check deployment", "next_run_at": "2026-05-09T17:00:00Z"},
            {"job_id": "j2", "name": "regular job", "schedule": "1h", "state": "active"},
        ]

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(jobs=jobs)):
            cli._handle_loop_command("/loop list")

        captured = capsys.readouterr()
        assert "Loop Jobs:" in captured.out
        assert "j1" in captured.out
        assert "regular job" not in captured.out  # filtered out

    def test_list_no_jobs(self, capsys):
        """/loop list with no loop jobs prints friendly message."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(jobs=[])):
            cli._handle_loop_command("/loop list")

        captured = capsys.readouterr()
        assert "No loop jobs found" in captured.out

    def test_pause_subcommand(self, capsys):
        """/loop pause <id> calls cronjob action=pause."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "job": {"name": "loop: test", "job_id": "abc"}})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            cli._handle_loop_command("/loop pause abc123")

        captured = capsys.readouterr()
        assert len(calls) == 1
        assert calls[0]["action"] == "pause"
        assert calls[0]["job_id"] == "abc123"
        assert calls[0]["reason"] == "paused from /loop"
        assert "Paused loop job" in captured.out

    def test_pause_missing_id(self, capsys):
        """/loop pause without id prints usage."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api()):
            cli._handle_loop_command("/loop pause")

        captured = capsys.readouterr()
        assert "Usage: /loop pause" in captured.out

    def test_resume_subcommand(self, capsys):
        """/loop resume <id> calls cronjob action=resume."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "job": {"name": "loop: test", "next_run_at": "2026-05-09T18:00:00Z"}})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            cli._handle_loop_command("/loop resume abc123")

        captured = capsys.readouterr()
        assert len(calls) == 1
        assert calls[0]["action"] == "resume"
        assert calls[0]["job_id"] == "abc123"
        assert "Resumed loop job" in captured.out

    def test_remove_subcommand(self, capsys):
        """/loop remove <id> calls cronjob action=remove."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "removed_job": {"name": "loop: test"}})

        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            cli._handle_loop_command("/loop remove abc123")

        captured = capsys.readouterr()
        assert len(calls) == 1
        assert calls[0]["action"] == "remove"
        assert calls[0]["job_id"] == "abc123"
        assert "Removed loop job" in captured.out

    def test_remove_missing_id(self, capsys):
        """/loop remove without id prints usage."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api()):
            cli._handle_loop_command("/loop remove")

        captured = capsys.readouterr()
        assert "Usage: /loop remove" in captured.out

    def test_create_failure(self, capsys):
        """Failed create prints error message."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        with patch("tools.cronjob_tools.cronjob", side_effect=_mock_cron_api(success=False, error="Invalid schedule")):
            cli._handle_loop_command("/loop bad check deployment")

        captured = capsys.readouterr()
        assert "Failed to create loop" in captured.out
        assert "Invalid schedule" in captured.out

    def test_prompt_truncated_in_name(self, capsys):
        """Very long prompt is truncated in job name but full prompt is passed."""
        from cli import HermesCLI
        cli = object.__new__(HermesCLI)

        calls = []
        def _capture_cronjob(**kwargs):
            calls.append(kwargs)
            return json.dumps({"success": True, "job_id": "loop_xyz", "schedule": "1h", "next_run_at": "2026-05-09T18:00:00Z"})

        long_prompt = "a" * 100
        with patch("tools.cronjob_tools.cronjob", side_effect=_capture_cronjob):
            cli._handle_loop_command(f"/loop 1h {long_prompt}")

        assert calls[0]["prompt"] == long_prompt
        assert calls[0]["name"] == f"loop: {'a' * 50}..."
