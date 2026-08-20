"""Regression tests for loading feedback on slow slash commands."""

import threading
import time
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

    def test_skills_command_sets_busy_state_and_prints_status(self, capsys):
        cli_obj = self._make_cli()
        seen = {}

        def fake_handle(cmd: str):
            seen["cmd"] = cmd
            seen["running"] = cli_obj._command_running
            seen["status"] = cli_obj._command_status
            print("skills done")

        with patch.object(cli_obj, "_handle_skills_command", side_effect=fake_handle), \
             patch.object(cli_obj, "_invalidate") as invalidate_mock:
            assert cli_obj.process_command("/skills search kubernetes")

        output = capsys.readouterr().out
        assert "⏳ Searching skills..." in output
        assert "skills done" in output
        assert seen == {
            "cmd": "/skills search kubernetes",
            "running": True,
            "status": "Searching skills...",
        }
        assert cli_obj._command_running is False
        assert cli_obj._command_status == ""
        assert invalidate_mock.call_count == 2

    def test_reload_mcp_sets_busy_state_and_prints_status(self, capsys):
        cli_obj = self._make_cli()
        seen = {}

        def fake_reload():
            seen["running"] = cli_obj._command_running
            seen["status"] = cli_obj._command_status
            seen["thread_id"] = threading.get_ident()
            print("reload done")

        # /reload-mcp now wraps the actual reload in a prompt-cache-invalidation
        # confirmation prompt (commit 4d7fc0f37).  This test exercises the
        # loading-indicator path, not the confirmation UX, so pre-approve the
        # reload via config so the handler goes straight into _reload_mcp().
        fake_cfg = {"approvals": {"mcp_reload_confirm": False}}

        with patch.object(cli_obj, "_reload_mcp", side_effect=fake_reload), \
             patch.object(cli_obj, "_invalidate") as invalidate_mock, \
             patch("cli.load_cli_config", return_value=fake_cfg):
            assert cli_obj.process_command("/reload-mcp")

        output = capsys.readouterr().out
        assert "⏳ Reloading MCP servers..." in output
        assert "reload done" in output
        assert seen == {
            "running": True,
            "status": "Reloading MCP servers...",
            "thread_id": seen["thread_id"],
        }
        assert seen["thread_id"] != threading.get_ident()
        assert cli_obj._command_running is False
        assert cli_obj._command_status == ""
        assert invalidate_mock.call_count == 2

    def test_reload_mcp_timeout_warns_without_blocking_forever(self, capsys):
        cli_obj = self._make_cli()
        stop_event = threading.Event()

        def fake_reload():
            stop_event.wait(0.1)

        with patch.object(cli_obj, "_reload_mcp", side_effect=fake_reload), \
             patch.object(cli_obj, "_invalidate"):
            start = time.monotonic()
            result = cli_obj._run_reload_mcp_with_timeout(timeout_seconds=0.01)
            elapsed = time.monotonic() - start

        stop_event.set()

        output = capsys.readouterr().out
        assert result is False
        assert elapsed < 0.05
        assert "MCP reload timed out (0.01s)." in output
