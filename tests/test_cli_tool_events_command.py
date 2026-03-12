"""Tests for the `/tool-events` slash command in the interactive CLI."""

import json
from unittest.mock import MagicMock

from cli import HermesCLI


class TestToolEventsCommand:
    def _make_cli(self):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.agent = MagicMock()
        cli_obj.console = MagicMock()
        cli_obj.session_id = "session-123"
        cli_obj._session_db = None
        return cli_obj

    def test_tool_events_shows_recent_events(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {
                "ts": "2026-03-12T00:00:00Z",
                "tool": "web_search",
                "tool_call_id": "c1",
                "duration_ms": 88,
                "ok": True,
                "envelope": True,
                "retryable": False,
                "error_type": None,
                "error_summary": "",
            }
        ]

        cli_obj.process_command("/tool-events 5")

        cli_obj.agent.get_recent_tool_execution_events.assert_called_once_with(
            limit=5,
            tool_name=None,
            ok=None,
            retryable=None,
        )
        output = capsys.readouterr().out
        assert "Tool Execution Events" in output
        assert "web_search" in output
        assert "88ms" in output

    def test_tool_events_supports_filters(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {
                "ts": "2026-03-12T00:00:01Z",
                "tool": "browser_click",
                "tool_call_id": "c2",
                "duration_ms": 23,
                "ok": False,
                "retryable": True,
                "error_summary": "missing ref",
            }
        ]

        cli_obj.process_command("/tool-events 7 errors retryable tool=browser_click")

        cli_obj.agent.get_recent_tool_execution_events.assert_called_once_with(
            limit=7,
            tool_name="browser_click",
            ok=False,
            retryable=True,
        )
        output = capsys.readouterr().out
        assert "browser_click" in output
        assert "missing ref" in output

    def test_tool_events_json_output(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {
                "ts": "2026-03-12T00:00:02Z",
                "tool": "web_search",
                "tool_call_id": "c3",
                "duration_ms": 12,
                "ok": True,
            }
        ]

        cli_obj.process_command("/tool-events json 1")

        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed[0]["tool"] == "web_search"

    def test_tool_events_export_writes_json_file(self, tmp_path, capsys, monkeypatch):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {
                "ts": "2026-03-12T00:00:03Z",
                "tool": "web_search",
                "tool_call_id": "c4",
                "duration_ms": 9,
                "ok": True,
            }
        ]

        monkeypatch.chdir(tmp_path)
        cli_obj.process_command("/tool-events export 1")

        output = capsys.readouterr().out
        assert "Exported 1 tool events" in output
        exported_files = list(tmp_path.glob("tool-events-*.json"))
        assert len(exported_files) == 1
        parsed = json.loads(exported_files[0].read_text(encoding="utf-8"))
        assert parsed[0]["tool_call_id"] == "c4"

    def test_tool_events_stats_mode_shows_percentiles_and_error_rate(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {"ts": "2026-03-12T00:00:05Z", "tool": "browser_click", "tool_call_id": "c5", "duration_ms": 10, "ok": False},
            {"ts": "2026-03-12T00:00:04Z", "tool": "browser_click", "tool_call_id": "c4", "duration_ms": 40, "ok": True},
            {"ts": "2026-03-12T00:00:03Z", "tool": "web_search", "tool_call_id": "c3", "duration_ms": 50, "ok": True},
            {"ts": "2026-03-12T00:00:02Z", "tool": "web_search", "tool_call_id": "c2", "duration_ms": 30, "ok": True},
            {"ts": "2026-03-12T00:00:01Z", "tool": "web_search", "tool_call_id": "c1", "duration_ms": 20, "ok": False},
        ]

        cli_obj.process_command("/tool-events stats 20")

        output = capsys.readouterr().out
        assert "Tool Event Stats" in output
        assert "web_search" in output
        assert "browser_click" in output
        assert "33.3%" in output  # web_search error rate (1/3 failed)
        assert "50.0%" in output  # browser_click error rate (1/2 failed)
        assert "p50=30ms" in output
        assert "p95=50ms" in output

    def test_tool_events_stats_json_output_is_machine_readable(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {"ts": "2026-03-12T00:00:05Z", "tool": "browser_click", "tool_call_id": "c5", "duration_ms": 10, "ok": False},
            {"ts": "2026-03-12T00:00:04Z", "tool": "browser_click", "tool_call_id": "c4", "duration_ms": 40, "ok": True},
            {"ts": "2026-03-12T00:00:03Z", "tool": "web_search", "tool_call_id": "c3", "duration_ms": 50, "ok": True},
        ]

        cli_obj.process_command("/tool-events stats json 10")

        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["mode"] == "stats"
        assert parsed["bucket_by"] == "tool"
        assert parsed["total_events"] == 3
        assert parsed["buckets"][0]["count"] >= 1

    def test_tool_events_stats_window_bucketing(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {"ts": "2026-03-12T00:00:06Z", "tool": "browser_click", "tool_call_id": "c6", "duration_ms": 12, "ok": True},
            {"ts": "2026-03-12T00:00:05Z", "tool": "browser_click", "tool_call_id": "c5", "duration_ms": 10, "ok": False},
            {"ts": "2026-03-12T00:00:04Z", "tool": "browser_click", "tool_call_id": "c4", "duration_ms": 40, "ok": True},
            {"ts": "2026-03-12T00:00:03Z", "tool": "web_search", "tool_call_id": "c3", "duration_ms": 50, "ok": True},
            {"ts": "2026-03-12T00:00:02Z", "tool": "web_search", "tool_call_id": "c2", "duration_ms": 30, "ok": True},
            {"ts": "2026-03-12T00:00:01Z", "tool": "web_search", "tool_call_id": "c1", "duration_ms": 20, "ok": False},
        ]

        cli_obj.process_command("/tool-events stats bucket=window window=2 20")

        output = capsys.readouterr().out
        assert "bucket=latest_1_2" in output
        assert "bucket=latest_3_4" in output
        assert "bucket=latest_5_6" in output

    def test_tool_events_shows_last_failure_replay_hint(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.agent.get_recent_tool_execution_events.return_value = [
            {
                "ts": "2026-03-12T00:00:02Z",
                "tool": "web_search",
                "tool_call_id": "ok-1",
                "duration_ms": 8,
                "ok": True,
                "args_preview": '{"query":"ok"}',
            },
            {
                "ts": "2026-03-12T00:00:01Z",
                "tool": "browser_click",
                "tool_call_id": "err-1",
                "duration_ms": 12,
                "ok": False,
                "error_summary": "missing ref",
                "args_preview": '{"ref":"@e9"}',
            },
        ]

        cli_obj.process_command("/tool-events 5")

        output = capsys.readouterr().out
        assert "Last failure replay hint" in output
        assert "browser_click" in output
        assert "tool=browser_click" in output
        assert "args={\"ref\":\"@e9\"}" in output

    def test_tool_events_without_agent_support_prints_guidance(self, capsys):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.agent = None
        cli_obj.console = MagicMock()
        cli_obj._session_db = None

        cli_obj.process_command("/tool-events")

        output = capsys.readouterr().out
        assert "unavailable" in output.lower()
