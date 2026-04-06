"""Tests for hermes_cli/logs.py — log viewing and filtering."""

import os
import textwrap
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.logs import (
    LOG_FILES,
    _extract_level,
    _matches_filters,
    _parse_line_timestamp,
    _parse_since,
    _read_last_n_lines,
    list_logs,
    tail_log,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Create a fake HERMES_HOME with a logs/ directory."""
    home = Path(os.environ["HERMES_HOME"])
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


@pytest.fixture
def sample_agent_log(log_dir):
    """Write a realistic agent.log with mixed levels and sessions."""
    lines = textwrap.dedent("""\
        2026-04-05 10:00:00,000 INFO run_agent: conversation turn: session=sess_aaa model=claude provider=openrouter platform=cli history=0 msg='hello'
        2026-04-05 10:00:01,000 INFO run_agent: tool terminal completed (0.50s, 200 chars)
        2026-04-05 10:00:02,000 INFO run_agent: API call #1: model=claude provider=openrouter in=1000 out=200 total=1200 latency=1.5s
        2026-04-05 10:00:03,000 WARNING run_agent: Tool web_search returned error (2.00s): timeout
        2026-04-05 10:00:04,000 INFO run_agent: conversation turn: session=sess_bbb model=gpt-5 provider=openai platform=telegram history=5 msg='fix bug'
        2026-04-05 10:00:05,000 ERROR run_agent: API call failed after 3 retries. rate limited
        2026-04-05 10:00:06,000 INFO run_agent: tool read_file completed (0.01s, 500 chars)
        2026-04-05 10:00:07,000 DEBUG run_agent: verbose internal detail
        2026-04-05 10:00:08,000 INFO credential_pool: credential pool: marking key-1 exhausted (status=429), rotating
        2026-04-05 10:00:09,000 INFO credential_pool: credential pool: rotated to key-2
    """)
    path = log_dir / "agent.log"
    path.write_text(lines)
    return path


@pytest.fixture
def sample_errors_log(log_dir):
    """Write a small errors.log."""
    lines = textwrap.dedent("""\
        2026-04-05 10:00:03,000 WARNING run_agent: Tool web_search returned error (2.00s): timeout
        2026-04-05 10:00:05,000 ERROR run_agent: API call failed after 3 retries. rate limited
    """)
    path = log_dir / "errors.log"
    path.write_text(lines)
    return path


# ---------------------------------------------------------------------------
# _parse_since
# ---------------------------------------------------------------------------

class TestParseSince:
    def test_hours(self):
        cutoff = _parse_since("2h")
        assert cutoff is not None
        assert (datetime.now() - cutoff).total_seconds() == pytest.approx(7200, abs=5)

    def test_minutes(self):
        cutoff = _parse_since("30m")
        assert cutoff is not None
        assert (datetime.now() - cutoff).total_seconds() == pytest.approx(1800, abs=5)

    def test_days(self):
        cutoff = _parse_since("1d")
        assert cutoff is not None
        assert (datetime.now() - cutoff).total_seconds() == pytest.approx(86400, abs=5)

    def test_seconds(self):
        cutoff = _parse_since("60s")
        assert cutoff is not None
        assert (datetime.now() - cutoff).total_seconds() == pytest.approx(60, abs=5)

    def test_invalid_returns_none(self):
        assert _parse_since("abc") is None
        assert _parse_since("") is None
        assert _parse_since("10x") is None

    def test_whitespace_handling(self):
        cutoff = _parse_since("  1h  ")
        assert cutoff is not None


# ---------------------------------------------------------------------------
# _parse_line_timestamp
# ---------------------------------------------------------------------------

class TestParseLineTimestamp:
    def test_standard_format(self):
        ts = _parse_line_timestamp("2026-04-05 10:00:00,123 INFO something")
        assert ts is not None
        assert ts.year == 2026
        assert ts.hour == 10

    def test_no_timestamp(self):
        assert _parse_line_timestamp("just some text") is None

    def test_continuation_line(self):
        assert _parse_line_timestamp("    at module.function (line 42)") is None


# ---------------------------------------------------------------------------
# _extract_level
# ---------------------------------------------------------------------------

class TestExtractLevel:
    def test_info(self):
        assert _extract_level("2026-04-05 10:00:00 INFO run_agent: something") == "INFO"

    def test_warning(self):
        assert _extract_level("2026-04-05 10:00:00 WARNING run_agent: bad") == "WARNING"

    def test_error(self):
        assert _extract_level("2026-04-05 10:00:00 ERROR run_agent: crash") == "ERROR"

    def test_debug(self):
        assert _extract_level("2026-04-05 10:00:00 DEBUG run_agent: detail") == "DEBUG"

    def test_no_level(self):
        assert _extract_level("just a plain line") is None


# ---------------------------------------------------------------------------
# _matches_filters
# ---------------------------------------------------------------------------

class TestMatchesFilters:
    def test_no_filters_always_matches(self):
        assert _matches_filters("any line") is True

    def test_level_filter_passes(self):
        assert _matches_filters(
            "2026-04-05 10:00:00 WARNING something",
            min_level="WARNING",
        ) is True

    def test_level_filter_rejects(self):
        assert _matches_filters(
            "2026-04-05 10:00:00 INFO something",
            min_level="WARNING",
        ) is False

    def test_session_filter_passes(self):
        assert _matches_filters(
            "session=sess_aaa model=claude",
            session_filter="sess_aaa",
        ) is True

    def test_session_filter_rejects(self):
        assert _matches_filters(
            "session=sess_aaa model=claude",
            session_filter="sess_bbb",
        ) is False

    def test_since_filter_passes(self):
        # Line from the future should always pass
        assert _matches_filters(
            "2099-01-01 00:00:00 INFO future",
            since=datetime.now(),
        ) is True

    def test_since_filter_rejects(self):
        assert _matches_filters(
            "2020-01-01 00:00:00 INFO past",
            since=datetime.now(),
        ) is False

    def test_combined_filters(self):
        line = "2099-01-01 00:00:00 WARNING run_agent: session=abc error"
        assert _matches_filters(
            line, min_level="WARNING", session_filter="abc",
            since=datetime.now(),
        ) is True
        # Fails session filter
        assert _matches_filters(
            line, min_level="WARNING", session_filter="xyz",
        ) is False


# ---------------------------------------------------------------------------
# _read_last_n_lines
# ---------------------------------------------------------------------------

class TestReadLastNLines:
    def test_reads_correct_count(self, sample_agent_log):
        lines = _read_last_n_lines(sample_agent_log, 3)
        assert len(lines) == 3

    def test_reads_all_when_fewer(self, sample_agent_log):
        lines = _read_last_n_lines(sample_agent_log, 100)
        assert len(lines) == 10  # sample has 10 lines

    def test_empty_file(self, log_dir):
        empty = log_dir / "empty.log"
        empty.write_text("")
        lines = _read_last_n_lines(empty, 10)
        assert lines == []

    def test_last_line_content(self, sample_agent_log):
        lines = _read_last_n_lines(sample_agent_log, 1)
        assert "rotated to key-2" in lines[0]


# ---------------------------------------------------------------------------
# tail_log
# ---------------------------------------------------------------------------

class TestTailLog:
    def test_basic_tail(self, sample_agent_log, capsys):
        tail_log("agent", num_lines=3)
        captured = capsys.readouterr()
        assert "agent.log" in captured.out
        # Should have the header + 3 lines
        lines = captured.out.strip().split("\n")
        assert len(lines) == 4  # 1 header + 3 content

    def test_level_filter(self, sample_agent_log, capsys):
        tail_log("agent", num_lines=50, level="ERROR")
        captured = capsys.readouterr()
        assert "level>=ERROR" in captured.out
        # Only the ERROR line should appear
        content_lines = [l for l in captured.out.strip().split("\n") if not l.startswith("---")]
        assert len(content_lines) == 1
        assert "API call failed" in content_lines[0]

    def test_session_filter(self, sample_agent_log, capsys):
        tail_log("agent", num_lines=50, session="sess_bbb")
        captured = capsys.readouterr()
        content_lines = [l for l in captured.out.strip().split("\n") if not l.startswith("---")]
        assert len(content_lines) == 1
        assert "sess_bbb" in content_lines[0]

    def test_errors_log(self, sample_errors_log, capsys):
        tail_log("errors", num_lines=10)
        captured = capsys.readouterr()
        assert "errors.log" in captured.out
        assert "WARNING" in captured.out or "ERROR" in captured.out

    def test_unknown_log_exits(self):
        with pytest.raises(SystemExit):
            tail_log("nonexistent")

    def test_missing_file_exits(self, log_dir):
        with pytest.raises(SystemExit):
            tail_log("agent")  # agent.log doesn't exist in clean log_dir


# ---------------------------------------------------------------------------
# list_logs
# ---------------------------------------------------------------------------

class TestListLogs:
    def test_lists_files(self, sample_agent_log, sample_errors_log, capsys):
        list_logs()
        captured = capsys.readouterr()
        assert "agent.log" in captured.out
        assert "errors.log" in captured.out

    def test_empty_dir(self, log_dir, capsys):
        list_logs()
        captured = capsys.readouterr()
        assert "no log files yet" in captured.out

    def test_shows_sizes(self, sample_agent_log, capsys):
        list_logs()
        captured = capsys.readouterr()
        # File is small, should show as bytes or KB
        assert "B" in captured.out or "KB" in captured.out
