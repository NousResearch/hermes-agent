"""Tests for hermes_cli.logs — log viewing and filtering."""

import builtins
from contextlib import redirect_stdout
from datetime import datetime, timedelta

import pytest

from hermes_cli.logs import (
    LOG_FILES,
    _extract_level,
    _extract_logger_name,
    _follow_log,
    _line_matches_component,
    _matches_filters,
    _parse_line_timestamp,
    _parse_since,
    _read_last_n_lines,
    _read_tail,
)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestParseSince:
    def test_hours(self):
        cutoff = _parse_since("2h")
        assert cutoff is not None
        assert abs((datetime.now() - cutoff).total_seconds() - 7200) < 2

    def test_minutes(self):
        cutoff = _parse_since("30m")
        assert cutoff is not None
        assert abs((datetime.now() - cutoff).total_seconds() - 1800) < 2

    def test_days(self):
        cutoff = _parse_since("1d")
        assert cutoff is not None
        assert abs((datetime.now() - cutoff).total_seconds() - 86400) < 2

    def test_seconds(self):
        cutoff = _parse_since("120s")
        assert cutoff is not None
        assert abs((datetime.now() - cutoff).total_seconds() - 120) < 2

    def test_invalid_returns_none(self):
        assert _parse_since("abc") is None
        assert _parse_since("") is None
        assert _parse_since("10x") is None

    def test_whitespace_tolerance(self):
        cutoff = _parse_since("  5m  ")
        assert cutoff is not None


class TestParseLineTimestamp:
    def test_standard_format(self):
        ts = _parse_line_timestamp("2026-04-11 10:23:45 INFO gateway.run: msg")
        assert ts == datetime(2026, 4, 11, 10, 23, 45)

    def test_no_timestamp(self):
        assert _parse_line_timestamp("no timestamp here") is None


class TestExtractLevel:
    def test_info(self):
        assert _extract_level("2026-01-01 00:00:00 INFO gateway.run: msg") == "INFO"

    def test_warning(self):
        assert _extract_level("2026-01-01 00:00:00 WARNING tools.file: msg") == "WARNING"

    def test_error(self):
        assert _extract_level("2026-01-01 00:00:00 ERROR run_agent: msg") == "ERROR"

    def test_debug(self):
        assert _extract_level("2026-01-01 00:00:00 DEBUG agent.aux: msg") == "DEBUG"

    def test_no_level(self):
        assert _extract_level("random text") is None


# ---------------------------------------------------------------------------
# Logger name extraction (new for component filtering)
# ---------------------------------------------------------------------------

class TestExtractLoggerName:
    def test_standard_line(self):
        line = "2026-04-11 10:23:45 INFO gateway.run: Starting gateway"
        assert _extract_logger_name(line) == "gateway.run"

    def test_nested_logger(self):
        line = "2026-04-11 10:23:45 INFO plugins.platforms.telegram.adapter: connected"
        assert _extract_logger_name(line) == "plugins.platforms.telegram.adapter"

    def test_warning_level(self):
        line = "2026-04-11 10:23:45 WARNING tools.terminal_tool: timeout"
        assert _extract_logger_name(line) == "tools.terminal_tool"

    def test_with_session_tag(self):
        line = "2026-04-11 10:23:45 INFO [abc123] tools.file_tools: reading file"
        assert _extract_logger_name(line) == "tools.file_tools"

    def test_with_session_tag_and_error(self):
        line = "2026-04-11 10:23:45 ERROR [sess_xyz] agent.context_compressor: failed"
        assert _extract_logger_name(line) == "agent.context_compressor"

    def test_top_level_module(self):
        line = "2026-04-11 10:23:45 INFO run_agent: starting conversation"
        assert _extract_logger_name(line) == "run_agent"

    def test_no_match(self):
        assert _extract_logger_name("random text") is None


class TestLineMatchesComponent:
    def test_gateway_component(self):
        line = "2026-04-11 10:23:45 INFO gateway.run: msg"
        assert _line_matches_component(line, ("gateway",))

    def test_gateway_nested(self):
        # Migrated platform adapters log under plugins.platforms.* (#41112) and
        # must still resolve to the gateway component. Use the real expanded
        # gateway prefixes (COMPONENT_PREFIXES["gateway"]) the CLI passes, not a
        # bare ("gateway",), since the logger name no longer literally starts
        # with "gateway".
        from hermes_logging import COMPONENT_PREFIXES
        line = "2026-04-11 10:23:45 INFO plugins.platforms.telegram.adapter: msg"
        assert _line_matches_component(line, COMPONENT_PREFIXES["gateway"])

    def test_gateway_core_nested(self):
        line = "2026-04-11 10:23:45 INFO gateway.run: msg"
        assert _line_matches_component(line, ("gateway",))

    def test_tools_component(self):
        line = "2026-04-11 10:23:45 INFO tools.terminal_tool: msg"
        assert _line_matches_component(line, ("tools",))

    def test_agent_with_multiple_prefixes(self):
        prefixes = ("agent", "run_agent", "model_tools")
        assert _line_matches_component(
            "2026-04-11 10:23:45 INFO agent.context_compressor: msg", prefixes)
        assert _line_matches_component(
            "2026-04-11 10:23:45 INFO run_agent: msg", prefixes)
        assert _line_matches_component(
            "2026-04-11 10:23:45 INFO model_tools: msg", prefixes)

    def test_no_match(self):
        line = "2026-04-11 10:23:45 INFO tools.browser: msg"
        assert not _line_matches_component(line, ("gateway",))

    def test_with_session_tag(self):
        line = "2026-04-11 10:23:45 INFO [abc] gateway.run: msg"
        assert _line_matches_component(line, ("gateway",))

    def test_unparseable_line(self):
        assert not _line_matches_component("random text", ("gateway",))


# ---------------------------------------------------------------------------
# Combined filter
# ---------------------------------------------------------------------------

class TestMatchesFilters:
    def test_no_filters_passes_everything(self):
        assert _matches_filters("any line")

    def test_level_filter(self):
        assert _matches_filters(
            "2026-01-01 00:00:00 WARNING x: msg", min_level="WARNING")
        assert not _matches_filters(
            "2026-01-01 00:00:00 INFO x: msg", min_level="WARNING")

    def test_session_filter(self):
        assert _matches_filters(
            "2026-01-01 00:00:00 INFO [abc123] x: msg", session_filter="abc123")
        assert not _matches_filters(
            "2026-01-01 00:00:00 INFO [xyz789] x: msg", session_filter="abc123")

    def test_component_filter(self):
        assert _matches_filters(
            "2026-01-01 00:00:00 INFO gateway.run: msg",
            component_prefixes=("gateway",))
        assert not _matches_filters(
            "2026-01-01 00:00:00 INFO tools.file: msg",
            component_prefixes=("gateway",))

    def test_combined_filters(self):
        """All filters must pass for a line to match."""
        line = "2026-04-11 10:00:00 WARNING [sess_1] gateway.run: connection lost"
        assert _matches_filters(
            line,
            min_level="WARNING",
            session_filter="sess_1",
            component_prefixes=("gateway",),
        )
        # Fails component filter
        assert not _matches_filters(
            line,
            min_level="WARNING",
            session_filter="sess_1",
            component_prefixes=("tools",),
        )

    def test_since_filter(self):
        # Line with a very old timestamp should be filtered out
        assert not _matches_filters(
            "2020-01-01 00:00:00 INFO x: old msg",
            since=datetime.now() - timedelta(hours=1))
        # Line with a recent timestamp should pass
        recent = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        assert _matches_filters(
            f"{recent} INFO x: recent msg",
            since=datetime.now() - timedelta(hours=1))


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

class TestReadTail:
    def test_read_small_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        lines = [f"2026-01-01 00:00:0{i} INFO x: line {i}\n" for i in range(10)]
        log_file.write_text("".join(lines))

        result = _read_last_n_lines(log_file, 5)
        assert len(result) == 5
        assert "line 9" in result[-1]

    def test_read_with_component_filter(self, tmp_path):
        log_file = tmp_path / "test.log"
        lines = [
            "2026-01-01 00:00:00 INFO gateway.run: gw msg\n",
            "2026-01-01 00:00:01 INFO tools.file: tool msg\n",
            "2026-01-01 00:00:02 INFO gateway.session: session msg\n",
            "2026-01-01 00:00:03 INFO agent.compressor: agent msg\n",
        ]
        log_file.write_text("".join(lines))

        result = _read_tail(
            log_file, 50,
            has_filters=True,
            component_prefixes=("gateway",),
        )
        assert len(result) == 2
        assert "gw msg" in result[0]
        assert "session msg" in result[1]

    def test_empty_file(self, tmp_path):
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        result = _read_last_n_lines(log_file, 10)
        assert result == []


class _StopFollow(Exception):
    pass


_STOP_FOLLOW = object()


def _install_follow_steps(monkeypatch, *steps):
    remaining = iter(steps)

    def fake_sleep(_seconds):
        try:
            step = next(remaining)
        except StopIteration:
            pytest.fail("follower performed an unexpected extra poll")
        if step is _STOP_FOLLOW:
            raise _StopFollow
        step()

    monkeypatch.setattr("hermes_cli.logs.time.sleep", fake_sleep)


class TestFollowLog:
    def test_emits_append_once_with_closed_streams(
        self, tmp_path, monkeypatch, capsys,
    ):
        log_file = tmp_path / "agent.log"
        log_file.write_text("historical\n", encoding="utf-8")
        opened_streams = []
        real_open = builtins.open

        def tracked_open(*args, **kwargs):
            stream = real_open(*args, **kwargs)
            opened_streams.append(stream)
            return stream

        def append_line():
            assert opened_streams
            assert all(stream.closed for stream in opened_streams)
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write("new\n")

        def check_streams_closed():
            assert all(stream.closed for stream in opened_streams)

        monkeypatch.setattr("hermes_cli.logs.open", tracked_open, raising=False)
        _install_follow_steps(
            monkeypatch,
            append_line,
            check_streams_closed,
            _STOP_FOLLOW,
        )

        with pytest.raises(_StopFollow):
            _follow_log(log_file)

        assert capsys.readouterr().out == "new\n"

    def test_drains_old_generation_before_replacement(
        self, tmp_path, monkeypatch, capsys,
    ):
        log_file = tmp_path / "agent.log"
        rotated_file = tmp_path / "agent.log.1"
        log_file.write_text("historical\n", encoding="utf-8")

        def rotate_with_unread_line():
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write("unread old generation\n")
            log_file.rename(rotated_file)
            log_file.write_text("first new generation\n", encoding="utf-8")

        _install_follow_steps(monkeypatch, rotate_with_unread_line, _STOP_FOLLOW)

        with pytest.raises(_StopFollow):
            _follow_log(log_file)

        assert capsys.readouterr().out == (
            "unread old generation\n"
            "first new generation\n"
        )

    def test_recovers_after_complete_missing_path_poll(
        self, tmp_path, monkeypatch, capsys,
    ):
        log_file = tmp_path / "agent.log"
        rotated_file = tmp_path / "agent.log.1"
        log_file.write_text("historical\n", encoding="utf-8")

        def remove_active_path():
            log_file.rename(rotated_file)

        def create_replacement():
            log_file.write_text("after gap\n", encoding="utf-8")

        _install_follow_steps(
            monkeypatch,
            remove_active_path,
            create_replacement,
            _STOP_FOLLOW,
        )

        with pytest.raises(_StopFollow):
            _follow_log(log_file)

        assert capsys.readouterr().out == "after gap\n"

    def test_reads_from_start_after_initial_missing_path(
        self, tmp_path, monkeypatch, capsys,
    ):
        log_file = tmp_path / "agent.log"

        def create_log():
            log_file.write_text("created during gap\n", encoding="utf-8")

        _install_follow_steps(monkeypatch, create_log, _STOP_FOLLOW)

        with pytest.raises(_StopFollow):
            _follow_log(log_file)

        assert capsys.readouterr().out == "created during gap\n"

    def test_normalizes_crlf_before_output(self, tmp_path, monkeypatch, capsys):
        log_file = tmp_path / "agent.log"
        log_file.write_bytes(b"historical\r\n")

        def append_crlf_line():
            with log_file.open("ab") as stream:
                stream.write(b"windows line\r\n")

        _install_follow_steps(monkeypatch, append_crlf_line, _STOP_FOLLOW)

        with pytest.raises(_StopFollow):
            _follow_log(log_file)

        assert capsys.readouterr().out == "windows line\n"

    def test_broken_pipe_propagates(self, tmp_path, monkeypatch):
        log_file = tmp_path / "agent.log"
        log_file.write_text("historical\n", encoding="utf-8")

        def append_line():
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write("new\n")

        def unexpected_poll():
            pytest.fail("BrokenPipeError was swallowed")

        class BrokenStdout:
            def write(self, _text):
                raise BrokenPipeError("consumer closed")

            def flush(self):
                pass

        _install_follow_steps(monkeypatch, append_line, unexpected_poll)

        with redirect_stdout(BrokenStdout()):
            with pytest.raises(BrokenPipeError, match="consumer closed"):
                _follow_log(log_file)

# ---------------------------------------------------------------------------
# LOG_FILES registry
# ---------------------------------------------------------------------------

class TestLogFiles:
    def test_known_log_files(self):
        assert "agent" in LOG_FILES
        assert "errors" in LOG_FILES
        assert "gateway" in LOG_FILES
        assert "gui" in LOG_FILES
