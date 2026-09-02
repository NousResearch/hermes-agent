"""Tests for hermes_cli.logs — log viewing and filtering."""

from datetime import datetime, timedelta


from hermes_cli.logs import (
    LOG_FILES,
    _extract_level,
    _extract_logger_name,
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


class TestExtractLevel:
    def test_info(self):
        assert _extract_level("2026-01-01 00:00:00 INFO gateway.run: msg") == "INFO"


# ---------------------------------------------------------------------------
# Logger name extraction (new for component filtering)
# ---------------------------------------------------------------------------

class TestExtractLoggerName:
    def test_standard_line(self):
        line = "2026-04-11 10:23:45 INFO gateway.run: Starting gateway"
        assert _extract_logger_name(line) == "gateway.run"


    def test_no_match(self):
        assert _extract_logger_name("random text") is None


class TestLineMatchesComponent:

    def test_gateway_nested(self):
        # Migrated platform adapters log under plugins.platforms.* (#41112) and
        # must still resolve to the gateway component. Use the real expanded
        # gateway prefixes (COMPONENT_PREFIXES["gateway"]) the CLI passes, not a
        # bare ("gateway",), since the logger name no longer literally starts
        # with "gateway".
        from hermes_logging import COMPONENT_PREFIXES
        line = "2026-04-11 10:23:45 INFO plugins.platforms.telegram.adapter: msg"
        assert _line_matches_component(line, COMPONENT_PREFIXES["gateway"])





    def test_unparseable_line(self):
        assert not _line_matches_component("random text", ("gateway",))


# ---------------------------------------------------------------------------
# Combined filter
# ---------------------------------------------------------------------------

class TestMatchesFilters:

    def test_level_filter(self):
        assert _matches_filters(
            "2026-01-01 00:00:00 WARNING x: msg", min_level="WARNING")
        assert not _matches_filters(
            "2026-01-01 00:00:00 INFO x: msg", min_level="WARNING")


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


# ---------------------------------------------------------------------------
# LOG_FILES registry
# ---------------------------------------------------------------------------

class TestLogFiles:
    def test_known_log_files(self):
        assert "agent" in LOG_FILES
        assert "errors" in LOG_FILES
        assert "gateway" in LOG_FILES
        assert "gui" in LOG_FILES


class TestSameFile:
    """Rotation detection for `hermes logs -f` (_same_file)."""

    def test_same_file_true(self, tmp_path):
        from hermes_cli.logs import _same_file

        p = tmp_path / "agent.log"
        p.write_text("hello\n")
        with open(p, "r") as f:
            assert _same_file(f, p) is True

    def test_rename_recreate_detected(self, tmp_path):
        from hermes_cli.logs import _same_file

        p = tmp_path / "agent.log"
        p.write_text("old\n")
        with open(p, "r") as f:
            # RotatingFileHandler-style rollover: rename + recreate.
            p.rename(tmp_path / "agent.log.1")
            p.write_text("new\n")
            assert _same_file(f, p) is False

    def test_missing_path_detected(self, tmp_path):
        from hermes_cli.logs import _same_file

        p = tmp_path / "agent.log"
        p.write_text("old\n")
        with open(p, "r") as f:
            p.unlink()
            assert _same_file(f, p) is False

    def test_truncate_in_place_detected(self, tmp_path):
        from hermes_cli.logs import _same_file

        p = tmp_path / "agent.log"
        p.write_text("some longer content here\n")
        with open(p, "r") as f:
            f.seek(0, 2)  # follower sits at EOF
            p.write_text("")  # truncated under us
            assert _same_file(f, p) is False


class TestFollowLogRotation:
    def test_follow_survives_rotation(self, tmp_path, monkeypatch):
        """_follow_log keeps emitting lines written after a rename+recreate
        rotation (the pre-fix follower held the old fd and went silent)."""
        import threading
        import time as _time

        import hermes_cli.logs as logs_mod

        p = tmp_path / "agent.log"
        p.write_text("before\n")

        captured = []
        stop = threading.Event()

        real_sleep = _time.sleep

        def fake_print(line, end=""):
            captured.append(line)

        def fast_sleep(_secs):
            real_sleep(0.02)
            if stop.is_set():
                raise KeyboardInterrupt

        monkeypatch.setattr("builtins.print", fake_print)
        monkeypatch.setattr(logs_mod.time, "sleep", fast_sleep)

        t = threading.Thread(
            target=lambda: _swallow_kbi(logs_mod._follow_log, p), daemon=True
        )
        t.start()
        try:
            real_sleep(0.15)
            # Rotate: rename live file away, create a fresh one at the path.
            p.rename(tmp_path / "agent.log.1")
            with open(p, "w") as f:
                f.write("after-rotation\n")
            deadline = _time.time() + 5.0
            while _time.time() < deadline:
                if any("after-rotation" in ln for ln in captured):
                    break
                real_sleep(0.05)
        finally:
            stop.set()
            t.join(timeout=5.0)

        assert any("after-rotation" in ln for ln in captured), (
            f"follower missed post-rotation lines; captured={captured!r}"
        )


def _swallow_kbi(fn, *args):
    try:
        fn(*args)
    except KeyboardInterrupt:
        pass
