"""Tests for hermes_cli.logs — log viewing and filtering."""

from datetime import datetime, timedelta


from hermes_cli.logs import (
    LOG_FILES,
    _FollowFilter,
    _extract_level,
    _extract_logger_name,
    _filter_records,
    _iter_records,
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

    def test_unfiltered_tail_is_a_plain_line_tail(self, tmp_path):
        """No regression: with no filters, `_read_tail` is still a raw last-N-lines
        tail (it may begin part-way through a record, like `tail -n`)."""
        log_file = tmp_path / "t.log"
        log_file.write_text(
            "2026-01-01 00:00:00 ERROR x: boom\n"
            "  frame one\n"
            "  frame two\n"
            "2026-01-01 00:00:01 INFO x: after\n"
        )
        assert _read_tail(log_file, 2, has_filters=False) == [
            "  frame two\n",
            "2026-01-01 00:00:01 INFO x: after\n",
        ]


# ---------------------------------------------------------------------------
# Record grouping + record-aware filtering (BUG A: sparse window, BUG B: multiline)
# ---------------------------------------------------------------------------

def _write(tmp_path, text):
    p = tmp_path / "log.log"
    p.write_text(text, encoding="utf-8")
    return p


class TestIterRecords:
    def test_groups_header_with_continuations(self):
        lines = [
            "2026-01-01 00:00:00 ERROR x: boom\n",
            "Traceback (most recent call last):\n",
            '  File "x.py", line 1\n',
            "2026-01-01 00:00:01 INFO x: ok\n",
        ]
        records = list(_iter_records(lines))
        assert [h for h, _ in records] == [lines[0], lines[3]]
        assert records[0][1] == lines[0:3]
        assert records[1][1] == [lines[3]]

    def test_leading_continuation_lines_are_an_orphan_record(self):
        lines = [
            "  orphan frame a\n",
            "  orphan frame b\n",
            "2026-01-01 00:00:00 INFO x: real\n",
        ]
        records = list(_iter_records(lines))
        assert records[0] == (None, ["  orphan frame a\n", "  orphan frame b\n"])
        assert records[1] == (lines[2], [lines[2]])


class TestFilterRecordsBugA:
    def test_sparse_match_far_behind_2000_nonmatches(self, tmp_path):
        """Three ERROR records, then 2001 INFO lines. `-n 2 --level ERROR` must
        still return the two latest ERROR records — the old fixed ~2000-line
        window would have returned nothing."""
        body = "".join(
            f"2026-01-01 00:00:0{i} ERROR x: err {i}\n" for i in range(3)
        ) + "".join(
            f"2026-01-01 01:{m // 60:02d}:{m % 60:02d} INFO x: noise {m}\n"
            for m in range(2001)
        )
        path = _write(tmp_path, body)

        out = _read_tail(path, 2, has_filters=True, min_level="ERROR")
        assert out == [
            "2026-01-01 00:00:01 ERROR x: err 1\n",
            "2026-01-01 00:00:02 ERROR x: err 2\n",
        ]


class TestFilterRecordsBugB:
    def test_matching_multiline_traceback_retained(self, tmp_path):
        path = _write(
            tmp_path,
            "2026-01-01 00:00:00 INFO x: before\n"
            "2026-01-01 00:00:01 ERROR x: kaboom\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 42, in f\n'
            "    raise ValueError('bad')\n"
            "ValueError: bad\n",
        )
        out = _read_tail(path, 5, has_filters=True, min_level="ERROR")
        assert out == [
            "2026-01-01 00:00:01 ERROR x: kaboom\n",
            "Traceback (most recent call last):\n",
            '  File "x.py", line 42, in f\n',
            "    raise ValueError('bad')\n",
            "ValueError: bad\n",
        ]

    def test_rejected_multiline_record_is_fully_suppressed(self, tmp_path):
        """A non-matching record's continuation lines must NOT leak (the old
        per-line filter let level-less continuation lines through)."""
        path = _write(
            tmp_path,
            "2026-01-01 00:00:00 INFO x: chatty\n"
            "  extra context line 1\n"
            "  extra context line 2\n"
            "2026-01-01 00:00:01 ERROR x: real\n"
            "  real frame\n",
        )
        out = _read_tail(path, 5, has_filters=True, min_level="ERROR")
        assert out == [
            "2026-01-01 00:00:01 ERROR x: real\n",
            "  real frame\n",
        ]
        assert not any("extra context" in ln for ln in out)

    def test_adjacent_records_different_components(self, tmp_path):
        from hermes_logging import COMPONENT_PREFIXES

        path = _write(
            tmp_path,
            "2026-01-01 00:00:00 INFO gateway.run: up\n"
            "  gateway detail\n"
            "2026-01-01 00:00:01 INFO tools.shell: ran\n"
            "  tools detail\n",
        )
        out = _read_tail(
            path, 10, has_filters=True,
            component_prefixes=COMPONENT_PREFIXES["gateway"],
        )
        assert out == [
            "2026-01-01 00:00:00 INFO gateway.run: up\n",
            "  gateway detail\n",
        ]

    def test_window_begins_on_a_continuation_line(self, tmp_path):
        """The file's first lines are continuation lines with no header (a record
        split by rotation). They are unattributable and must be dropped, never
        emitted as if they matched."""
        path = _write(
            tmp_path,
            "  dangling frame from a rotated-away header\n"
            "  another dangling frame\n"
            "2026-01-01 00:00:00 ERROR x: fresh\n"
            "  fresh frame\n",
        )
        out = _read_tail(path, 5, has_filters=True, min_level="ERROR")
        assert out == [
            "2026-01-01 00:00:00 ERROR x: fresh\n",
            "  fresh frame\n",
        ]

    def test_tail_boundary_counts_records_not_lines(self, tmp_path):
        five = "".join(
            f"2026-01-01 00:00:0{i} ERROR x: e{i}\n" for i in range(5)
        )
        path = _write(tmp_path, five)
        # -n 2 → exactly the last two records, not 1, not 3.
        assert _read_tail(path, 2, has_filters=True, min_level="ERROR") == [
            "2026-01-01 00:00:03 ERROR x: e3\n",
            "2026-01-01 00:00:04 ERROR x: e4\n",
        ]
        # -n counts records: one record with a 3-line traceback, -n 1 → 3 lines.
        path2 = _write(
            tmp_path,
            "2026-01-01 00:00:00 ERROR x: a\n"
            "2026-01-01 00:00:01 ERROR x: b\n"
            "frame 1\n"
            "frame 2\n",
        )
        assert _read_tail(path2, 1, has_filters=True, min_level="ERROR") == [
            "2026-01-01 00:00:01 ERROR x: b\n",
            "frame 1\n",
            "frame 2\n",
        ]

    def test_filter_records_is_memory_bounded(self, tmp_path):
        """Streaming: only `want` records are held regardless of file size."""
        gen = (f"2026-01-01 00:00:00 INFO x: line {i}\n" for i in range(100_000))
        kept = _filter_records(gen, want=3, min_level="INFO")
        assert kept == [
            "2026-01-01 00:00:00 INFO x: line 99997\n",
            "2026-01-01 00:00:00 INFO x: line 99998\n",
            "2026-01-01 00:00:00 INFO x: line 99999\n",
        ]


class TestFollowFilter:
    def test_matching_header_prints_its_continuation_rejected_hides_its_own(self):
        ff = _FollowFilter(min_level="ERROR")
        seq = [
            ("2026-01-01 00:00:00 INFO x: hi\n", False),      # rejected header
            ("  info continuation\n", False),                  # its continuation hidden
            ("2026-01-01 00:00:01 ERROR x: boom\n", True),     # matching header
            ("Traceback (most recent call last):\n", True),    # its continuation shown
            ('  File "x.py", line 1\n', True),
            ("2026-01-01 00:00:02 DEBUG x: noise\n", False),   # rejected header
            ("  debug continuation\n", False),                 # hidden again
        ]
        for line, expected in seq:
            assert ff.should_emit(line) is expected, line

    def test_continuation_before_any_header_is_not_emitted(self):
        ff = _FollowFilter(min_level="ERROR")
        assert ff.should_emit("  dangling frame, no header yet\n") is False
        # a real matching header then resumes normal behaviour
        assert ff.should_emit("2026-01-01 00:00:00 ERROR x: y\n") is True
        assert ff.should_emit("  now-attributed frame\n") is True


# ---------------------------------------------------------------------------
# LOG_FILES registry
# ---------------------------------------------------------------------------

class TestLogFiles:
    def test_known_log_files(self):
        assert "agent" in LOG_FILES
        assert "errors" in LOG_FILES
        assert "gateway" in LOG_FILES
        assert "gui" in LOG_FILES
