"""Tests for hermes_cli.logs — log viewing and filtering."""

import logging
import logging.handlers
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

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

    def test_iso_t_separated_update_and_handoff_stamps(self):
        # posix.sh: date +%Y-%m-%dT%H:%M:%S%z; windows.ps1: yyyy-MM-ddTHH:mm:ssK
        assert _parse_line_timestamp("2026-09-29T21:36:18+08:00 update| step done") == datetime(
            2026, 9, 29, 21, 36, 18
        )
        assert _parse_line_timestamp("2026-09-29T21:36:18Z step stalled") == datetime(
            2026, 9, 29, 21, 36, 18
        )

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

    def test_unstamped_lines_share_the_verdict_of_the_record_above(self, tmp_path):
        old = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S,000")
        new = datetime.now().strftime("%Y-%m-%d %H:%M:%S,000")
        frames = ["Traceback (most recent call last):\n", '  File "x.py", line 1, in f\n']
        log_file = tmp_path / "errors.log"
        log_file.write_text("".join([
            "orphan tail of a record that started before the window\n",
            f"{old} ERROR gateway.run: old failure\n", *frames, "ValueError: old\n",
            f"{new} INFO tools.x: multi-line info\n", "  info continuation\n",
            f"{new} ERROR gateway.run: new failure\n", *frames, "ValueError: new\n",
        ]))
        since = datetime.now() - timedelta(hours=1)
        from hermes_logging import COMPONENT_PREFIXES

        def read(**filters):
            return "".join(_read_tail(log_file, 50, has_filters=True, **filters))

        assert read(since=since) == "".join([
            f"{new} INFO tools.x: multi-line info\n", "  info continuation\n",
            f"{new} ERROR gateway.run: new failure\n", *frames, "ValueError: new\n",
        ])
        new_failure = "".join([f"{new} ERROR gateway.run: new failure\n", *frames, "ValueError: new\n"])
        assert read(since=since, min_level="WARNING") == new_failure
        assert read(since=since, component_prefixes=COMPONENT_PREFIXES["gateway"]) == new_failure
        # No time/level filter: the orphan lines before the first stamp stay visible.
        assert read(session_filter="orphan") == "orphan tail of a record that started before the window\n"

# ---------------------------------------------------------------------------
# LOG_FILES registry
# ---------------------------------------------------------------------------

def _python_log_line(logger_name: str) -> str:
    import logging

    from agent.redact import RedactingFormatter
    from hermes_logging import _LOG_FORMAT

    record = logging.LogRecord(logger_name, logging.WARNING, __file__, 1, "sample", None, None)
    record.session_tag = ""
    return RedactingFormatter(_LOG_FORMAT).format(record)


def _mcp_output_line() -> str:
    import io

    from tools.mcp_tool_config import _StderrTee

    log = io.StringIO()
    tee = _StderrTee(log)
    tee.sink.write(b"server says hello\n")
    tee.close()
    return log.getvalue()


def _update_log_line() -> str:
    """The update.log run banner hermes_cli.main_dashboard writes on every update."""
    import datetime as dt

    return f"\n=== hermes update started {dt.datetime.now().isoformat(timespec='seconds')} ===\n".lstrip()


def _handoff_log_line() -> str:
    """A desktop-update-handoff.log line from scripts/desktop-update/posix.sh's log()."""
    import subprocess

    line = subprocess.run(
        ["bash", "-c", 'log() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $1"; }; log "update| → Checking if desktop app needs rebuilding..."'],
        capture_output=True, text=True, check=True,
    ).stdout
    return line.rstrip("\n")


def _log_file_samples() -> dict:
    """One line per LOG_FILES entry, produced by that file's real writer where Python can run it."""
    return {
        "agent": _python_log_line("run_agent"),
        "errors": _python_log_line("run_agent"),
        "gateway": _python_log_line("gateway.run"),
        "gui": _python_log_line("hermes_cli.web_server"),
        # Written by TypeScript; apps/desktop/electron/desktop-log-line.test.ts pins the same shape.
        "desktop": "2026-09-28 13:18:46,062 [hermes] [boot] ready",
        "mcp": _mcp_output_line(),
        # update.log mirrors raw update output; handoff lines carry the shim's ISO-8601 stamp.
        "update": _update_log_line(),
        "handoff": _handoff_log_line(),
    }


def test_every_log_file_writes_a_stamp_hermes_logs_since_can_read():
    samples = _log_file_samples()
    assert set(samples) == set(LOG_FILES), "add a real sample line for each new LOG_FILES entry"
    for name, line in samples.items():
        assert _parse_line_timestamp(line) is not None, (name, line)
    # gateway.error.log (launchd stderr, not in LOG_FILES) uses the shared stamper.
    from hermes_cli.stderr_timestamp import stamp_line
    assert _parse_line_timestamp(stamp_line("raw gateway stderr")) is not None


# ---------------------------------------------------------------------------
# hermes logs -f
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_log():
    """``agent.log`` under the test HERMES_HOME, written by a real rotating handler."""
    from hermes_constants import get_hermes_home
    path = get_hermes_home() / "logs" / "agent.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("test_logs_follow")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("seed")
    yield path, handler, logger
    logger.removeHandler(handler)
    handler.close()


def _follow(monkeypatch, capsys, writer_steps):
    """Run ``hermes logs agent -f``: each idle poll performs the next writer step, and once they
    (plus a couple of spare polls) are used up the follower is stopped the way Ctrl+C stops it."""
    import hermes_cli.logs as logs_mod

    pending = list(writer_steps) + [lambda: None, lambda: None]

    def idle_poll(_seconds):
        if not pending:
            raise KeyboardInterrupt
        pending.pop(0)()

    monkeypatch.setattr(logs_mod, "time", SimpleNamespace(sleep=idle_poll))
    logs_mod.tail_log("agent", num_lines=1, follow=True)
    return capsys.readouterr().out


def test_follow_keeps_printing_after_the_log_rotates(agent_log, monkeypatch, capsys):
    path, handler, log = agent_log
    out = _follow(monkeypatch, capsys, [
        lambda: log.info("MARK-1 before rollover"),
        lambda: (handler.doRollover(), log.info("MARK-2 after rollover")),
    ])
    assert path.with_name("agent.log.1").exists()  # the rollover really renamed the file
    assert "MARK-1" in out and "MARK-2" in out
    assert out.index("MARK-1") < out.index("MARK-2")


def test_follow_resumes_after_the_log_is_truncated_in_place(agent_log, monkeypatch, capsys):
    path, _handler, log = agent_log

    def truncate_then_log():
        path.write_bytes(b"")  # copytruncate-style rotation; the handler appends at the new end
        log.info("MARK-2")

    out = _follow(monkeypatch, capsys, [lambda: log.info("MARK-1 before truncation"), truncate_then_log])
    assert "MARK-1" in out and "MARK-2" in out


@pytest.mark.platforms("windows")
def test_follow_does_not_block_the_writers_rollover(monkeypatch, capsys):
    """An attached ``hermes logs -f`` must neither stop Hermes' own writer from rotating the log
    nor lose the entries written after the rotation.

    Windows-only: only a Windows handle opened without FILE_SHARE_DELETE refuses the rename, and
    only there is the writer concurrent-log-handler, which swallows the failed rename.
    """
    import hermes_logging
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "logs" / "agent.log"
    handler = hermes_logging._new_file_handler(
        path, level=logging.INFO, max_bytes=1_000_000, backup_count=1,
        formatter=logging.Formatter("%(message)s"),
    )

    def log(message):
        handler.handle(logging.LogRecord("test", logging.INFO, "", 0, message, (), None))

    log("seed")
    try:
        out = _follow(monkeypatch, capsys, [
            lambda: log("MARK-1 before the rollover"),
            lambda: (handler.doRollover(), log("MARK-2 after the rollover")),
        ])
    finally:
        handler.close()

    backup = path.with_name("agent.log.1")
    assert backup.exists(), "the rollover's rename was refused while the follower held agent.log"
    assert "MARK-1" in backup.read_text(encoding="utf-8-sig")
    assert "MARK-1" in out and "MARK-2" in out
