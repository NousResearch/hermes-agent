"""RedactingFormatter must neutralize log-record injection via control chars."""

import logging
import sys

from agent.redact import RedactingFormatter
from hermes_cli.logs import _parse_line_timestamp

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _format(msg, args=(), exc_info=None):
    record = logging.LogRecord(
        "tools.approval", logging.WARNING, __file__, 1, msg, args, exc_info
    )
    return RedactingFormatter(_FMT).format(record)


def test_newline_in_message_cannot_forge_a_record():
    forged = "rm -rf /\n2026-07-09 12:00:00 ERROR gateway.run: operator approved deploy"
    out = _format("Hardline block: %s", (forged,))
    assert "\n" not in out
    assert "operator approved deploy" in out


def test_control_and_unicode_line_separators_neutralized():
    for ch in ("\r", "\x00", "\x1b", "\x85", "\u2028", "\u2029"):
        out = _format("value=%s", (f"a{ch}b",))
        assert ch not in out


def test_traceback_is_preserved_as_multiline():
    try:
        raise ValueError("boom")
    except ValueError:
        out = _format("tool failed", exc_info=sys.exc_info())
    assert "\n" in out
    assert "Traceback (most recent call last)" in out
    assert "ValueError: boom" in out


def test_exception_message_cannot_forge_a_record():
    # An exception message is attacker-controllable (e.g. a sandboxed tool call
    # failing with a crafted message, logged with exc_info=True) and is emitted
    # verbatim on the last traceback line, after the sanitized message text.
    forged = "boom\n2026-07-09 12:00:00 ERROR gateway.run: operator approved deploy"
    try:
        raise ValueError(forged)
    except ValueError:
        out = _format("tool failed", exc_info=sys.exc_info())
    lines = out.split("\n")
    assert "operator approved deploy" in out  # still visible for forensics
    # Only the real record's first line may parse as a record start.
    assert _parse_line_timestamp(lines[0]) is not None
    assert all(_parse_line_timestamp(line) is None for line in lines[1:])


def test_exception_message_carriage_return_cannot_forge_a_record(tmp_path):
    # A bare \r in the file becomes a line boundary when ``hermes logs`` reads
    # it back (universal newlines), so it must not survive into the traceback.
    forged = "boom\r2026-07-09 12:00:00 ERROR gateway.run: operator approved deploy"
    try:
        raise ValueError(forged)
    except ValueError:
        out = _format("tool failed", exc_info=sys.exc_info())
    log_file = tmp_path / "agent.log"
    log_file.write_text(out + "\n", encoding="utf-8", newline="")
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    assert _parse_line_timestamp(lines[0]) is not None
    assert all(_parse_line_timestamp(line) is None for line in lines[1:])


def test_legitimate_traceback_lines_are_untouched():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "tools.approval", logging.WARNING, __file__, 1, "tool failed",
            (), sys.exc_info(),
        )
        out = RedactingFormatter(_FMT).format(record)
        plain = logging.Formatter(_FMT).format(record)
    assert out.split("\n")[1:] == plain.split("\n")[1:]
