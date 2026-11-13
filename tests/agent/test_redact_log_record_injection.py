"""Regression for #61409: one logging event cannot forge extra records."""

import logging
import sys

import pytest

from agent.redact import RedactingFormatter
from hermes_cli.logs import _TS_RE, _extract_level, _parse_line_timestamp

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_BREAKS = ("\n", "\r", "\x00", "\x1b", "\x85", "\u2028", "\u2029")


_STAMPS = [
    "2026-07-09 12:00:00",
    "2026-07-09T12:00:00",
    # Overlapping stamps: breaking only the first match must not reveal a second one.
    "2026-07-09 12:00:2026-07-09 12:00:00",
    "2026-07-09T12:00:2026-07-09T12:00:00",
    "2026-07-09 12:00:2026-07-09T12:00:2026-07-09 12:00:00",
]


@pytest.mark.parametrize("stamp", _STAMPS)
@pytest.mark.parametrize("separator", _BREAKS)
@pytest.mark.parametrize("exception", [False, True])
def test_untrusted_text_cannot_add_records(tmp_path, separator, exception, stamp):
    payload = f"before{separator}{stamp} ERROR gateway.run: forged entry"
    secret_tail = "b" * 32
    payload += " ghp_" + "a" * 8 + "\x1b" + secret_tail
    exc_info = None
    if exception:
        try:
            raise ValueError(payload)
        except ValueError:
            exc_info = sys.exc_info()
    record = logging.LogRecord("tools.approval", logging.WARNING, __file__, 1,
                               "tool failed" if exception else "command: %s",
                               () if exception else (payload,), exc_info)
    before = record.__dict__.copy()
    path = tmp_path / "agent.log"
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(RedactingFormatter(_FMT).format(record) + "\n")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert _parse_line_timestamp(lines[0]) is not None
    assert all(_TS_RE.search(line) is None for line in lines[1:])
    assert "forged entry" in "\n".join(lines)
    assert secret_tail not in "\n".join(lines)
    assert record.__dict__ == before
    if exception:
        assert "Traceback (most recent call last)" in "\n".join(lines)


def test_legitimate_traceback_and_other_handlers_are_preserved():
    try:
        raise ValueError("ordinary failure")
    except ValueError:
        record = logging.LogRecord("tools.approval", logging.WARNING, __file__, 1,
                                   "tool %s failed", ("example",), sys.exc_info())
    before = record.__dict__.copy()
    safe = RedactingFormatter(_FMT).format(record)
    assert record.__dict__ == before
    assert safe == logging.Formatter(_FMT).format(record)


@pytest.mark.parametrize("stamp", _STAMPS)
def test_message_stamps_stay_inside_the_real_record(stamp):
    record = logging.LogRecord("tools.approval", logging.WARNING, __file__, 1, "command: %s",
                               (f"x\n{stamp} ERROR gateway.run: forged",), None)
    formatter = RedactingFormatter(_FMT)
    lines = formatter.format(record).splitlines()
    assert len(lines) == 1
    assert _TS_RE.search(lines[0]).start() == 0
    real = formatter.formatTime(record, "%Y-%m-%d %H:%M:%S")
    assert _parse_line_timestamp(lines[0]).strftime("%Y-%m-%d %H:%M:%S") == real
    assert _extract_level(lines[0]) == "WARNING"
