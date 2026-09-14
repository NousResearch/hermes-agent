"""Regression for #61411: one logging event cannot forge extra records."""

import logging
import sys

import pytest

from agent.redact import RedactingFormatter
from hermes_cli.logs import _parse_line_timestamp

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_BREAKS = ("\n", "\r", "\x00", "\x1b", "\x85", "\u2028", "\u2029")


@pytest.mark.parametrize("separator", _BREAKS)
@pytest.mark.parametrize("exception", [False, True])
def test_untrusted_text_cannot_add_records(tmp_path, separator, exception):
    payload = f"before{separator}2026-07-09 12:00:00 ERROR gateway.run: forged entry"
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
    assert all(_parse_line_timestamp(line) is None for line in lines[1:])
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
