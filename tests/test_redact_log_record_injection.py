"""RedactingFormatter must neutralize log-record injection via control chars."""

import logging
import sys

from agent.redact import RedactingFormatter

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
