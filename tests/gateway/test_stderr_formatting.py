"""Regression tests for operator-visible gateway stderr formatting."""

from __future__ import annotations

import logging
import re

from gateway.run import _GatewayDefaultStderrFilter, _gateway_stderr_formatter


def test_gateway_stderr_formatter_includes_timestamp() -> None:
    record = logging.LogRecord(
        name="gateway.run",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="delivery failed",
        args=(),
        exc_info=None,
    )

    rendered = _gateway_stderr_formatter().format(record)

    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", rendered), rendered
    assert "delivery failed" in rendered


def test_default_gateway_stderr_allows_only_warnings_and_operator_notices() -> None:
    filt = _GatewayDefaultStderrFilter()

    def record(level: int, *, notice: bool = False) -> logging.LogRecord:
        item = logging.LogRecord(
            name="gateway.run", level=level, pathname=__file__, lineno=1,
            msg="message", args=(), exc_info=None,
        )
        if notice:
            item.gateway_console_notice = True
        return item

    assert filt.filter(record(logging.INFO)) is False
    assert filt.filter(record(logging.INFO, notice=True)) is True
    assert filt.filter(record(logging.WARNING)) is True
