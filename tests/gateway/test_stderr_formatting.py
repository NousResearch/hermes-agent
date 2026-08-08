"""Regression tests for operator-visible gateway stderr formatting."""

from __future__ import annotations

import logging
import re

from gateway.run import _gateway_stderr_formatter


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

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} "
        r"ERROR gateway\.run: delivery failed",
        rendered,
    )
