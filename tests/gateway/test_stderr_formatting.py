"""Regression tests for operator-visible gateway stderr formatting."""

from __future__ import annotations

import io
import logging
import re

from gateway import run as gateway_run
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


def test_gateway_stderr_handler_filters_routine_mcp_cleanup_noise() -> None:
    build_handler = getattr(gateway_run, "_build_gateway_stderr_handler", None)
    assert callable(build_handler), (
        "gateway stderr must be built through the shared transport-noise filter"
    )

    stream = io.StringIO()
    handler = build_handler(logging.INFO, stream=stream)
    try:
        records = [
            ("httpx2", logging.INFO, "routine MCP HTTP 200"),
            (
                "mcp.client.streamable_http",
                logging.INFO,
                "Received session ID: gateway-test-session",
            ),
            (
                "mcp.client.streamable_http",
                logging.WARNING,
                "Session termination failed: 404",
            ),
            (
                "mcp.client.streamable_http",
                logging.WARNING,
                "Session termination failed: 500",
            ),
            ("httpx2", logging.WARNING, "important transport warning"),
        ]
        for name, level, message in records:
            handler.handle(
                logging.LogRecord(
                    name=name,
                    level=level,
                    pathname=__file__,
                    lineno=1,
                    msg=message,
                    args=(),
                    exc_info=None,
                )
            )
    finally:
        handler.close()

    rendered = stream.getvalue()
    assert "routine MCP HTTP 200" not in rendered
    assert "gateway-test-session" not in rendered
    assert "Session termination failed: 404" not in rendered
    assert "Session termination failed: 500" in rendered
    assert "important transport warning" in rendered
