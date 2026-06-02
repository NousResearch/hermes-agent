"""Tests for Signal SSE reconnect log-noise handling.

The Signal SSE stream reconnects ~5-6x/hour with a benign, self-healing
"Server disconnected" (httpx pooled-socket churn). Those transient reconnects
should not surface as WARNING spam — only a persistent, backing-off failure
should warn.
"""
import logging

from gateway.platforms.signal import _sse_reconnect_log_level, SSE_RETRY_DELAY_INITIAL


def test_transient_reconnect_logs_at_debug():
    # First reconnect (backoff still at the initial value) is benign and
    # self-heals on the next connect → debug, not warning.
    assert _sse_reconnect_log_level(SSE_RETRY_DELAY_INITIAL) == logging.DEBUG


def test_escalating_reconnect_logs_at_warning():
    # Once the backoff has grown, the daemon is persistently unreachable →
    # warn so a real outage is visible.
    assert _sse_reconnect_log_level(SSE_RETRY_DELAY_INITIAL * 4) == logging.WARNING
