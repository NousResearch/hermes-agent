"""Regression coverage for stalled preflight compression."""

from types import SimpleNamespace

import pytest

from agent.turn_context import (
    PreflightCompressionTimedOut,
    _fail_closed_after_preflight_timeout,
)


def test_preflight_timeout_blocks_unchanged_provider_payload():
    agent = SimpleNamespace(_last_compression_timed_out=True)

    with pytest.raises(PreflightCompressionTimedOut, match="provider call was not sent"):
        _fail_closed_after_preflight_timeout(agent, 190_035)


def test_structural_noop_keeps_existing_preflight_behavior():
    agent = SimpleNamespace(_last_compression_timed_out=False)

    _fail_closed_after_preflight_timeout(agent, 190_035)
