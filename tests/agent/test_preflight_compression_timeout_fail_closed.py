"""Regression coverage for stalled preflight compression."""

from types import SimpleNamespace

import pytest

from agent.turn_context import (
    PreflightCompressionTimedOut,
    _fail_closed_after_preflight_timeout,
)


def test_preflight_timeout_blocks_unchanged_provider_payload():
    """Unknown window (no compressor): the conservative #98424 default — never send blind."""
    agent = SimpleNamespace(_last_compression_timed_out=True)

    with pytest.raises(PreflightCompressionTimedOut, match="provider call was not sent"):
        _fail_closed_after_preflight_timeout(agent, 190_035)


def test_structural_noop_keeps_existing_preflight_behavior():
    agent = SimpleNamespace(_last_compression_timed_out=False)

    _fail_closed_after_preflight_timeout(agent, 190_035)


def test_preflight_timeout_sends_a_request_that_fits_the_model_window():
    """Over the compression threshold but under the window (#113646): the turn runs uncompressed, exactly
    like the cooldown-blocked path sends it; above the window it still fails closed."""
    fits = SimpleNamespace(
        _last_compression_timed_out=True, context_compressor=SimpleNamespace(context_length=120_000),
    )
    _fail_closed_after_preflight_timeout(fits, 99_347)

    over = SimpleNamespace(
        _last_compression_timed_out=True, context_compressor=SimpleNamespace(context_length=1_000_000),
    )
    with pytest.raises(PreflightCompressionTimedOut):
        _fail_closed_after_preflight_timeout(over, 1_313_423)
