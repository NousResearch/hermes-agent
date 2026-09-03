"""Regression coverage for main-turn model-not-found failover."""

from __future__ import annotations

from pathlib import Path


def test_main_turn_model_not_found_is_eager_fallback_reason():
    """A stale Hermes model alias should switch to configured fallbacks immediately.

    The classifier already marks model-not-found as non-retryable and
    fallback-worthy. This pins the main conversation loop so it does not burn
    generic retries before trying ``fallback_providers``.
    """

    source = (
        Path(__file__).resolve().parent.parent.parent
        / "agent"
        / "conversation_loop.py"
    ).read_text(encoding="utf-8")

    assert "FailoverReason.model_not_found" in source
    assert "_is_model_unavailable" in source
    assert "_is_model_unavailable\n                    or (is_rate_limited" in source
    assert "Model unavailable — switching to fallback provider" in source

