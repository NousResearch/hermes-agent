"""Regression tests for #131993 (cooldown-msg leg): when a fallback candidate's
credential pool is unusable but carries no wait information (an unfilled borrowed
row — access_token="" — or an empty pool), the skip log must not claim
"every entry in cooldown", because no entry is actually in cooldown. A genuine
exhaustion cooldown keeps the original wording."""

import logging
import time
from types import SimpleNamespace


def _make_pool(provider, entries):
    from agent.credential_pool import CredentialPool

    return CredentialPool(provider, entries)


def _borrowed_unfilled_entry(provider="openrouter"):
    """Shape of the #131993 report: a borrowed row that was never filled — the
    metadata-only form from test_credential_pool.test_credential_pool_never_selects_empty_borrowed_entry."""
    from agent.credential_pool import PooledCredential

    return PooledCredential(
        provider=provider,
        id="borrowed-1",
        label="VAULT_REF",
        auth_type="api_key",
        priority=0,
        source="vault:openrouter/api-key",
        access_token="",
    )


def _agent_with_pool(pool):
    return SimpleNamespace(_credential_pool=pool, _entitlement_rejected_models=None)


def test_skip_message_distinguishes_unfilled_borrowed_row(caplog):
    """Pool unusable (sole borrowed row unfilled) with no entry in cooldown:
    the candidate is still skipped, but the log must say there is no wait
    information rather than claiming every entry is in cooldown."""
    from agent.chat_completion_helpers import (
        _pool_exhaustion_detail,
        _should_skip_fallback_candidate,
    )

    pool = _make_pool("openrouter", [_borrowed_unfilled_entry()])
    agent = _agent_with_pool(pool)

    assert _pool_exhaustion_detail(agent, "openrouter", "some-model") == "no-wait-info"

    with caplog.at_level(logging.WARNING):
        skipped = _should_skip_fallback_candidate(
            agent,
            {"provider": "openrouter", "model": "some-model"},
            ("openrouter", "some-model"),
            "openrouter",
            "some-model",
            set(),
        )
    assert skipped is True
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    pool_msgs = [m for m in warnings if "credential pool is exhausted" in m]
    assert pool_msgs, f"expected a pool-exhausted skip warning, got: {warnings}"
    assert all("every entry in cooldown" not in m for m in pool_msgs)
    assert any("no wait information" in m for m in pool_msgs)


def test_skip_message_keeps_cooldown_wording_for_real_cooldown(caplog):
    """Companion arm: a genuine exhaustion cooldown (next_available_at() in the
    future, beyond the 600s cap) keeps the original "every entry in cooldown"
    wording, which is accurate there."""
    from agent.chat_completion_helpers import (
        _pool_exhaustion_detail,
        _should_skip_fallback_candidate,
    )
    from agent.credential_pool import PooledCredential

    entry = PooledCredential(
        provider="openrouter",
        id="real-1",
        label="OPENROUTER_API_KEY",
        auth_type="api_key",
        priority=0,
        source="env:OPENROUTER_API_KEY",
        access_token="sk-real-token",
        last_status="exhausted",
        last_status_at=time.time(),
        last_error_reset_at=time.time() + 3600,
    )
    pool = _make_pool("openrouter", [entry])
    agent = _agent_with_pool(pool)

    assert _pool_exhaustion_detail(agent, "openrouter", "some-model") == "cooldown"

    with caplog.at_level(logging.WARNING):
        skipped = _should_skip_fallback_candidate(
            agent,
            {"provider": "openrouter", "model": "some-model"},
            ("openrouter", "some-model"),
            "openrouter",
            "some-model",
            set(),
        )
    assert skipped is True
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("every entry in cooldown" in m for m in warnings)
