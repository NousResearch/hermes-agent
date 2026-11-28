"""Anthropic billing route: OAuth subscription seats vs metered API keys.

Claude Max / Claude Code / Pro seats authenticate with an OAuth token and
carry no per-token invoice — usage is metered against rolling quota windows
instead. Console API keys (``sk-ant-api…``) bill every token. Before this
fix ``resolve_billing_route`` returned ``official_docs_snapshot`` for every
``provider == "anthropic"`` caller, so subscription sessions were priced as
if they were metered.

The regression guard that matters most here is the metered direction: a
user on a real API key must keep seeing real costs. Under-reporting actual
spend is worse than the ``unknown`` status this replaces, so detection is
positive-only and fails closed.
"""

from decimal import Decimal
from unittest.mock import patch

import pytest

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    get_pricing_entry,
    has_known_pricing,
    resolve_billing_route,
)

OAUTH_TOKEN = "sk-ant-oat01-" + "x" * 40
API_KEY = "sk-ant-api03-" + "x" * 40

USAGE = CanonicalUsage(
    input_tokens=1000,
    output_tokens=500,
    cache_read_tokens=20000,
    cache_write_tokens=2000,
)


def _pool(*tokens_with_priority):
    """Build a credential-pool payload shaped like auth.json."""
    return [
        {"label": f"seat-{i}", "auth_type": "oauth", "priority": p, "access_token": t}
        for i, (t, p) in enumerate(tokens_with_priority)
    ]


def _pool_returning(payload):
    """``read_credential_pool`` is imported inside the detector, so patch the
    binding it reads (``hermes_cli.auth``), not a copy."""
    return patch("hermes_cli.auth.read_credential_pool", return_value=payload)


def _no_pool():
    return _pool_returning([])


# ---------------------------------------------------------------------------
# Explicit credential wins
# ---------------------------------------------------------------------------


def test_oauth_token_routes_to_subscription_included():
    with _no_pool():
        route = resolve_billing_route(
            "claude-opus-5", provider="anthropic", api_key=OAUTH_TOKEN
        )
    assert route.billing_mode == "subscription_included"


def test_console_api_key_stays_metered():
    """The guard that matters: a metered user must keep seeing real costs."""
    with _no_pool():
        route = resolve_billing_route(
            "claude-opus-5", provider="anthropic", api_key=API_KEY
        )
    assert route.billing_mode == "official_docs_snapshot"


def test_api_key_wins_over_oauth_pool_entry():
    """An explicit key is the credential the caller used; the pool must not
    override it and silently zero out a metered user's spend."""
    with _pool_returning(_pool((OAUTH_TOKEN, 0))):
        route = resolve_billing_route(
            "claude-opus-5", provider="anthropic", api_key=API_KEY
        )
    assert route.billing_mode == "official_docs_snapshot"


# ---------------------------------------------------------------------------
# Credential-pool fallback (callers that price after the fact and have no
# credential to thread: auxiliary accounting, insights)
# ---------------------------------------------------------------------------


def test_pool_oauth_token_detected_without_explicit_key():
    with _pool_returning(_pool((OAUTH_TOKEN, 0))):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "subscription_included"


def test_all_oauth_pool_is_subscription_regardless_of_order():
    with _pool_returning(_pool((OAUTH_TOKEN, 1), (OAUTH_TOKEN, 0))):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "subscription_included"


@pytest.mark.parametrize(
    "entries",
    [
        _pool((OAUTH_TOKEN, 0), (API_KEY, 1)),
        _pool((API_KEY, 0), (OAUTH_TOKEN, 1)),
    ],
    ids=["oauth-first", "api-key-first"],
)
def test_mixed_pool_stays_metered(entries):
    """A pool holding both an OAuth seat and a Console key cannot tell which
    one served a given request; priority order is not evidence either (the
    runtime rotates on cooldown). Fail closed: keep the metered path rather
    than zero out spend that may be real."""
    with _pool_returning(entries):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "official_docs_snapshot"


def test_pool_skips_entries_without_token():
    """Claude Code seats can appear with an empty or null access_token; skip
    them rather than reading the blank as 'not OAuth'."""
    entries = [
        {"label": "empty", "auth_type": "oauth", "priority": 0, "access_token": ""},
        {"label": "null", "auth_type": "oauth", "priority": 1, "access_token": None},
        {"label": "real", "auth_type": "oauth", "priority": 2, "access_token": OAUTH_TOKEN},
    ]
    with _pool_returning(entries):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "subscription_included"


def test_empty_pool_stays_metered():
    with _no_pool():
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "official_docs_snapshot"


@pytest.mark.parametrize(
    "payload",
    [
        {"anthropic": _pool((OAUTH_TOKEN, 0))},  # whole-store shape, not the provider slice
        None,
        "garbage",
        [42, "junk", None, {"access_token": 7}],
    ],
    ids=["dict-shaped", "none", "string", "garbage-entries"],
)
def test_unexpected_pool_payload_stays_metered(payload):
    """``read_credential_pool("anthropic")`` returns the provider's list of
    entry dicts. Anything else is not positive evidence of a subscription
    seat, so the metered path stays."""
    with _pool_returning(payload):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "official_docs_snapshot"


def test_pool_read_failure_fails_closed():
    """An unreadable auth.json is re-raised by the store loader on purpose; a
    cost estimate must neither take the turn down nor zero out cost."""
    with patch("hermes_cli.auth.read_credential_pool", side_effect=RuntimeError("boom")):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "official_docs_snapshot"


# ---------------------------------------------------------------------------
# End-to-end cost reporting
# ---------------------------------------------------------------------------


def test_subscription_cost_is_zero_and_labelled_included():
    with _no_pool():
        result = estimate_usage_cost(
            "claude-opus-5", USAGE, provider="anthropic", api_key=OAUTH_TOKEN
        )
    assert result.status == "included"
    assert result.amount_usd == Decimal("0")
    assert any("subscription" in note for note in result.notes)


def test_metered_opus_5_is_estimated_at_the_opus_tier():
    """Opus 5 bills at the same tier as Opus 4.5-4.8 (platform.claude.com
    pricing page), so a metered session must produce a real estimate equal
    to the Opus 4.8 estimate for the same usage — not ``unknown``/$0."""
    with _no_pool():
        result = estimate_usage_cost("claude-opus-5", USAGE, provider="anthropic", api_key=API_KEY)
        reference = estimate_usage_cost("claude-opus-4-8", USAGE, provider="anthropic", api_key=API_KEY)
    assert result.status == "estimated"
    assert result.amount_usd is not None and result.amount_usd > 0
    assert result.amount_usd == reference.amount_usd


def test_subscription_pricing_entry_is_all_zero():
    with _no_pool():
        entry = get_pricing_entry(
            "claude-opus-5", provider="anthropic", api_key=OAUTH_TOKEN
        )
    assert entry is not None
    assert entry.input_cost_per_million == Decimal("0")
    assert entry.output_cost_per_million == Decimal("0")


def test_has_known_pricing_true_for_both_modes():
    with _no_pool():
        assert has_known_pricing("claude-opus-5", provider="anthropic", api_key=OAUTH_TOKEN)
        assert has_known_pricing("claude-opus-5", provider="anthropic", api_key=API_KEY)


# ---------------------------------------------------------------------------
# Pricing table: Opus 5 was missing entirely
# ---------------------------------------------------------------------------


def test_opus_5_has_metered_pricing_entry_with_cache_rates():
    """Opus 5 shipped 2026-07-24 with no entry, so metered users saw
    cost_status='unknown' and a $0 total for every Opus 5 session. The row
    must carry cache rates too (Hermes sessions are cache-heavy) and sit on
    the same tier as Opus 4.8."""
    with _no_pool():
        entry = get_pricing_entry("claude-opus-5", provider="anthropic", api_key=API_KEY)
        opus_4_8 = get_pricing_entry("claude-opus-4-8", provider="anthropic", api_key=API_KEY)
    assert entry is not None and opus_4_8 is not None
    assert entry.source == "official_docs_snapshot"
    assert entry.cache_read_cost_per_million is not None
    assert entry.cache_write_cost_per_million is not None
    assert (entry.input_cost_per_million, entry.output_cost_per_million) == (
        opus_4_8.input_cost_per_million, opus_4_8.output_cost_per_million,
    )


# ---------------------------------------------------------------------------
# Regression guards on neighbouring routes
# ---------------------------------------------------------------------------


def test_other_providers_are_unaffected():
    with _no_pool():
        assert (
            resolve_billing_route("gpt-5.6-luna", provider="openai-codex").billing_mode
            == "subscription_included"
        )
        assert (
            resolve_billing_route(
                "moonshotai/kimi-k3", provider="openrouter"
            ).billing_mode
            == "official_models_api"
        )
        assert (
            resolve_billing_route("gpt-5.6", provider="openai").billing_mode
            == "official_docs_snapshot"
        )


def test_anthropic_model_prefix_is_still_stripped():
    """Provider inference from an ``anthropic/…`` model string must keep
    working through the new branch."""
    with _no_pool():
        route = resolve_billing_route("anthropic/claude-opus-5", api_key=API_KEY)
    assert route.provider == "anthropic"
    assert route.model == "claude-opus-5"
