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


def _no_pool():
    return patch("hermes_cli.auth.read_credential_pool", return_value=[])


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
    with patch(
        "hermes_cli.auth.read_credential_pool", return_value=_pool((OAUTH_TOKEN, 0))
    ):
        route = resolve_billing_route(
            "claude-opus-5", provider="anthropic", api_key=API_KEY
        )
    assert route.billing_mode == "official_docs_snapshot"


# ---------------------------------------------------------------------------
# Credential-pool fallback (the common case: token is not in the environment)
# ---------------------------------------------------------------------------


def test_pool_oauth_token_detected_without_explicit_key():
    with patch(
        "hermes_cli.auth.read_credential_pool", return_value=_pool((OAUTH_TOKEN, 0))
    ):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "subscription_included"


def test_pool_respects_priority_order():
    """Priority 0 is the seat actually used; a lower-priority API key entry
    must not flip an OAuth seat back to metered (or vice versa)."""
    with patch(
        "hermes_cli.auth.read_credential_pool",
        return_value=_pool((API_KEY, 1), (OAUTH_TOKEN, 0)),
    ):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "subscription_included"

    with patch(
        "hermes_cli.auth.read_credential_pool",
        return_value=_pool((OAUTH_TOKEN, 1), (API_KEY, 0)),
    ):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "official_docs_snapshot"


def test_pool_skips_entries_without_token():
    """Claude Code seats can appear with an empty access_token; skip them
    rather than reading the blank as 'not OAuth'."""
    entries = [
        {"label": "empty", "auth_type": "oauth", "priority": 0, "access_token": ""},
        {"label": "real", "auth_type": "oauth", "priority": 1, "access_token": OAUTH_TOKEN},
    ]
    with patch("hermes_cli.auth.read_credential_pool", return_value=entries):
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "subscription_included"


def test_empty_pool_stays_metered():
    with _no_pool():
        route = resolve_billing_route("claude-opus-5", provider="anthropic")
    assert route.billing_mode == "official_docs_snapshot"


def test_pool_read_failure_fails_closed():
    """Any error in detection must keep the metered path, never zero out cost."""
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


def test_metered_cost_uses_published_rates():
    """Opus 5: $5 in / $25 out / $0.50 cache read / $6.25 cache write per MTok.
    Source: platform.claude.com/docs/en/about-claude/pricing
    """
    with _no_pool():
        result = estimate_usage_cost(
            "claude-opus-5", USAGE, provider="anthropic", api_key=API_KEY
        )
    assert result.status == "estimated"
    expected = (
        Decimal(1000) * Decimal("5.00")
        + Decimal(500) * Decimal("25.00")
        + Decimal(20000) * Decimal("0.50")
        + Decimal(2000) * Decimal("6.25")
    ) / Decimal(1_000_000)
    assert result.amount_usd == expected


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


def test_opus_5_has_metered_pricing_entry():
    """Opus 5 shipped 2026-07-24 with no entry, so metered users saw
    cost_status='unknown' and a $0 total for every Opus 5 session."""
    with _no_pool():
        entry = get_pricing_entry("claude-opus-5", provider="anthropic", api_key=API_KEY)
    assert entry is not None
    assert entry.input_cost_per_million == Decimal("5.00")
    assert entry.output_cost_per_million == Decimal("25.00")
    assert entry.cache_read_cost_per_million == Decimal("0.50")
    # 5m-TTL cache write rate; the 1h rate ($10.00) is not representable here.
    assert entry.cache_write_cost_per_million == Decimal("6.25")


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
