"""External (dynamic-catalog) pricing source: models.dev by default.

Every test seeds ``agent.models_dev``'s in-memory registry cache directly, so
the whole suite is hermetic — no network, no disk cache, and no dependency on
what models.dev happens to publish today.
"""

from decimal import Decimal

import pytest

import agent.models_dev as models_dev
import agent.usage_pricing as usage_pricing
from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    get_pricing_entry,
)


@pytest.fixture
def seeded_models_dev(monkeypatch):
    """Install a deterministic models.dev registry and return a seeder."""

    def seed(registry):
        monkeypatch.setattr(models_dev, "_models_dev_cache", registry)
        monkeypatch.setattr(models_dev, "_models_dev_cache_time", float("inf"))

    seed({})
    return seed


@pytest.fixture
def default_source_order(monkeypatch):
    """Pin the source order to the shipped default.

    Keeps a test independent of whatever ``config.yaml`` the surrounding
    HERMES_HOME happens to contain. Tests that exercise the config knob itself
    deliberately do NOT use this.
    """
    monkeypatch.setattr(
        usage_pricing, "_pricing_source_order", lambda: ("models_dev", "openrouter")
    )


def write_pricing_config(body: str) -> None:
    """Write ``config.yaml`` into the test's HERMES_HOME and drop the cache."""
    from hermes_cli.config import _LOAD_CONFIG_CACHE, get_config_path

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    _LOAD_CONFIG_CACHE.clear()


# ---------------------------------------------------------------------------
# _extract_cost — the models.dev cost-block reader
# ---------------------------------------------------------------------------


def test_extract_cost_reads_all_four_rate_classes():
    cost = models_dev._extract_cost(
        {"cost": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25}}
    )

    assert cost == {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write": 6.25,
    }


def test_extract_cost_omits_absent_cache_rates_rather_than_zeroing_them():
    """An absent cache rate must be absent, not 0.

    A 0 would be read downstream as "this route bills cache tokens for free",
    which silently under-bills. Absence is what makes ``estimate_usage_cost``
    report ``unknown`` for a route whose cache rates aren't published.
    """
    cost = models_dev._extract_cost({"cost": {"input": 3, "output": 15}})

    assert cost == {"input": 3.0, "output": 15.0}


def test_extract_cost_ignores_long_context_tier_surcharges():
    """Only the flat base rates are read; a tiers block must not leak in."""
    cost = models_dev._extract_cost(
        {
            "cost": {
                "input": 5,
                "output": 30,
                "tiers": [{"input": 10, "output": 45}],
                "context_over_200k": {"input": 10, "output": 45},
            }
        }
    )

    assert cost == {"input": 5.0, "output": 30.0}


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {},
        {"cost": None},
        {"cost": {}},
        {"cost": {"cache_read": 1.0}},
        {"cost": {"input": True, "output": False}},
        {"cost": {"input": "5", "output": "25"}},
        {"cost": {"input": -1, "output": -2}},
    ],
    ids=[
        "not-a-dict",
        "no-cost-key",
        "null-cost",
        "empty-cost",
        "cache-rate-only",
        "bools-are-not-rates",
        "strings-are-not-rates",
        "negative-rates",
    ],
)
def test_extract_cost_returns_none_for_unpriced_or_malformed_entries(entry):
    assert models_dev._extract_cost(entry) is None


# ---------------------------------------------------------------------------
# lookup_models_dev_pricing — provider-scoped resolution
# ---------------------------------------------------------------------------


def test_lookup_pricing_resolves_through_the_provider_id_mapping(seeded_models_dev):
    """A Hermes provider slug resolves via PROVIDER_TO_MODELS_DEV, not verbatim.

    ``gemini`` is the Hermes slug; models.dev files those models under
    ``google``. Same mapping the context-window lookup already uses.
    """
    seeded_models_dev(
        {"google": {"models": {"gemini-9-pro": {"cost": {"input": 2, "output": 12}}}}}
    )

    assert models_dev.lookup_models_dev_pricing("gemini", "gemini-9-pro") == {
        "input": 2.0,
        "output": 12.0,
    }


def test_lookup_pricing_matches_model_ids_case_insensitively(seeded_models_dev):
    seeded_models_dev(
        {"anthropic": {"models": {"claude-x-1": {"cost": {"input": 1, "output": 2}}}}}
    )

    assert models_dev.lookup_models_dev_pricing("anthropic", "CLAUDE-X-1") == {
        "input": 1.0,
        "output": 2.0,
    }


def test_lookup_pricing_is_provider_scoped_not_a_global_id_scan(seeded_models_dev):
    """A model id must never be priced off some *other* provider's catalog.

    ``custom``/``local``/``unknown`` are not in ``PROVIDER_TO_MODELS_DEV``. If
    the lookup fell back to scanning every provider for the bare id, a
    self-hosted model sharing a name with a hosted one would be billed at the
    hosted vendor's rate. It must resolve to None instead.
    """
    seeded_models_dev(
        {"anthropic": {"models": {"claude-x-1": {"cost": {"input": 1, "output": 2}}}}}
    )

    assert models_dev.lookup_models_dev_pricing("custom", "claude-x-1") is None
    assert models_dev.lookup_models_dev_pricing("local", "claude-x-1") is None
    assert models_dev.lookup_models_dev_pricing("unknown", "claude-x-1") is None


def test_lookup_pricing_returns_none_for_a_model_absent_from_the_catalog(seeded_models_dev):
    seeded_models_dev({"anthropic": {"models": {}}})

    assert models_dev.lookup_models_dev_pricing("anthropic", "claude-x-1") is None


# ---------------------------------------------------------------------------
# get_pricing_entry — snapshot precedence and the external fallback
# ---------------------------------------------------------------------------


def test_snapshot_miss_now_prices_from_models_dev(seeded_models_dev, default_source_order):
    """The regression this change fixes.

    A model absent from the curated snapshot used to resolve to no pricing
    entry at all, so every turn on it recorded cost "n/a" until a maintainer
    hand-edited the table. It must now price from the external catalog.
    """
    model = "claude-not-in-the-snapshot-1"
    assert ("anthropic", model) not in usage_pricing._OFFICIAL_DOCS_PRICING

    seeded_models_dev(
        {
            "anthropic": {
                "models": {
                    model: {
                        "cost": {
                            "input": 5,
                            "output": 25,
                            "cache_read": 0.5,
                            "cache_write": 6.25,
                        }
                    }
                }
            }
        }
    )

    entry = get_pricing_entry(model, provider="anthropic")

    assert entry is not None
    assert entry.input_cost_per_million == Decimal("5")
    assert entry.output_cost_per_million == Decimal("25")
    assert entry.cache_read_cost_per_million == Decimal("0.5")
    assert entry.cache_write_cost_per_million == Decimal("6.25")
    assert entry.pricing_version == "models-dev-api"
    assert entry.source_url == "https://models.dev/api.json"


def test_models_dev_rates_are_per_million_not_per_token(
    seeded_models_dev, default_source_order
):
    """models.dev publishes per-MILLION USD; OpenRouter publishes per-token.

    Reusing the OpenRouter converter here would multiply by 1e6 and overcharge
    by a factor of a million. This pins the unit contract end to end: 1M input
    tokens at a published rate of 5 costs exactly $5.
    """
    model = "unit-contract-probe-1"
    seeded_models_dev(
        {"anthropic": {"models": {model: {"cost": {"input": 5, "output": 25}}}}}
    )

    result = estimate_usage_cost(
        model,
        CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        provider="anthropic",
    )

    assert result.status == "estimated"
    assert result.amount_usd == Decimal("30")


def test_curated_snapshot_always_beats_the_external_catalog(
    seeded_models_dev, default_source_order
):
    """The snapshot encodes corrections a live catalog can't know.

    The canonical case is list-price-not-intro-promo: a vendor's launch
    discount shows up in models.dev, but billing must follow the standing list
    price the snapshot records. A snapshot hit must never be overridden.
    """
    (provider, model), snapshot = next(iter(usage_pricing._OFFICIAL_DOCS_PRICING.items()))
    decoy = (snapshot.input_cost_per_million or Decimal("0")) + Decimal("1234")

    seeded_models_dev(
        {
            models_dev.PROVIDER_TO_MODELS_DEV.get(provider, provider): {
                "models": {model: {"cost": {"input": float(decoy), "output": float(decoy)}}}
            }
        }
    )

    entry = get_pricing_entry(model, provider=provider)

    assert entry is not None
    assert entry.input_cost_per_million == snapshot.input_cost_per_million
    assert entry.output_cost_per_million == snapshot.output_cost_per_million
    assert entry.input_cost_per_million != decoy


def test_unpriced_stays_unpriced_when_no_catalog_has_the_model(
    seeded_models_dev, default_source_order, monkeypatch
):
    seeded_models_dev({"anthropic": {"models": {}}})
    monkeypatch.setattr(usage_pricing, "fetch_model_metadata", dict)

    assert get_pricing_entry("claude-nobody-has-heard-of-1", provider="anthropic") is None


def test_a_catalog_failure_degrades_to_unpriced_instead_of_raising(
    monkeypatch, default_source_order
):
    """Pricing must never raise into a turn."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("catalog unreachable")

    monkeypatch.setattr(models_dev, "fetch_models_dev", _boom)
    monkeypatch.setattr(usage_pricing, "fetch_model_metadata", _boom)

    assert get_pricing_entry("claude-anything-1", provider="anthropic") is None


def test_the_non_primary_source_is_used_when_the_primary_misses(
    monkeypatch, seeded_models_dev, default_source_order
):
    """models.dev primary, OpenRouter fallback: a models.dev miss still prices.

    The OpenRouter lane runs for real here (only its HTTP catalog is stubbed),
    so this also pins the unit difference between the two sources: OpenRouter
    publishes per-TOKEN rates, which the shared converter scales to
    per-million.
    """
    model = "only-openrouter-knows-this-1"
    seeded_models_dev({"anthropic": {"models": {}}})
    monkeypatch.setattr(
        usage_pricing,
        "fetch_model_metadata",
        lambda: {model: {"pricing": {"prompt": "0.000007", "completion": "0.000014"}}},
    )

    entry = get_pricing_entry(model, provider="anthropic")

    assert entry is not None
    assert entry.pricing_version == "openrouter-models-api"
    assert entry.input_cost_per_million == Decimal("7.000000")
    assert entry.output_cost_per_million == Decimal("14.000000")


def test_openrouter_routes_still_price_from_openrouter_first(
    monkeypatch, seeded_models_dev, default_source_order
):
    """A ``provider: openrouter`` route bills at OpenRouter's rate card.

    OpenRouter is the actual billing vendor for that route, so its own models
    API stays primary even though models.dev is the default *fallback* catalog
    everywhere else.
    """
    model = "vendor/model-1"
    seeded_models_dev(
        {"openrouter": {"models": {model: {"cost": {"input": 99, "output": 99}}}}}
    )
    monkeypatch.setattr(
        usage_pricing,
        "fetch_model_metadata",
        lambda: {model: {"pricing": {"prompt": "0.000003", "completion": "0.000015"}}},
    )

    entry = get_pricing_entry(model, provider="openrouter")

    assert entry is not None
    assert entry.pricing_version == "openrouter-models-api"
    assert entry.input_cost_per_million == Decimal("3.000000")


def test_subscription_included_routes_are_untouched_by_the_external_catalog(
    seeded_models_dev, default_source_order
):
    """An included route reports $0/included, never a catalog rate."""
    seeded_models_dev({"openai": {"models": {"gpt-9": {"cost": {"input": 100, "output": 200}}}}})

    entry = get_pricing_entry("gpt-9", provider="openai-codex")

    assert entry is not None
    assert entry.pricing_version == "included-route"
    assert entry.input_cost_per_million == Decimal("0")


# ---------------------------------------------------------------------------
# pricing.external_source — the config knob
# ---------------------------------------------------------------------------


def test_default_source_order_is_models_dev_then_openrouter():
    write_pricing_config("model:\n  default: gpt-4o\n")

    assert usage_pricing._pricing_source_order() == ("models_dev", "openrouter")


def test_config_can_select_openrouter_as_the_primary_source():
    write_pricing_config("pricing:\n  external_source: openrouter\n")

    assert usage_pricing._pricing_source_order() == ("openrouter", "models_dev")


@pytest.mark.parametrize(
    "body",
    [
        "pricing:\n  external_source: not_a_real_source\n",
        "pricing:\n  external_source: 17\n",
        "pricing: not-a-mapping\n",
        "pricing: {}\n",
    ],
    ids=["unknown-name", "wrong-type", "section-not-a-mapping", "empty-section"],
)
def test_an_unusable_config_value_degrades_to_the_default_order(body):
    write_pricing_config(body)

    assert usage_pricing._pricing_source_order() == ("models_dev", "openrouter")


def test_source_order_always_covers_every_valid_source():
    """Invariant: the order is a permutation of the valid sources.

    Whichever source is selected, the other stays reachable as fallback —
    selecting one must never make a model that only the other catalog knows
    about unpriceable.
    """
    for source in usage_pricing._VALID_PRICING_SOURCES:
        write_pricing_config(f"pricing:\n  external_source: {source}\n")
        order = usage_pricing._pricing_source_order()

        assert order[0] == source
        assert sorted(order) == sorted(usage_pricing._VALID_PRICING_SOURCES)
        assert len(set(order)) == len(order)


def test_every_named_source_has_a_builder():
    """Invariant: the config accepts exactly the sources that can be built."""
    assert set(usage_pricing._PRICING_SOURCE_BUILDERS) == set(
        usage_pricing._VALID_PRICING_SOURCES
    )
    assert usage_pricing._DEFAULT_PRICING_SOURCE in usage_pricing._VALID_PRICING_SOURCES


def test_selecting_openrouter_makes_it_the_primary_for_a_snapshot_miss(
    monkeypatch, seeded_models_dev
):
    """The knob has real effect: with openrouter selected, it is tried first."""
    write_pricing_config("pricing:\n  external_source: openrouter\n")

    model = "both-catalogs-know-this-1"
    seeded_models_dev(
        {"anthropic": {"models": {model: {"cost": {"input": 1, "output": 1}}}}}
    )
    monkeypatch.setattr(
        usage_pricing,
        "fetch_model_metadata",
        lambda: {model: {"pricing": {"prompt": "0.000003", "completion": "0.000015"}}},
    )

    entry = get_pricing_entry(model, provider="anthropic")

    assert entry is not None
    assert entry.pricing_version == "openrouter-models-api"
    assert entry.input_cost_per_million == Decimal("3.000000")


# ---------------------------------------------------------------------------
# Provider scoping of the external catalogs (the billing-vendor guarantee)
# ---------------------------------------------------------------------------
#
# ``lookup_models_dev_pricing`` documents that a self-hosted model sharing an
# id with a hosted one is never billed at the hosted vendor's rate. The
# OpenRouter catalog resolves by BARE MODEL ID with no provider scoping, so
# that guarantee only holds if unidentified-vendor routes refuse every
# external catalog. These tests pin exactly that.


@pytest.fixture
def openrouter_catalog_knows_glm5(monkeypatch):
    """OpenRouter publishes a hosted 'glm-5' at $0.95/M input."""
    monkeypatch.setattr(
        usage_pricing,
        "fetch_model_metadata",
        lambda: {
            "glm-5": {"pricing": {"prompt": "0.00000095", "completion": "0.0000038"}}
        },
    )
    # raising=False so this fixture also applies against a tree where the
    # cache-only accessor does not exist yet — that keeps the RED proof
    # behavioural (a wrong PRICE) instead of a fixture-setup error.
    monkeypatch.setattr(
        usage_pricing,
        "cached_model_metadata",
        lambda: {
            "glm-5": {"pricing": {"prompt": "0.00000095", "completion": "0.0000038"}}
        },
        raising=False,
    )


@pytest.mark.parametrize("provider", [None, "unknown", "custom", "local"])
def test_an_unidentified_billing_vendor_is_never_priced_from_a_hosted_catalog(
    provider, seeded_models_dev, default_source_order, openrouter_catalog_knows_glm5
):
    """The core guarantee: no identified vendor => no external catalog at all.

    A self-hosted 'glm-5' must not be billed at OpenRouter's hosted rate just
    because the id collides. Unpriced is the correct answer — a wrong price is
    silently attributed to the user's spend, while 'n/a' is visible.
    """
    route = usage_pricing.resolve_billing_route("glm-5", provider=provider)
    assert route.billing_mode == "unknown", "precondition: vendor unidentified"

    assert usage_pricing._external_pricing_entry(route) is None
    assert get_pricing_entry("glm-5", provider=provider) is None
    assert usage_pricing.has_known_pricing("glm-5", provider) is False


def test_the_bare_id_scan_still_finds_the_model_it_is_refused_for(
    openrouter_catalog_knows_glm5,
):
    """Control: the refusal is the guard, not an empty catalog.

    Without this, the test above would pass vacuously if the OpenRouter stub
    simply had no 'glm-5'.
    """
    route = usage_pricing.resolve_billing_route("glm-5", provider="unknown")
    entry = usage_pricing._openrouter_pricing_entry(route)

    assert entry is not None, "the bare-id scan does match; the guard is what refuses"
    assert entry.input_cost_per_million == Decimal("0.950000")


def test_an_identified_vendor_still_reaches_the_external_catalog(
    seeded_models_dev, default_source_order
):
    """The guard must not disable the feature for legitimate routes."""
    seeded_models_dev(
        {"anthropic": {"models": {"claude-brand-new-1": {"cost": {"input": 3, "output": 15}}}}}
    )
    entry = get_pricing_entry("claude-brand-new-1", provider="anthropic")

    assert entry is not None
    assert entry.input_cost_per_million == Decimal("3")


def test_openrouter_routes_keep_their_own_bare_id_lookup(openrouter_catalog_knows_glm5):
    """On an OpenRouter route the flat id space IS the correct namespace."""
    entry = get_pricing_entry("glm-5", provider="openrouter")

    assert entry is not None
    assert entry.input_cost_per_million == Decimal("0.950000")


# ---------------------------------------------------------------------------
# has_known_pricing is a pure predicate — no network from a display loop
# ---------------------------------------------------------------------------


def test_has_known_pricing_makes_no_outbound_connection(monkeypatch):
    """It runs per-row in display loops; a blocking fetch there is a defect."""
    import socket

    import agent.model_metadata as model_metadata
    import agent.models_dev as models_dev

    # Cold caches, so only a network call could produce an answer.
    monkeypatch.setattr(model_metadata, "_model_metadata_cache", {})
    monkeypatch.setattr(model_metadata, "_model_metadata_cache_time", 0)
    monkeypatch.setattr(model_metadata, "_load_model_metadata_disk_cache", dict)
    monkeypatch.setattr(model_metadata, "_model_metadata_disk_cache_age_seconds", lambda: None)
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    monkeypatch.setattr(models_dev, "_load_disk_cache", dict)

    attempts = []

    def _refuse(self, address):
        attempts.append(address)
        raise AssertionError(f"has_known_pricing opened a connection to {address}")

    monkeypatch.setattr(socket.socket, "connect", _refuse)

    for model, provider in [
        ("glm-5", None),
        ("gpt-4o", "openai"),
        ("some-model", "openrouter"),
        ("another", "anthropic"),
    ]:
        usage_pricing.has_known_pricing(model, provider)

    assert attempts == []


def test_has_known_pricing_still_answers_true_from_a_warm_cache(
    seeded_models_dev, default_source_order
):
    """Cache-only must not mean always-False for catalog-priced models."""
    seeded_models_dev(
        {"anthropic": {"models": {"claude-cached-1": {"cost": {"input": 3, "output": 15}}}}}
    )
    assert usage_pricing.has_known_pricing("claude-cached-1", "anthropic") is True
