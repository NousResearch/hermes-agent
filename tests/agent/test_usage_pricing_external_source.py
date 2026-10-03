"""External pricing fallback and display-path cache safety.

The catalog fixtures are in-memory only: no test depends on live models.dev or
OpenRouter data.
"""

from decimal import Decimal

import pytest
import yaml

import agent.model_metadata as model_metadata
import agent.models_dev as models_dev
import agent.usage_pricing as usage_pricing
from agent.usage_pricing import get_pricing_entry


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
    monkeypatch.setattr(
        usage_pricing,
        "_pricing_source_order",
        lambda: ("models_dev", "openrouter"),
        raising=False,
    )


@pytest.fixture
def openrouter_catalog_knows_glm5(monkeypatch):
    catalog = {
        "glm-5": {
            "pricing": {
                "prompt": "0.00000095",
                "completion": "0.0000038",
            }
        }
    }
    monkeypatch.setattr(
        usage_pricing,
        "fetch_model_metadata",
        lambda *args, **kwargs: catalog,
    )
    return catalog


def test_models_dev_cost_reader_preserves_all_rate_classes():
    assert models_dev._extract_cost(
        {
            "cost": {
                "input": 5,
                "output": 25,
                "cache_read": 0.5,
                "cache_write": 6.25,
            }
        }
    ) == {
        "input": 5.0,
        "output": 25.0,
        "cache_read": 0.5,
        "cache_write": 6.25,
    }


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {},
        {"cost": {}},
        {"cost": {"cache_read": 1}},
        {"cost": {"input": True, "output": False}},
        {"cost": {"input": "5", "output": "25"}},
        {"cost": {"input": -1, "output": -2}},
    ],
)
def test_models_dev_cost_reader_rejects_unpriced_or_malformed_entries(entry):
    assert models_dev._extract_cost(entry) is None


def test_models_dev_pricing_is_provider_scoped(seeded_models_dev):
    seeded_models_dev(
        {
            "xiaomi": {
                "models": {
                    "mimo-new": {"cost": {"input": 1, "output": 3}},
                }
            }
        }
    )

    assert models_dev.lookup_models_dev_pricing("xiaomi", "mimo-new") == {
        "input": 1.0,
        "output": 3.0,
    }
    assert models_dev.lookup_models_dev_pricing("custom", "mimo-new") is None
    assert models_dev.lookup_models_dev_pricing("unknown", "mimo-new") is None


def test_snapshot_miss_prices_from_models_dev(seeded_models_dev, default_source_order):
    """A relay's own provider-scoped models.dev row prices it (opencode-go is not a vendor catalog)."""
    seeded_models_dev(
        {
            "opencode-go": {
                "models": {
                    "relay-model-1": {
                        "cost": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25}
                    }
                }
            }
        }
    )

    entry = get_pricing_entry("relay-model-1", provider="opencode-go")

    assert entry is not None
    assert entry.input_cost_per_million == Decimal("5")
    assert entry.output_cost_per_million == Decimal("25")
    assert entry.cache_read_cost_per_million == Decimal("0.5")
    assert entry.cache_write_cost_per_million == Decimal("6.25")
    assert entry.pricing_version == "models-dev-api"


def test_models_dev_rates_are_already_per_million(seeded_models_dev, default_source_order):
    """models.dev quotes $/1M tokens; a second 1e6 scale-up turns a $0.33 day into ~$330k."""
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    seeded_models_dev({"opencode-go": {"models": {"relay-model-1": {"cost": {"input": 0.3, "output": 1.2}}}}})

    cost = estimate_usage_cost(
        "relay-model-1", CanonicalUsage(input_tokens=1_000_000, output_tokens=0), provider="opencode-go"
    )

    assert (cost.status, cost.amount_usd) == ("estimated", Decimal("0.3"))


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("xai-oauth", "https://api.x.ai/v1"),  # vendor catalog on a subscription route
        ("openai-codex", ""),
        ("xai", "https://grok-relay.example.com/v1"),  # vendor provider id on someone else's host
        ("anthropic", "https://relay.example.com/v1"),
    ],
)
def test_vendor_rate_card_stays_behind_the_direct_gate(
    provider, base_url, seeded_models_dev, default_source_order, monkeypatch
):
    """The generic fallback must not hand a vendor's list price to a route the direct gate refused."""
    monkeypatch.setattr(usage_pricing, "fetch_endpoint_model_metadata", lambda *_a, **_k: {})
    seeded_models_dev(
        {
            "xai": {"models": {"grok-new": {"cost": {"input": 1.25, "output": 2.5}}}},
            "openai": {"models": {"grok-new": {"cost": {"input": 1.25, "output": 2.5}}}},
            "anthropic": {"models": {"grok-new": {"cost": {"input": 1.25, "output": 2.5}}}},
        }
    )

    route = usage_pricing.resolve_billing_route("grok-new", provider=provider, base_url=base_url)
    if route.billing_mode == "subscription_included":
        pytest.skip(f"{provider} is billed as included; nothing to price")
    assert usage_pricing._models_dev_scoped_pricing_entry(route) is None
    assert get_pricing_entry("grok-new", provider=provider, base_url=base_url) is None


def test_direct_vendor_route_is_still_priced_once(seeded_models_dev, default_source_order, monkeypatch):
    """The direct gate owns the vendor row; the generic source is not consulted a second time."""
    calls = []
    real = usage_pricing._models_dev_scoped_pricing_entry
    monkeypatch.setattr(
        usage_pricing, "_PRICING_SOURCE_BUILDERS",
        {**usage_pricing._PRICING_SOURCE_BUILDERS,
         "models_dev": lambda route, **kw: calls.append(route) or real(route, **kw)},
    )
    seeded_models_dev({"xai": {"models": {"grok-new": {"cost": {"input": 1.25, "output": 2.5}}}}})

    entry = get_pricing_entry("grok-new", provider="xai", base_url="https://api.x.ai/v1")

    assert entry is not None and entry.pricing_version == "models.dev"
    assert calls == []


def test_curated_snapshot_still_precedes_external_catalog(
    seeded_models_dev, default_source_order
):
    seeded_models_dev(
        {
            "openai": {
                "models": {
                    "gpt-4o": {"cost": {"input": 999, "output": 999}},
                }
            }
        }
    )

    entry = get_pricing_entry("gpt-4o", provider="openai")

    assert entry is not None
    assert entry.pricing_version != "models-dev-api"
    assert entry.input_cost_per_million != Decimal("999")


@pytest.mark.parametrize("provider", [None, "unknown", "custom", "local", "xiaomi"])
def test_openrouter_bare_id_lookup_is_reserved_for_openrouter_routes(
    provider, openrouter_catalog_knows_glm5
):
    route = usage_pricing.resolve_billing_route("glm-5", provider=provider)
    assert route.provider != "openrouter"

    assert usage_pricing._openrouter_pricing_entry(route) is None


def test_openrouter_guard_is_not_vacuous(openrouter_catalog_knows_glm5):
    entry = usage_pricing._pricing_entry_from_metadata(
        openrouter_catalog_knows_glm5,
        "glm-5",
        source_url="test",
        pricing_version="test",
    )

    assert entry is not None
    assert entry.input_cost_per_million == Decimal("0.95000000")


@pytest.mark.parametrize("provider", [None, "unknown", "custom", "local"])
def test_unknown_or_self_hosted_route_never_falls_through_to_openrouter(
    provider,
    seeded_models_dev,
    default_source_order,
    openrouter_catalog_knows_glm5,
):
    seeded_models_dev({"xiaomi": {"models": {}}})

    assert get_pricing_entry("glm-5", provider=provider) is None
    assert usage_pricing.has_known_pricing("glm-5", provider=provider) is False


def test_mapped_named_provider_reaches_models_dev_despite_unknown_billing_mode(
    seeded_models_dev, default_source_order
):
    seeded_models_dev(
        {
            "xiaomi": {
                "models": {
                    "mimo-new": {"cost": {"input": 1, "output": 3}},
                }
            }
        }
    )
    route = usage_pricing.resolve_billing_route("mimo-new", provider="xiaomi")
    assert route.billing_mode == "unknown"

    entry = get_pricing_entry("mimo-new", provider="xiaomi")

    assert entry is not None
    assert entry.input_cost_per_million == Decimal("1")
    assert entry.pricing_version == "models.dev"  # xiaomi is a direct first-party host on main


def test_openrouter_route_keeps_its_bare_id_rate_card(openrouter_catalog_knows_glm5):
    entry = get_pricing_entry("glm-5", provider="openrouter")

    assert entry is not None
    assert entry.input_cost_per_million == Decimal("0.95000000")
    assert entry.pricing_version == "openrouter-models-api"


def test_external_source_order_is_configurable(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"pricing": {"external_source": "openrouter"}},
    )

    assert usage_pricing._pricing_source_order() == ("openrouter", "models_dev")
    assert set(usage_pricing._pricing_source_order()) == set(
        usage_pricing._VALID_PRICING_SOURCES
    )


def test_documented_pricing_config_path_is_registered_and_writable(
    tmp_path, monkeypatch
):
    from hermes_cli import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert config._validate_config_key("pricing.external_source") == (True, None)

    config.set_config_value("pricing.external_source", "openrouter")

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["pricing"]["external_source"] == "openrouter"


def test_has_known_pricing_makes_no_outbound_connection(monkeypatch):
    import socket

    monkeypatch.setattr(model_metadata, "_model_metadata_cache", {})
    monkeypatch.setattr(model_metadata, "_model_metadata_cache_time", 0)
    monkeypatch.setattr(model_metadata, "_load_model_metadata_disk_cache", dict)
    monkeypatch.setattr(
        model_metadata,
        "_model_metadata_disk_cache_age_seconds",
        lambda: None,
    )
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    monkeypatch.setattr(models_dev, "_load_disk_cache", dict)

    attempts = []

    def refuse(self, address):
        attempts.append(address)
        raise AssertionError(f"has_known_pricing opened a connection to {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)

    for model, provider in [
        ("glm-5", None),
        ("gpt-4o", "openai"),
        ("some-model", "openrouter"),
        ("another", "anthropic"),
    ]:
        usage_pricing.has_known_pricing(model, provider)

    assert attempts == []


def test_has_known_pricing_uses_warm_models_dev_cache(
    seeded_models_dev, default_source_order
):
    seeded_models_dev(
        {
            "anthropic": {
                "models": {
                    "claude-cached-1": {"cost": {"input": 3, "output": 15}},
                }
            },
            "opencode-go": {"models": {"relay-cached-1": {"cost": {"input": 3, "output": 15}}}},
        }
    )

    assert usage_pricing.has_known_pricing("claude-cached-1", "anthropic") is True
    assert usage_pricing.has_known_pricing("relay-cached-1", "opencode-go") is True
