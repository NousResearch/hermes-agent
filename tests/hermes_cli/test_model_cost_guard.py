from decimal import Decimal

from agent.models_dev import ModelInfo
from agent.usage_pricing import PricingEntry
from hermes_cli.model_cost_guard import expensive_model_warning


def test_no_warning_when_known_prices_are_at_threshold():
    info = ModelInfo(
        id="edge/model",
        name="edge/model",
        family="",
        provider_id="test",
        cost_input=20.0,
        cost_output=100.0,
    )

    assert expensive_model_warning("edge/model", provider="test", model_info=info) is None


def test_warns_when_models_dev_input_price_exceeds_threshold():
    info = ModelInfo(
        id="expensive/input",
        name="expensive/input",
        family="",
        provider_id="test",
        cost_input=20.01,
        cost_output=1.0,
    )

    warning = expensive_model_warning(
        "expensive/input",
        provider="test",
        model_info=info,
    )

    assert warning is not None
    assert warning.input_cost_per_million == Decimal("20.01")
    assert "EXPENSIVE MODEL WARNING" in warning.message
    assert "$20/M input" in warning.message


def test_warns_when_pricing_entry_output_price_exceeds_threshold(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "agent.usage_pricing.get_pricing_entry",
        lambda *_args, **_kwargs: PricingEntry(
            input_cost_per_million=Decimal("1.00"),
            output_cost_per_million=Decimal("100.01"),
            source="provider_models_api",
        ),
    )

    warning = expensive_model_warning("provider/expensive-output", provider="openrouter")

    assert warning is not None
    assert warning.output_cost_per_million == Decimal("100.01")
    assert "$100.01/M" in warning.message


def test_openai_gpt55_pro_adds_suggestion(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "agent.usage_pricing.get_pricing_entry",
        lambda *_args, **_kwargs: PricingEntry(
            input_cost_per_million=Decimal("25"),
            output_cost_per_million=Decimal("125"),
            source="provider_models_api",
        ),
    )

    warning = expensive_model_warning("openai/gpt-5.5-pro", provider="openrouter")

    assert warning is not None
    assert "did you mean to select openai/gpt-5.5?" in warning.message


def test_openai_gpt55_pro_warns_for_nous_portal_pricing(monkeypatch):
    monkeypatch.setattr("agent.models_dev.get_model_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda base_url, api_key="": {
            "openai/gpt-5.5-pro": {
                "pricing": {
                    "prompt": "0.000025",
                    "completion": "0.000125",
                }
            }
        },
    )

    warning = expensive_model_warning("openai/gpt-5.5-pro", provider="nous")

    assert warning is not None
    assert warning.input_cost_per_million == Decimal("25.000000")
    assert warning.output_cost_per_million == Decimal("125.000000")
    assert "did you mean to select openai/gpt-5.5?" in warning.message


# ---------------------------------------------------------------------------
# Custom-endpoint provider identity must be validated against the endpoint's
# base_url, never its display-name slug.  Regression coverage requested in the
# Friendli PR review: (1) the host-validated custom route resolves to the
# vendor's models.dev catalog (per-million pricing), and (2) a catalog miss —
# an endpoint the host map does not recognize — falls through to the per-token
# endpoint pricing path rather than mis-assigning a vendor's pricing.
# ---------------------------------------------------------------------------

_FRIENDLI_REGISTRY = {
    "friendli": {
        "id": "friendli",
        "name": "Friendli",
        "api": "https://api.friendli.ai/serverless/v1",
        "models": {
            "zai-org/GLM-5.2": {
                "id": "zai-org/GLM-5.2",
                "cost": {"input": 1.4, "output": 4.4},
                "limit": {"context": 1048576, "output": 1048576},
            },
            "zai-org/GLM-5.2-premium": {
                "id": "zai-org/GLM-5.2-premium",
                "cost": {"input": 50.0, "output": 1.0},
                "limit": {"context": 1048576, "output": 1048576},
            },
        },
    },
}


def test_mapped_custom_friendli_endpoint_does_not_spuriously_trip(monkeypatch):
    """The initial issue: a genuine Friendli custom endpoint resolved its
    per-million ``/models`` pricing through the unit-blind endpoint path,
    turning $1.4/M into $1,400,000/M and tripping the guard.  Validating the
    host (``api.friendli.ai``) resolves it to the ``friendli`` catalog instead,
    so a $1.4/M model does NOT trip the $20/M threshold.
    """
    from agent.models_dev import _reset_upstream_provider_host_index_cache

    _reset_upstream_provider_host_index_cache()
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: _FRIENDLI_REGISTRY,
    )

    warning = expensive_model_warning(
        "zai-org/GLM-5.2",
        provider="custom:friendli",
        base_url="https://api.friendli.ai/serverless/v1",
    )

    assert warning is None


def test_mapped_custom_friendli_endpoint_uses_catalog_per_million_pricing(monkeypatch):
    """When the mapped route's catalog price does exceed the threshold, the
    warning carries the real per-million value ($50/M), not the $50,000,000/M
    the unit-blind endpoint path would have produced.
    """
    from agent.models_dev import _reset_upstream_provider_host_index_cache

    _reset_upstream_provider_host_index_cache()
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: _FRIENDLI_REGISTRY,
    )

    warning = expensive_model_warning(
        "zai-org/GLM-5.2-premium",
        provider="custom:friendli",
        base_url="https://api.friendli.ai/serverless/v1",
    )

    assert warning is not None
    assert warning.input_cost_per_million == Decimal("50.0")
    # Per-million catalog value, NOT the $50,000,000/M a second ×1M would yield.
    assert "Input tokens: $50.00/M" in warning.message


def test_imposter_friendli_endpoint_does_not_inherit_friendli_pricing(monkeypatch):
    """A catalog miss: an endpoint slugged ``custom:friendli`` but pointed at an
    unrelated host must NOT pick up Friendli catalog pricing — vendor identity
    is the base_url, not the display name.  It falls through to the per-token
    endpoint pricing path instead.
    """
    from agent.models_dev import _reset_upstream_provider_host_index_cache

    # The friendli registry is loaded, but the imposter host does not match the
    # registered ``api.friendli.ai`` host, so no vendor identity is assigned.
    _reset_upstream_provider_host_index_cache()
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: _FRIENDLI_REGISTRY)
    endpoint_meta = {
        # resolve_billing_route strips the "zai-org/" prefix for the slug
        # fallback route, so alias under both keys (as the real fetcher does).
        "zai-org/GLM-5.2": {"pricing": {"prompt": "0.000025", "completion": "0.00005"}},
        "GLM-5.2": {"pricing": {"prompt": "0.000025", "completion": "0.00005"}},
    }
    monkeypatch.setattr(
        "agent.usage_pricing.fetch_endpoint_model_metadata",
        lambda base_url, api_key="": endpoint_meta,
    )

    warning = expensive_model_warning(
        "zai-org/GLM-5.2",
        provider="custom:friendli",
        base_url="https://example-friendli-imposter.dev/v1",
    )

    assert warning is not None
    # Endpoint path scales per-token by 1M: $0.000025/token -> $25/M, NOT $1.4/M
    # (Friendli catalog) and NOT $25,000,000 (a second spurious ×1M).
    assert warning.input_cost_per_million == Decimal("25.000000")
