"""Publisher-namespaced catalogs must not hijack a user's provider.

Vertex's catalog ids carry the PUBLISHER namespace (``google/gemini-4-argon``), but that is also
the aggregator slug form every metered aggregator (nous / openrouter / ai-gateway / kilocode)
publishes. A prefix match against such a catalog silently switched the user onto that provider's
own billing account — a GCP project for Vertex — the moment they typed a model name their
aggregator already served.

The invariant: a publisher-namespaced catalog id never AUTO-SWITCHES the provider away from an
aggregator. The provider stays reachable by name, in its own picker, and via an explicit model
choice for that provider.
"""

from __future__ import annotations

import pytest

from hermes_cli import models
from hermes_cli.models_catalog_static import _PROVIDER_MODELS, _VENDOR_NAMESPACE_CATALOG_PROVIDERS


def _aggregators() -> list[str]:
    return sorted(p for p in ("nous", "openrouter", "ai-gateway", "kilocode") if p in _PROVIDER_MODELS)


@pytest.mark.parametrize("provider", _aggregators())
def test_publisher_namespace_id_does_not_switch_off_aggregator(provider):
    """Typing an id that a publisher-namespaced catalog also lists keeps the aggregator."""
    for pid in sorted(_VENDOR_NAMESPACE_CATALOG_PROVIDERS):
        for model in _PROVIDER_MODELS.get(pid, []):
            detected = models.detect_static_provider_for_model(model, provider)
            assert detected is None or detected[0] != pid, (
                f"{model!r} (catalog {pid!r}) hijacked the {provider!r} provider")


@pytest.mark.parametrize("provider", _aggregators())
def test_publisher_namespace_provider_still_reachable_by_name(provider):
    """The guard suppresses INFERENCE only — naming the provider still switches to it."""
    for pid in sorted(_VENDOR_NAMESPACE_CATALOG_PROVIDERS):
        detected = models.detect_static_provider_for_model(pid, provider)
        assert detected is not None and detected[0] == pid, (
            f"provider {pid!r} must stay reachable by name")


def test_vertex_reachable_from_its_own_provider():
    """A vertex user typing a vertex id is never re-routed elsewhere."""
    for model in _PROVIDER_MODELS["vertex"]:
        detected = models.detect_static_provider_for_model(model, "vertex")
        assert detected is None or detected[0] == "vertex"


def test_vertex_still_listed_in_the_picker():
    """Vertex keeps its own curated picker rows (no /models route on Vertex)."""
    from unittest.mock import patch

    from hermes_cli.model_switch import list_authenticated_providers

    with patch("agent.vertex_adapter.has_vertex_credentials", return_value=True):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    vertex = next((p for p in providers if p["slug"] == "vertex"), None)
    assert vertex is not None, "vertex should appear when credentials are configured"
    assert vertex["models"], "vertex row must carry the curated model list"


def test_catalog_ids_carry_the_publisher_prefix():
    """Vertex ids keep the ``google/`` publisher prefix the endpoint expects."""
    assert all(m.startswith("google/") for m in _PROVIDER_MODELS["vertex"])
