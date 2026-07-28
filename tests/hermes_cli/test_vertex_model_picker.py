"""Vertex visibility in the /model picker (follow-up to PR #56688).

Community verification of the vertex-registration fix found two remaining
gaps that kept the provider invisible/unusable in the /model menu:

1. ``list_authenticated_providers`` had no credential gate for the
   ``vertex`` auth_type (only ``aws_sdk`` was special-cased), so the
   provider was silently hidden even when fully configured.
2. Vertex's OpenAI-compatible endpoint has no ``/models`` listing route,
   so without a curated ``_PROVIDER_MODELS["vertex"]`` entry the picker
   only ever showed the currently-configured model.

No network calls.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import _PROVIDER_MODELS


def test_vertex_has_curated_model_list():
    """Vertex has no /models route — the picker needs a static curated list."""
    models = _PROVIDER_MODELS.get("vertex")
    assert models, "_PROVIDER_MODELS must have a non-empty 'vertex' entry"
    # Vertex's openapi endpoint expects the google/ publisher prefix.
    assert all(m.startswith("google/") for m in models)


def test_vertex_appears_when_credentials_configured():
    """has_vertex_credentials() == True must surface vertex in the picker."""
    with patch("agent.vertex_adapter.has_vertex_credentials", return_value=True):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    vertex = next((p for p in providers if p["slug"] == "vertex"), None)
    assert vertex is not None, "vertex should appear when credentials are configured"
    assert vertex["models"], "vertex row must carry the curated model list"
    assert "google/gemini-3-pro-preview" in vertex["models"]


def test_vertex_hidden_without_credentials():
    """No service-account path / project override → vertex stays hidden."""
    with patch("agent.vertex_adapter.has_vertex_credentials", return_value=False):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    vertex = next((p for p in providers if p["slug"] == "vertex"), None)
    assert vertex is None, "vertex should not appear without credentials"
