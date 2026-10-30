"""Test that overlay providers with mismatched models.dev keys resolve correctly.

HERMES_OVERLAYS keys may be models.dev IDs (e.g. "kimi-for-coding") while
_PROVIDER_MODELS and config.yaml use Hermes IDs ("kimi-coding").  The slug
resolution in list_authenticated_providers() Section 2 must bridge this gap.

Covers: #5223, #6492
"""

import os
from unittest.mock import patch


from hermes_cli.model_switch import list_authenticated_providers


# -- Overlay slug resolution (env var path) ----------------------------------

@patch.dict(os.environ, {"KIMI_API_KEY": "fake-kimi"}, clear=False)
def test_overlay_provider_uses_hermes_slug():
    """kimi-for-coding overlay should resolve to slug='kimi-coding' with curated models."""
    providers = list_authenticated_providers(current_provider="kimi-coding")

    kimi = next((p for p in providers if p["slug"] == "kimi-coding"), None)
    assert kimi is not None, "kimi-coding should appear when KIMI_API_KEY is set"
    assert kimi["total_models"] > 0, "kimi-coding should have curated models"
    assert kimi["is_current"] is True

    # Must NOT appear under the models.dev key
    kimi_md = next((p for p in providers if p["slug"] == "kimi-for-coding"), None)
    assert kimi_md is None, "kimi-for-coding slug should not appear (resolved to kimi-coding)"




# -- kimi-for-coding alias in auth.py ----------------------------------------



# -- Generic slug mismatch providers -----------------------------------------






def test_mapped_provider_credential_pool_visibility(monkeypatch):
    """Mapped providers should appear when credentials live only in auth-store credential_pool."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {"google-ai-studio": {"env": ["GEMINI_API_KEY"]}})
    monkeypatch.setattr("agent.models_dev.PROVIDER_TO_MODELS_DEV", {"gemini": "google-ai-studio"})
    monkeypatch.setattr(
        "hermes_cli.auth._load_auth_store",
        lambda: {"providers": {}, "credential_pool": {"gemini": {"token": "fake"}}},
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    providers = list_authenticated_providers(current_provider="gemini")

    gemini = next((p for p in providers if p["slug"] == "gemini"), None)
    assert gemini is not None, "gemini should appear when auth-store credential_pool has creds"
    assert gemini["is_current"] is True
    assert gemini["total_models"] > 0
