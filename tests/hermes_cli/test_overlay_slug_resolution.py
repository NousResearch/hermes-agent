"""Test that overlay providers with mismatched models.dev keys resolve correctly.

HERMES_OVERLAYS keys may be models.dev IDs (e.g. "github-copilot") while
_PROVIDER_MODELS and config.yaml use Hermes IDs ("copilot").  The slug
resolution in list_authenticated_providers() Section 2 must bridge this gap.

Covers: #5223, #6492
"""

import os
from unittest.mock import patch


from hermes_cli.model_switch import list_authenticated_providers


# -- Copilot slug resolution (env var path) ----------------------------------

@patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "fake-ghu"}, clear=False)
def test_copilot_uses_hermes_slug():
    """github-copilot overlay should resolve to slug='copilot' with curated models."""
    providers = list_authenticated_providers(current_provider="copilot")

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)
    assert copilot is not None, "copilot should appear when COPILOT_GITHUB_TOKEN is set"
    assert copilot["total_models"] > 0, "copilot should have curated models"
    assert copilot["is_current"] is True

    # Must NOT appear under the models.dev key
    gh_copilot = next((p for p in providers if p["slug"] == "github-copilot"), None)
    assert gh_copilot is None, "github-copilot slug should not appear (resolved to copilot)"




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
