"""Test that opencode-go appears in /model list when credentials are set."""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


@patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "test-key"}, clear=False)
def test_opencode_go_appears_when_api_key_set():
    """opencode-go should appear in list_authenticated_providers when OPENCODE_GO_API_KEY is set."""
    providers = list_authenticated_providers(current_provider="openrouter")
    
    # Find opencode-go in results
    opencode_go = next((p for p in providers if p["slug"] == "opencode-go"), None)
    
    assert opencode_go is not None, "opencode-go should appear when OPENCODE_GO_API_KEY is set"
    assert opencode_go["models"] == ["glm-5.1", "glm-5", "kimi-k2.5", "mimo-v2-pro", "mimo-v2-omni", "minimax-m2.7", "minimax-m2.5"]
    # opencode-go can appear as "built-in" (from PROVIDER_TO_MODELS_DEV when
    # models.dev is reachable) or "hermes" (from HERMES_OVERLAYS fallback when
    # the API is unavailable, e.g. in CI).
    assert opencode_go["source"] in ("built-in", "hermes")


def test_opencode_go_not_appears_when_no_creds():
    """opencode-go should NOT appear when no credentials are set."""
    # Ensure OPENCODE_GO_API_KEY is not set
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENCODE_GO_API_KEY"}
    
    with patch.dict(os.environ, env_without_key, clear=True):
        providers = list_authenticated_providers(current_provider="openrouter")
        
        # opencode-go should not be in results
        opencode_go = next((p for p in providers if p["slug"] == "opencode-go"), None)
        assert opencode_go is None, "opencode-go should not appear without credentials"
