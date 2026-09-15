"""GPT-6 Astra support contracts for the GitHub Copilot provider."""

from unittest.mock import patch

from hermes_cli.models import copilot_model_api_mode, provider_model_ids


def test_copilot_offline_fallback_keeps_gpt6_astra() -> None:
    """A transient catalog outage must not remove the GA model from the picker."""
    with patch("hermes_cli.models._PROVIDER_CATALOG_FETCHERS", {}):
        models = provider_model_ids("copilot")

    assert "gpt-6-astra" in models


def test_copilot_gpt6_astra_uses_responses_api() -> None:
    """Copilot exposes Astra only on /responses, never chat/completions."""
    assert copilot_model_api_mode("gpt-6-astra", catalog=[]) == "codex_responses"
