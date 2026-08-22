"""Regression tests for xAI provider label disambiguation."""

import agent.models_dev as models_dev
from hermes_cli.models import provider_label
from hermes_cli.providers import get_label


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label():
    """The model picker must distinguish xAI API-key and OAuth providers."""
    assert get_label("xai") == "xAI"
    assert get_label("xai-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"
    assert get_label("grok-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"


def test_xai_label_override_wins_when_models_dev_unavailable(monkeypatch):
    """The _LABEL_OVERRIDES entry must supply "xAI" even on the catalog-down
    fallback path.

    When the models.dev provider lookup is unavailable, ``get_provider`` can no
    longer contribute a display name from the live catalog, so the display name
    would otherwise degrade to the raw canonical slug ("xai"). ``get_label``
    consults ``_LABEL_OVERRIDES`` first, so the override must still win. This
    guards the regression where a missing override let the fallback surface the
    lowercase slug instead of "xAI".
    """

    def _unavailable(*args, **kwargs):
        raise RuntimeError("models.dev catalog unavailable")

    monkeypatch.setattr(models_dev, "get_provider_info", _unavailable)

    assert get_label("xai") == "xAI"


