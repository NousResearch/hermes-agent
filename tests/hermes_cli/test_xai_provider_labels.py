"""Regression tests for xAI provider label disambiguation."""

import agent.models_dev as models_dev
import hermes_cli.providers as providers
from hermes_cli.models import provider_label
from hermes_cli.providers import get_label


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label():
    """The model picker must distinguish xAI API-key and OAuth providers."""
    assert get_label("xai") == "xAI"
    assert get_label("xai-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"
    assert get_label("grok-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"


def test_xai_fallback_without_override_degrades_to_slug(monkeypatch):
    """Guard the exact regression the override fixes.

    On the catalog-unavailable fallback path ``get_provider`` cannot supply
    a display name from the live models.dev catalog. WITHOUT the ``xai``
    entry in ``_LABEL_OVERRIDES`` the label degrades to the raw canonical
    slug ("xai"). This test pins that failure mode so a future refactor that
    drops the override is caught.
    """

    def _unavailable(*args, **kwargs):
        raise RuntimeError("models.dev catalog unavailable")

    monkeypatch.setattr(models_dev, "get_provider_info", _unavailable)

    overrides = dict(providers._LABEL_OVERRIDES)
    overrides.pop("xai", None)
    monkeypatch.setattr(providers, "_LABEL_OVERRIDES", overrides)

    assert get_label("xai") == "xai"


def test_xai_label_override_wins_when_models_dev_unavailable(monkeypatch):
    """The ``_LABEL_OVERRIDES`` entry supplies "xAI" on the fallback path.

    With models.dev unavailable and the override entry present,
    ``get_label`` consults ``_LABEL_OVERRIDES`` first, so the override must
    still win and return "xAI" rather than the lowercase slug.
    """

    def _unavailable(*args, **kwargs):
        raise RuntimeError("models.dev catalog unavailable")

    monkeypatch.setattr(models_dev, "get_provider_info", _unavailable)

    assert get_label("xai") == "xAI"
