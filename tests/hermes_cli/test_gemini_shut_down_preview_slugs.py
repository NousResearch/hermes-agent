"""ai.google.dev "Previous models" marks gemini-3-pro-preview and gemini-3.1-flash-lite-preview
"(Shut down)", so the Google AI Studio picker must not offer them and the models.dev merge must
hide them the same way the already-retired 2.0 slugs are hidden."""

from agent.models_dev import _GOOGLE_HIDDEN_MODELS, _should_hide_from_provider_catalog
from hermes_cli.models import _PROVIDER_MODELS

_SHUT_DOWN = ("gemini-3-pro-preview", "gemini-3.1-flash-lite-preview")


def test_shut_down_preview_slugs_are_hidden_from_the_google_catalogs():
    for model_id in _SHUT_DOWN:
        assert _should_hide_from_provider_catalog("gemini", model_id) is True, model_id
        assert _should_hide_from_provider_catalog("google", model_id) is True, model_id


def test_shut_down_preview_slugs_are_off_the_curated_gemini_picker():
    for model_id in _SHUT_DOWN:
        assert model_id not in _PROVIDER_MODELS["gemini"], model_id


def test_current_preview_models_are_untouched():
    """Gemini 3.1 Pro Preview is still a current Preview model, not a "Previous model"."""
    assert "gemini-3.1-pro-preview" in _PROVIDER_MODELS["gemini"]
    assert "gemini-3.1-pro-preview" not in _GOOGLE_HIDDEN_MODELS
    assert _should_hide_from_provider_catalog("gemini", "gemini-3.1-pro-preview") is False
