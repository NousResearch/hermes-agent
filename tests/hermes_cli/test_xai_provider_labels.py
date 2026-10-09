"""Regression tests for xAI provider label disambiguation."""

from providers import get_provider_label


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label():
    """The model picker must distinguish xAI API-key and OAuth providers."""
    assert get_provider_label("xai-oauth") != get_provider_label("xai")
    assert get_provider_label("grok-oauth") == get_provider_label("xai-oauth")


