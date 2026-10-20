"""Regression tests for xAI provider label disambiguation."""

from hermes_cli.providers import get_label


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label():
    """The model picker must distinguish xAI API-key and OAuth providers."""
    assert get_label("xai-oauth") != get_label("xai")
    assert get_label("grok-oauth") == get_label("xai-oauth")


