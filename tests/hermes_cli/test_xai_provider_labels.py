"""Regression tests for xAI provider label disambiguation."""

from hermes_cli.models import provider_label
from hermes_cli.providers import get_label


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label():
    """The model picker must distinguish xAI API-key and OAuth providers."""
    assert get_label("xai") == "xAI"
    assert get_label("xai-oauth") == "xAI Grok OAuth (SuperGrok Subscription)"
    assert get_label("grok-oauth") == "xAI Grok OAuth (SuperGrok Subscription)"


def test_xai_oauth_provider_labels_match_canonical_model_labels():
    """Provider helpers should agree on the OAuth display label."""
    assert get_label("xai-oauth") == provider_label("xai-oauth")
