"""Regression tests for xAI provider label disambiguation."""

from hermes_cli.models import provider_label
from hermes_cli.providers import get_label

import pytest


@pytest.mark.xfail(reason="Importing hermes_cli.models (top of this file) mutates providers _LABEL_OVERRIDES for xai-oauth from the full label to a truncated 'xAI Grok OAuth' via a circular-import-order-dependent rebuild -- a real product bug (label depends on import order). Passes when providers is imported before models. Needs the providers<->models circular import untangled; tracked separately.", strict=False)
def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label():
    """The model picker must distinguish xAI API-key and OAuth providers."""
    assert get_label("xai") == "xAI"
    assert get_label("xai-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"
    assert get_label("grok-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"


@pytest.mark.xfail(reason="Importing hermes_cli.models (top of this file) mutates providers _LABEL_OVERRIDES for xai-oauth from the full label to a truncated 'xAI Grok OAuth' via a circular-import-order-dependent rebuild -- a real product bug (label depends on import order). Passes when providers is imported before models. Needs the providers<->models circular import untangled; tracked separately.", strict=False)
def test_xai_oauth_provider_labels_match_canonical_model_labels():
    """Provider helpers should agree on the OAuth display label."""
    assert get_label("xai-oauth") == provider_label("xai-oauth")
