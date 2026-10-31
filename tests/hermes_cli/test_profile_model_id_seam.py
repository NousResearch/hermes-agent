"""``ProviderProfile.normalize_model_id``: a profile that owns its id space repairs ids before the generic rules."""

from __future__ import annotations

import pytest

import providers
from providers.base import ProviderProfile

NAME = "l4-fake-idspace"


class _IdSpaceProfile(ProviderProfile):
    def normalize_model_id(self, model, *, api_key=""):
        return {"fancy-4-5": "fancy-4.5"}.get(model.split("/", 1)[-1], model)


@pytest.fixture
def id_profile(request):
    providers.list_providers()
    providers.register_provider(_IdSpaceProfile(name=NAME, base_url="https://api.idspace.test/v1"))

    def _cleanup():
        providers._REGISTRY.pop(NAME, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)


def test_profile_normalizer_wins_over_generic_rules(id_profile):
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("fancy-4-5", NAME) == "fancy-4.5"
    assert normalize_model_for_provider("vendor/fancy-4-5", NAME) == "fancy-4.5"


def test_profiles_without_the_hook_keep_generic_normalization():
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("openai/gpt-5.4", "openai-codex") == "gpt-5.4"
    assert normalize_model_for_provider("claude-sonnet-4.6", "anthropic") == "claude-sonnet-4-6"
