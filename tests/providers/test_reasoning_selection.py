"""Providers must opt into reasoning validation before preferences are changed."""

import pytest

from providers import get_provider_profile
from providers.reasoning import resolve_provider_reasoning_config


@pytest.mark.parametrize("provider", ["anthropic", "openrouter", "unregistered-fixture"])
@pytest.mark.parametrize("explicit", [False, True])
def test_non_validating_providers_keep_their_reasoning_config(provider, explicit):
    profile = get_provider_profile(provider)
    assert profile is None or not profile.validate_reasoning_selection
    for effort in ("budget:5000", "budget:-1", "high", "none", "auto"):
        config = {"enabled": effort != "none", "effort": effort}
        assert resolve_provider_reasoning_config(
            provider, "fixture-model", config, explicit=explicit
        ) is config
    assert resolve_provider_reasoning_config(provider, "fixture-model", None, explicit=explicit) is None
