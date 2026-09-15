"""Contract tests for the native Google Gemini provider profile."""

from __future__ import annotations

import pytest


@pytest.fixture
def gemini_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("gemini")
    assert profile is not None, "gemini provider profile must be registered"
    return profile


def test_native_gemini_auxiliary_default_is_in_curated_catalog(gemini_profile):
    """The profile's default_aux_model must stay in lockstep with the curated
    model picker catalog — whatever model the default points at has to be
    one the picker can actually offer. Deliberately durable against future
    model-generation bumps: it does not pin either side to a frozen
    model-name string, only to the invariant that they never drift apart.
    """
    from hermes_cli.models import _PROVIDER_MODELS

    assert gemini_profile.default_aux_model in _PROVIDER_MODELS["gemini"]
