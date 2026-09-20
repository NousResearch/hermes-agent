"""Muse Spark hosts map to the right models.dev provider id (context/pricing)."""

from agent.models_dev import PROVIDER_TO_MODELS_DEV


def test_meta_ai_maps_to_meta():
    assert PROVIDER_TO_MODELS_DEV.get("meta-ai") == "meta"
    assert PROVIDER_TO_MODELS_DEV.get("meta") == "meta"


def test_opencode_free_is_no_longer_mapped():
    # The keyless OpenCode free tier was removed (OpenCode 403s anonymous access outside its
    # own client), so `opencode-free` must not claim a models.dev catalog anymore.
    assert "opencode-free" not in PROVIDER_TO_MODELS_DEV
