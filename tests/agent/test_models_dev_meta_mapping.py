"""Muse Spark hosts map to the right models.dev provider id (context/pricing)."""

from agent.models_dev import PROVIDER_TO_MODELS_DEV


def test_meta_ai_maps_to_meta():
    assert PROVIDER_TO_MODELS_DEV.get("meta-ai") == "meta"
    assert PROVIDER_TO_MODELS_DEV.get("meta") == "meta"


def test_opencode_free_maps_to_zen_catalog():
    # The free tier is served by the Zen relay, whose models.dev id is "opencode".
    assert PROVIDER_TO_MODELS_DEV.get("opencode-free") == "opencode"
