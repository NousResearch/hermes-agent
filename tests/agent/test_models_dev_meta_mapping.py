"""Meta Model API maps to the models.dev 'meta' provider id (context/pricing)."""

from agent.models_dev import PROVIDER_TO_MODELS_DEV


def test_meta_ai_maps_to_meta():
    assert PROVIDER_TO_MODELS_DEV.get("meta-ai") == "meta"
    assert PROVIDER_TO_MODELS_DEV.get("meta") == "meta"
