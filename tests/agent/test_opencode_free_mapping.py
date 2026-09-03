"""opencode-free resolves through the opencode-go catalog; muse-spark is 1M."""

from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
from agent.models_dev import PROVIDER_TO_MODELS_DEV


def test_opencode_free_maps_to_opencode_go():
    assert PROVIDER_TO_MODELS_DEV.get("opencode-free") == "opencode-go"
    assert PROVIDER_TO_MODELS_DEV.get("opencode-go") == "opencode-go"


def test_muse_spark_catalog_is_1m():
    assert DEFAULT_CONTEXT_LENGTHS.get("muse-spark") == 1_048_576
