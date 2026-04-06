import pytest
from benchmarks.backends.honcho_adapter import HonchoBenchmarkAdapter


def test_honcho_adapter_requires_base_url():
    with pytest.raises(RuntimeError, match="HONCHO_BASE_URL"):
        HonchoBenchmarkAdapter(base_url="", api_key="test")


def test_honcho_adapter_requires_api_key_for_remote():
    # Non-localhost URL without API key should fail
    with pytest.raises(RuntimeError, match="HONCHO_API_KEY"):
        HonchoBenchmarkAdapter(base_url="http://remote-server:8000", api_key="")
