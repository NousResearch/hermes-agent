"""Per-model ``context_length`` overrides from custom_providers reach the compressor's deferred resolution (#83324)."""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def test_compressor_threads_custom_providers_into_context_length_resolution():
    providers = [{"name": "p1", "models": {"m": {"context_length": 99999}}}]
    captured = {}

    def fake_resolve(model, **kwargs):
        captured.update(kwargs)
        return 99999

    with patch("agent.context_compressor.get_model_context_length", side_effect=fake_resolve):
        comp = ContextCompressor(model="m", base_url="https://x.example.com/v1", api_key="k", provider="custom",
                                 custom_providers=providers)
        assert comp.context_length == 99999

    assert captured["custom_providers"] == providers
