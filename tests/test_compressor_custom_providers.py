"""Regression tests for custom_providers context-length threading (#83324)."""

from agent.context_compressor import ContextCompressor
from agent.model_metadata import get_model_context_length


def test_custom_providers_override_wins_over_catalog(monkeypatch, tmp_path):
    """A per-model context_length in custom_providers must beat the hardcoded
    catalog default for the same model (the #15779 / #83324 gap)."""
    providers = [
        {
            "name": "myproxy",
            "base_url": "https://proxy.example.com/v1",
            "api_key": "sk-test",
            "models": [
                {"model": "kimi-k2", "context_length": 262144},
            ],
        }
    ]
    ctx = get_model_context_length(
        "kimi-k2",
        base_url="https://proxy.example.com/v1",
        api_key="sk-test",
        provider="custom",
        custom_providers=providers,
    )
    assert ctx == 262144


def test_compressor_threads_custom_providers(tmp_path, monkeypatch):
    """ContextCompressor must pass its custom_providers snapshot into
    _resolve_context_length so per-model overrides apply."""
    captured = {}

    def fake_resolve(self):
        from agent import model_metadata

        captured["custom_providers"] = self.custom_providers
        self._resolved_context_length = model_metadata.get_model_context_length(
            self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            config_context_length=self._config_context_length,
            provider=self.provider,
            custom_providers=self.custom_providers,
        )
        return self._resolved_context_length

    monkeypatch.setattr(ContextCompressor, "_resolve_context_length", fake_resolve)

    providers = [{"name": "p1"}]
    comp = ContextCompressor(
        model="m",
        base_url="https://x.example.com/v1",
        api_key="k",
        provider="custom",
        custom_providers=providers,
    )
    # Caller mutates the list after construction — compressor view must not change.
    providers.append({"name": "p2"})
    comp.context_length
    assert captured["custom_providers"] == [{"name": "p1"}]
