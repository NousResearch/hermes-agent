from hermes_cli.models import _rank_nvidia_model_ids, provider_model_ids


def test_nvidia_ranker_promotes_agentic_models_and_demotes_utility():
    ranked = _rank_nvidia_model_ids([
        "google/gemma-4-31b-it",
        "nvidia/llama-nemotron-embed-1b-v2",
        "minimaxai/minimax-m3",
        "deepseek-ai/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "nvidia/nemotron-parse",
    ])

    assert ranked.index("minimaxai/minimax-m3") < ranked.index("google/gemma-4-31b-it")
    assert ranked.index("deepseek-ai/deepseek-v4-pro") < ranked.index("google/gemma-4-31b-it")
    assert ranked.index("moonshotai/kimi-k2.6") < ranked.index("google/gemma-4-31b-it")
    assert ranked.index("nvidia/llama-nemotron-embed-1b-v2") > ranked.index("google/gemma-4-31b-it")
    assert ranked.index("nvidia/nemotron-parse") > ranked.index("google/gemma-4-31b-it")


def test_nvidia_provider_model_ids_ranks_live_catalog(monkeypatch):
    live_catalog = [
        "google/gemma-4-31b-it",
        "nvidia/llama-nemotron-embed-1b-v2",
        "deepseek-ai/deepseek-v4-pro",
        "minimaxai/minimax-m3",
        "moonshotai/kimi-k2.6",
    ]

    def fake_fetch_models(self, api_key: str, base_url: str | None = None):
        return list(live_catalog)

    monkeypatch.setattr(
        "providers.base.ProviderProfile.fetch_models",
        fake_fetch_models,
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda provider: {
            "api_key": "nvapi-test",
            "base_url": "https://integrate.api.nvidia.com/v1",
        },
    )

    models = provider_model_ids("nvidia")

    assert models.index("minimaxai/minimax-m3") < models.index("google/gemma-4-31b-it")
    assert models.index("deepseek-ai/deepseek-v4-pro") < models.index("google/gemma-4-31b-it")
    assert models.index("moonshotai/kimi-k2.6") < models.index("google/gemma-4-31b-it")
    assert models.index("nvidia/llama-nemotron-embed-1b-v2") > models.index("google/gemma-4-31b-it")
