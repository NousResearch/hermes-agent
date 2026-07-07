from agent.model_metadata import get_model_context_length


CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def test_codex_gpt55_uses_curated_400k_context_even_when_live_probe_reports_272k(monkeypatch):
    """Codex /models can report a 272K compatibility cap for GPT-5.5.

    Hermes uses the curated runtime budget so compression does not start too
    early for Codex OAuth sessions that can operate at 400K.
    """

    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths",
        lambda access_token: {"gpt-5.5": 272_000},
    )

    assert (
        get_model_context_length(
            "gpt-5.5",
            provider="openai-codex",
            base_url=CODEX_BASE_URL,
            api_key="x",
        )
        == 400_000
    )


def test_codex_gpt54_mini_uses_curated_400k_context_for_compression_lane(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths",
        lambda access_token: {"gpt-5.4-mini": 272_000},
    )

    assert (
        get_model_context_length(
            "gpt-5.4-mini",
            provider="openai-codex",
            base_url=CODEX_BASE_URL,
            api_key="x",
        )
        == 400_000
    )


def test_codex_non_overridden_models_keep_endpoint_reported_context(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths",
        lambda access_token: {"gpt-5.3-codex": 272_000},
    )

    assert (
        get_model_context_length(
            "gpt-5.3-codex",
            provider="openai-codex",
            base_url=CODEX_BASE_URL,
            api_key="x",
        )
        == 272_000
    )


def test_codex_non_overridden_models_invalidate_stale_400k_cache(monkeypatch):
    invalidated = []

    monkeypatch.setattr(
        "agent.model_metadata.get_cached_context_length",
        lambda model, base_url: 400_000,
    )
    monkeypatch.setattr(
        "agent.model_metadata._invalidate_cached_context_length",
        lambda model, base_url: invalidated.append((model, base_url)),
    )
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths",
        lambda access_token: {"gpt-5.3-codex": 272_000},
    )

    assert (
        get_model_context_length(
            "gpt-5.3-codex",
            provider="openai-codex",
            base_url=CODEX_BASE_URL,
            api_key="x",
        )
        == 272_000
    )
    assert invalidated == [("gpt-5.3-codex", CODEX_BASE_URL)]
