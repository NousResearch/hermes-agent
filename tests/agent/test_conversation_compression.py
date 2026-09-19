"""Transport-aware auxiliary feasibility uses the selected route, not the main route."""
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import conversation_compression as compression
from agent import model_metadata as metadata


@pytest.mark.parametrize("via_fallback", [False, True])
@pytest.mark.parametrize("api_mode", ["codex_responses", "chat_completions"])
def test_auxiliary_feasibility_uses_selected_transport(monkeypatch, via_fallback, api_mode):
    model = "gpt-6-astra"
    route = {
        "provider": "custom", "model": model, "base_url": "http://127.0.0.1:8317/v1",
        "api_key": "test-key", "api_mode": api_mode,
    }
    # Keep routing and transport adapters real; only replace SDK construction and probes.
    monkeypatch.setattr(aux, "_create_openai_client", lambda **kw: SimpleNamespace(
        base_url=kw["base_url"], api_key=kw["api_key"],
    ))
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda task: (
        ("unavailable-test-provider", model, None, None, "chat_completions") if via_fallback else
        (route["provider"], model, route["base_url"], route["api_key"], api_mode)
    ))
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {"fallback_chain": [route]})
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda *a, **kw: False)
    for probe in ("get_cached_context_length", "_resolve_endpoint_context_length",
                  "_probe_local_context_length", "_query_ollama_api_show"):
        monkeypatch.setattr(metadata, probe, lambda *a, **kw: None)
    main_window = metadata.DEFAULT_CONTEXT_LENGTHS[model]
    agent = SimpleNamespace(
        compression_enabled=True, model="main-model", provider="custom",
        base_url="https://main.example/v1", api_mode=(
            "chat_completions" if api_mode == "codex_responses" else "codex_responses"
        ),
        _current_main_runtime=lambda: None, _custom_providers=[],
        _aux_compression_context_length_config=None, _emit_diagnostic_status=lambda message: None,
        context_compressor=SimpleNamespace(context_length=main_window, threshold_tokens=main_window),
    )

    compression.check_compression_model_feasibility(agent)

    expected = (metadata._CODEX_OAUTH_CONTEXT_FALLBACK if api_mode == "codex_responses"
                else metadata.DEFAULT_CONTEXT_LENGTHS)[model]
    assert agent.context_compressor.threshold_tokens == expected
