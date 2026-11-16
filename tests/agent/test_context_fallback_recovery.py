"""A recovered local inference server must override an offline context fallback.

The fallback is useful when a gateway is down, but not after its runtime
returns with a model-specific configured window.
"""

from unittest.mock import patch

from agent import model_metadata as mm


def test_recovered_local_server_prefers_live_window_over_offline_fallback(tmp_path, monkeypatch):
    # Keep the production one-day TTL so recovery, not expiry, clears the stale result.
    monkeypatch.delenv("HERMES_CONTEXT_FALLBACK_TTL_SECONDS", raising=False)
    model = "unlisted-local-model-2026"
    url = "http://127.0.0.1:31477/v1"
    with (
        patch.object(mm, "_context_fallback_cache_path", return_value=tmp_path / "fallback.json"),
        patch.object(mm, "_answers_as_local_inference_server", side_effect=[False, True]),
        patch.object(mm, "_local_catalog_context_length", return_value=None),
        patch.object(mm, "_resolve_endpoint_context_length", side_effect=[None, 8192]) as endpoint,
        patch.object(mm, "_is_local_server", return_value=False),
        patch.object(mm, "_query_ollama_api_show", return_value=None),
        patch.object(mm, "_warn_context_length_fallback"),
    ):
        offline = mm._resolve_custom_endpoint_context_length(model, url, "", "custom")
        recovered = mm._resolve_custom_endpoint_context_length(model, url, "", "custom")

    assert offline == mm.DEFAULT_FALLBACK_CONTEXT
    assert recovered == 8192
    assert endpoint.call_count == 2
