"""Context resolution must not re-pay a network probe for a window known locally.

Two independent guards, both about session-start latency:

- a model id the local catalog (models.dev cache / hardcoded table) knows answers before any
  probe, except on a detected local inference server whose window is runtime-specific;
- a failed probe's fallback is memoised for a day, so the next session.create reads it locally
  instead of running the 30 s waterfall again.
"""

from unittest.mock import patch

from agent import model_metadata as mm

_BASE_URL = "http://127.0.0.1:3100/zen/go/v1"


def _expected_catalog_ctx(model: str) -> int:
    hit = mm._longest_key_match(mm.DEFAULT_CONTEXT_LENGTHS, model.lower())
    return hit[1] if hit else mm.DEFAULT_FALLBACK_CONTEXT


def test_known_model_answers_from_the_local_catalog_without_probing():
    model = "zzz-known-only-in-the-table"
    with (
        patch.object(mm, "_answers_as_local_inference_server", return_value=False),
        patch.object(mm, "_local_catalog_context_length", return_value=1_000_000),
        patch.object(mm, "_resolve_endpoint_context_length") as probe,
        patch.object(mm, "_query_ollama_api_show") as ollama_show,
    ):
        resolved = mm.get_model_context_length(model, base_url=_BASE_URL)

    assert resolved == 1_000_000
    probe.assert_not_called()
    ollama_show.assert_not_called()


def test_local_inference_server_is_probed_before_the_catalog():
    model = "zzz-known-only-in-the-table"
    with (
        patch.object(mm, "_answers_as_local_inference_server", return_value=True),
        patch.object(mm, "_local_catalog_context_length", return_value=1_000_000),
        patch.object(mm, "_resolve_endpoint_context_length", return_value=262_144) as probe,
    ):
        resolved = mm.get_model_context_length(model, base_url=_BASE_URL)

    assert resolved == 262_144
    probe.assert_called_once()


def test_local_catalog_never_uses_a_fill_gap_default():
    with patch("agent.models_dev.lookup_models_dev_context", return_value=None) as lookup:
        assert mm._local_catalog_context_length("zzz-absent-model", "opencode-go") is None

    lookup.assert_called_once()
    assert lookup.call_args.kwargs["fill_gap_default"] is False
    assert lookup.call_args.kwargs["allow_network"] is False


def test_default_fallback_ttl_is_one_day():
    import os

    os.environ.pop("HERMES_CONTEXT_FALLBACK_TTL_SECONDS", None)
    assert mm._context_fallback_ttl_seconds() == 86_400.0


def test_failed_custom_probe_is_answered_from_the_fallback_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_CONTEXT_FALLBACK_TTL_SECONDS", "3600")
    model = "deepseek-v4.1-flash"
    with (
        patch.object(mm, "_context_fallback_cache_path", return_value=tmp_path / "fallback.json"),
        patch.object(mm, "_answers_as_local_inference_server", return_value=False),
        patch.object(mm, "_local_catalog_context_length", return_value=None),
        patch.object(mm, "get_cached_context_length", return_value=None),
        patch.object(mm, "_resolve_endpoint_context_length", return_value=None) as probe,
        patch.object(mm, "_probe_local_context_length", return_value=None),
        patch.object(mm, "is_local_endpoint", return_value=False),
        patch.object(mm, "_query_ollama_api_show", return_value=None) as ollama_show,
        patch.object(mm, "fetch_model_metadata", return_value={}),
        patch.object(mm, "_warn_context_length_fallback"),
    ):
        first = mm.get_model_context_length(model, base_url=_BASE_URL)
        second = mm.get_model_context_length(model, base_url=_BASE_URL)

    assert first == second == _expected_catalog_ctx(model)
    # The whole waterfall runs once; the second create reads the memo.
    probe.assert_called_once()
    ollama_show.assert_called_once()


def test_expired_fallback_is_reprobed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_CONTEXT_FALLBACK_TTL_SECONDS", "0")
    model = "deepseek-v4.1-flash"
    with (
        patch.object(mm, "_context_fallback_cache_path", return_value=tmp_path / "fallback.json"),
        patch.object(mm, "_answers_as_local_inference_server", return_value=False),
        patch.object(mm, "_local_catalog_context_length", return_value=None),
        patch.object(mm, "get_cached_context_length", return_value=None),
        patch.object(mm, "_resolve_endpoint_context_length", return_value=None) as probe,
        patch.object(mm, "_probe_local_context_length", return_value=None),
        patch.object(mm, "is_local_endpoint", return_value=False),
        patch.object(mm, "_query_ollama_api_show", return_value=None),
        patch.object(mm, "fetch_model_metadata", return_value={}),
        patch.object(mm, "_warn_context_length_fallback"),
    ):
        mm.get_model_context_length(model, base_url=_BASE_URL)
        mm.get_model_context_length(model, base_url=_BASE_URL)

    assert probe.call_count == 2
