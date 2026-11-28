"""Provider-error cache poisoning regression (NousResearch/hermes-agent#150519)."""

from unittest.mock import patch

from agent import model_metadata as mm


def test_subminimum_gateway_cache_re_resolves_without_touching_sibling(tmp_path, monkeypatch):
    """An output cap mistaken for a window must not brick every later agent init."""
    cache_file = tmp_path / "context_length_cache.yaml"
    monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)

    model, base_url = "agnes-3.0-flash", "https://gateway.example/v1"
    mm.save_context_length(model, base_url, 4_096)  # already poisoned by a 400
    mm.save_context_length("other-model", base_url, 131_072)
    assert mm.get_cached_context_length(model, base_url) == 4_096

    # Stub only the network probe; exercise real public resolution and disk repair.
    with patch.object(mm, "_resolve_custom_endpoint_context_length", return_value=200_000) as probe:
        assert mm.get_model_context_length(model, base_url=base_url, provider="custom") == 200_000
    probe.assert_called_once_with(model, base_url, "", "custom")
    assert mm.get_cached_context_length(model, base_url) is None
    assert mm.get_cached_context_length("other-model", base_url) == 131_072
