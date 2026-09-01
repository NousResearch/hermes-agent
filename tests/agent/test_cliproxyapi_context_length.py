from unittest.mock import patch


def test_loopback_provider_profile_does_not_alias_all_local_proxies():
    from agent import model_metadata as mm

    assert mm._infer_provider_from_url("http://127.0.0.1:8317/v1") is None
    assert not mm._is_known_provider_base_url("http://127.0.0.1:8317/v1")
    assert mm._is_ambiguous_local_provider_host("localhost")
    assert mm._is_ambiguous_local_provider_host("10.0.0.5")
    assert not mm._is_ambiguous_local_provider_host("api.example.com")


def test_cliproxyapi_uses_live_endpoint_context_before_family_fallback():
    from agent.model_metadata import get_model_context_length

    base_url = "http://127.0.0.1:8317/v1"
    with patch(
        "agent.model_metadata.fetch_endpoint_model_metadata",
        return_value={"gpt-5.5": {"context_length": 272_000}},
    ), patch("agent.model_metadata.get_cached_context_length", return_value=None):
        assert get_model_context_length(
            "gpt-5.5",
            base_url=base_url,
            api_key="test-key",
            provider="cliproxyapi",
        ) == 272_000
