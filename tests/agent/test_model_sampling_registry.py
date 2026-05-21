"""Tests for local/backend sampling registry."""

from agent.model_sampling_registry import apply_sampling_defaults, resolve_sampling_profile


def test_resolve_sampling_profile_matches_ollama_provider_or_url():
    by_provider = resolve_sampling_profile(provider="ollama", base_url="", model="qwen")
    by_url = resolve_sampling_profile(provider="custom", base_url="http://127.0.0.1:11434/v1", model="qwen")

    assert by_provider is not None
    assert by_provider.name == "ollama_tool_stable"
    assert by_url is not None
    assert by_url.name == "ollama_tool_stable"


def test_apply_sampling_defaults_never_overrides_explicit_request_values():
    profile = resolve_sampling_profile(provider="ollama", base_url="", model="qwen")
    kwargs = {"model": "qwen", "temperature": 0.7, "extra_body": {"top_p": 0.5, "num_ctx": 8192}}

    out = apply_sampling_defaults(kwargs, profile)

    assert out["temperature"] == 0.7
    assert out["extra_body"]["top_p"] == 0.5
    assert out["extra_body"]["num_ctx"] == 8192


def test_apply_sampling_defaults_adds_conservative_values_when_absent():
    profile = resolve_sampling_profile(provider="lmstudio", base_url="", model="local")
    out = apply_sampling_defaults({"model": "local"}, profile)

    assert out["temperature"] == 0.2
    assert out["extra_body"]["top_p"] == 0.9
