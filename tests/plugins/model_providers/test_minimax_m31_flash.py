"""MiniMax-M3.1-Flash: catalog entry, 1M context, and OpenAI-route reasoning extras."""

import pytest


@pytest.mark.parametrize("provider", ["minimax", "minimax-cn", "minimax-oauth"])
def test_catalog_lists_m31_flash_first(provider):
    from hermes_cli.models_catalog_static import _PROVIDER_MODELS

    assert _PROVIDER_MODELS[provider][:2] == ["MiniMax-M3.1-Flash", "MiniMax-M3"]


def test_context_length_is_1m():
    from agent.model_metadata import get_model_context_length

    assert get_model_context_length("MiniMax-M3.1-Flash", provider="minimax") == 1_000_000


def test_openai_route_never_disables_thinking():
    import providers

    profile = providers.get_provider_profile("minimax")
    kwargs = dict(base_url="https://api.minimax.io/v1")
    extra, _ = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False}, model="MiniMax-M3.1-Flash", **kwargs)
    assert extra == {"reasoning_split": True, "thinking": {"type": "adaptive"}}
    extra, _ = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False}, model="MiniMax-M3", **kwargs)
    assert extra["thinking"] == {"type": "disabled"}
