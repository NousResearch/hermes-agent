"""Thinking-toggle / reasoning_effort wire invariants shared by the Moonshot- and
DeepSeek-style chat_completions profiles (all route through
``agent.reasoning_effort.thinking_toggle_extras``)."""

import pytest

from agent.reasoning_effort import DEEPSEEK_V4_EFFORTS
from providers import get_provider_profile

REASONING_MATRIX = (
    None,
    {"enabled": False},
    {"enabled": True},
    *({"enabled": True, "effort": e} for e in ("low", "medium", "high", "xhigh", "max", "none")),
)


@pytest.mark.parametrize("reasoning_config", REASONING_MATRIX, ids=str)
def test_thinking_toggle_and_effort_never_both_on_moonshot_wire(reasoning_config):
    for provider, model in (("kimi-coding", "kimi-k3"), ("opencode-go", "kimi-k2.6"), ("opencode-go", "deepseek-v4-pro")):
        extra_body, top_level = get_provider_profile(provider).build_api_kwargs_extras(
            reasoning_config=reasoning_config, model=model
        )
        assert not ("thinking" in extra_body and "reasoning_effort" in top_level), (provider, model, reasoning_config)

    # DeepSeek's own API wants the toggle on every request (omitting it defaults thinking on
    # and then demands reasoning_content echoes); effort rides alongside only when supported.
    extra_body, top_level = get_provider_profile("deepseek").build_api_kwargs_extras(
        reasoning_config=reasoning_config, model="deepseek-v4-pro"
    )
    assert extra_body["thinking"]["type"] in ("enabled", "disabled")
    assert top_level.get("reasoning_effort", DEEPSEEK_V4_EFFORTS[0]) in DEEPSEEK_V4_EFFORTS
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        assert (extra_body, top_level) == ({"thinking": {"type": "disabled"}}, {})


@pytest.mark.parametrize("reasoning_config", REASONING_MATRIX, ids=str)
def test_ox_alpha_translation_identical_on_zen_and_free(reasoning_config):
    zen = get_provider_profile("opencode-zen").build_api_kwargs_extras(
        reasoning_config=reasoning_config, model="x-preview-f-free"
    )
    free = get_provider_profile("opencode-free").build_api_kwargs_extras(
        reasoning_config=reasoning_config, model="x-preview-f-free"
    )
    assert zen == free
