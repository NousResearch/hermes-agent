"""Tests for the opencode-go provider's per-model reasoning-effort levels.

The dashboard reasoning picker asks the provider profile which Hermes
reasoning-effort levels a model actually accepts, so the UI shows only
valid choices instead of the full scale. These tests pin the family gates
to the wire behaviour of the OpenCode Go relay:

- DeepSeek thinking models accept the full scale minus ``ultra`` (the relay
  rejects ``ultra`` with "unknown variant").
- Kimi K2 accepts ``none`` + low/medium/high (higher levels collapse to high).
- GLM-5.2 accepts ``none`` + the two enabled levels high/max.
- Everything else (MiMo, Nemotron, plain GLM) gets no reasoning params, so
  the profile reports an empty list (no dial).
"""

from providers import get_provider_profile


def _levels(model: str) -> list[str]:
    profile = get_provider_profile("opencode-go")
    assert profile is not None
    result = profile.reasoning_effort_levels(model)
    assert result is not None
    return result


def test_deepseek_thinking_models_exclude_ultra():
    # Verified live: the relay rejects `ultra` with
    # "unknown variant `ultra`, expected one of none/minimal/low/medium/high/xhigh/max".
    assert _levels("deepseek-v4-flash") == [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert _levels("deepseek-v4-pro") == _levels("deepseek-v4-flash")
    assert _levels("deepseek-reasoner") == _levels("deepseek-v4-flash")


def test_kimi_k2_models_get_none_plus_low_medium_high():
    # Upstream accepts only low/medium/high; xhigh/max/ultra collapse to high.
    assert _levels("kimi-k2.6") == ["none", "low", "medium", "high"]
    assert _levels("kimi-k2.5") == ["none", "low", "medium", "high"]


def test_glm_5_2_gets_two_enabled_levels():
    # GLM-5.2's native knob has exactly two enabled levels: high and max.
    assert _levels("glm-5.2") == ["none", "high", "max"]


def test_models_without_reasoning_params_get_empty_list():
    # MiMo/Nemotron/other GLM get no reasoning parameters on this relay, so
    # the dial is meaningless — empty list tells the UI to hide it.
    assert _levels("mimo-v2.5") == []
    assert _levels("mimo-v2.5-pro") == []
    assert _levels("glm-5.1") == []
    assert _levels("glm-5") == []
    assert _levels("nemotron") == []


def test_unknown_model_is_conservative_empty():
    # Unknown bare IDs should not claim reasoning support.
    assert _levels("some-future-model") == []


def test_opencode_zen_base_profile_returns_none():
    # The plain Zen profile has no family gates — unknown → None so callers
    # fall back to the full option list (never hide a capable model).
    profile = get_provider_profile("opencode-zen")
    assert profile is not None
    assert profile.reasoning_effort_levels("deepseek-v4-flash") is None
