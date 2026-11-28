"""Kimi auxiliary requests preserve effort and omit temperature, including bare K3 ids (#142365)."""

import pytest

from agent.auxiliary_client import _build_call_kwargs


@pytest.mark.parametrize(
    "model,omit_temperature",
    [
        ("k3", True),
        ("k3-256k", True),
        ("k3.1-pro", True),
        ("kimi-coding/k3", True),
        ("kimi-k3-cot", True),
        ("kimi-for-coding", True),
        ("moonshotai/kimi-k2.6", True),
        ("anthropic/claude-sonnet-4-6", False),
        ("deepseek-chat", False),
        ("mk3000", False),
    ],
)
def test_auxiliary_temperature_for_kimi_and_other_models(model, omit_temperature):
    kwargs = _build_call_kwargs(
        provider="openrouter",
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.3,
    )
    if omit_temperature:
        assert "temperature" not in kwargs
    else:
        assert kwargs["temperature"] == 0.3


@pytest.mark.parametrize("model", ["k3", "kimi-k2-turbo-preview"])
def test_kimi_reasoning_uses_requested_top_level_effort(model):
    kwargs = _build_call_kwargs(
        "kimi-coding",
        model,
        [{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": True, "effort": "medium"},
        base_url="https://api.moonshot.ai/v1",
    )
    assert kwargs["reasoning_effort"] == "medium"
    assert "reasoning" not in kwargs.get("extra_body", {})
    assert "thinking" not in kwargs.get("extra_body", {})
