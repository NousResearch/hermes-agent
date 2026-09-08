"""Regression coverage: ``model_overrides.<provider>.<alias>`` can declare ``canonical_model`` and
``supports_reasoning`` so a custom-endpoint alias gets reasoning extra_body and model-specific effort clamping.
Aliases without an override keep the current fail-closed behaviour."""
from types import SimpleNamespace

import pytest

from agent import models_dev
from agent.reasoning_effort import is_astra_model, canonical_model_id
from agent.reasoning_params import ReasoningParamsMixin
from agent.transports.chat_completions import ChatCompletionsTransport

BASE = "https://router.example.com/v1"


@pytest.fixture
def overrides(monkeypatch):
    def _set(section):
        monkeypatch.setattr(models_dev, "_load_model_overrides", lambda: {"my-router": section})
    return _set


def _agent(model):
    return SimpleNamespace(model=model, provider="my-router", _base_url_lower=BASE, _is_openrouter_url=lambda: False)


def test_canonical_model_override_resolves_alias(overrides):
    overrides({"astra-alias": {"canonical_model": "gpt-6-astra"}})
    assert canonical_model_id("astra-alias", "my-router") == "gpt-6-astra"
    assert is_astra_model("astra-alias", provider="my-router")
    assert not is_astra_model("astra-alias")  # no provider → no override lookup


def test_unregistered_alias_stays_silent(overrides):
    overrides({})
    assert not is_astra_model("astra-alias", provider="my-router")
    assert not ReasoningParamsMixin._supports_reasoning_extra_body(_agent("sol-alias"))
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="sol-alias", messages=[{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": True, "effort": "xhigh"}, supports_reasoning=False,
    )
    assert "reasoning" not in kwargs.get("extra_body", {})


def test_supports_reasoning_override_enables_extra_body_for_sol_alias(overrides):
    overrides({"sol-alias": {"canonical_model": "gpt-5.6-sol", "supports_reasoning": True}})
    assert ReasoningParamsMixin._supports_reasoning_extra_body(_agent("sol-alias"))
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="sol-alias", messages=[{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": True, "effort": "ultra"}, supports_reasoning=True, provider="my-router",
    )
    assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "max"}


def test_astra_alias_clamps_disabled_effort_to_low_on_chat_wire(overrides):
    overrides({"astra-alias": {"canonical_model": "gpt-6-astra", "supports_reasoning": True}})
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="astra-alias", messages=[{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": False, "effort": "none"}, supports_reasoning=True, provider="my-router",
    )
    assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "low"}
    assert "prompt_cache_options" not in kwargs
