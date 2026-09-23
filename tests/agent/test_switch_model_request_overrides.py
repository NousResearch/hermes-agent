"""Regression tests for the in-place /model switch (CLI/TUI) carrying a custom
provider's request_overrides (extra_body) — _apply_switched_provider_request_overrides.

Before the fix, agent_runtime_helpers.switch_model() swapped model/provider/
base_url/api_key in place but never touched request_overrides, so a /model
switch to a thinking-enabled custom provider in the TUI/CLI kept the old
provider's extra_body.

The switched-to entry is matched by provider key + base_url + model (the same
condition agent_init._merge_custom_provider_extra_body uses at build time), so a
*different* model selected at the same named endpoint does not inherit an
extra_body configured for another model.
"""

import agent.agent_runtime_helpers as arh


class _Agent:
    pass


# Two entries share the same named endpoint / base_url but pin different models —
# the exact case a name-only match got wrong.
CUSTOM_PROVIDERS = [
    {
        "name": "main-think",
        "base_url": "http://10.0.0.1:8000/v1",
        "model": "think-model",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    {
        "name": "main-plain",
        "base_url": "http://10.0.0.1:8000/v1",
        "model": "plain-model",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
]


def _agent(*, model, base_url, request_overrides, custom_providers=CUSTOM_PROVIDERS):
    a = _Agent()
    # switch_model() sets these on the live agent before calling the helper.
    a.model = model
    a.base_url = base_url
    a.provider = "custom"
    a.request_overrides = request_overrides
    a._custom_providers = custom_providers  # init-time cache the helper reads
    return a


def test_switch_applies_matched_provider_extra_body():
    """Switching to the matching provider+model applies its extra_body.

    A stale Fast pin is re-gated against the new route: this custom host does
    not bill Priority Processing, so ``service_tier`` is dropped even if the
    previous session had /fast on.
    """
    a = _agent(
        model="think-model",
        base_url="http://10.0.0.1:8000/v1",
        request_overrides={"service_tier": "priority", "temperature": 0.2},
    )
    a.service_tier = "priority"
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert a.request_overrides["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "service_tier" not in a.request_overrides
    assert a.request_overrides["temperature"] == 0.2


def test_switch_to_noncustom_clears_stale_extra_body():
    """Switching to a built-in provider clears the previous provider's extra_body.

    ``claude-x`` is not an Anthropic Fast model, so a leftover ``service_tier``
    pin is stripped rather than sent to api.anthropic.com.
    """
    a = _agent(
        model="claude-x",
        base_url="https://api.anthropic.com",
        request_overrides={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "service_tier": "priority",
        },
    )
    a.provider = "anthropic"
    a.service_tier = "priority"
    arh._apply_switched_provider_request_overrides(a, "anthropic")
    assert "extra_body" not in a.request_overrides  # stale extra_body cleared
    assert "service_tier" not in a.request_overrides


def test_switch_keeps_fast_on_qualifying_first_party_route():
    """Static Fast survives a /model switch only when the new route bills for it."""
    a = _agent(
        model="grok-4.6",
        base_url="https://api.x.ai/v1",
        request_overrides={"service_tier": "priority"},
        custom_providers=[],
    )
    a.provider = "xai-oauth"
    a.service_tier = "priority"
    arh._apply_switched_provider_request_overrides(a, "xai-oauth")
    assert a.request_overrides.get("service_tier") == "priority"


def test_switch_strips_fast_when_service_tier_flag_is_unset():
    """An orphan pin with no static Fast flag is dropped, not preserved."""
    a = _agent(
        model="grok-4.6",
        base_url="https://api.x.ai/v1",
        request_overrides={"service_tier": "priority"},
        custom_providers=[],
    )
    a.provider = "xai-oauth"
    arh._apply_switched_provider_request_overrides(a, "xai-oauth")
    assert "service_tier" not in a.request_overrides


def test_switch_strips_anthropic_speed_on_proxy_route():
    a = _agent(
        model="gpt-5.4",
        base_url="https://openrouter.ai/api/v1",
        request_overrides={"speed": "fast"},
        custom_providers=[],
    )
    a.provider = "openrouter"
    a.service_tier = "priority"
    arh._apply_switched_provider_request_overrides(a, "openrouter")
    assert "speed" not in a.request_overrides
    assert "service_tier" not in a.request_overrides


def test_switch_from_none_overrides():
    """A None request_overrides is handled and gets the matched extra_body."""
    a = _agent(
        model="plain-model",
        base_url="http://10.0.0.1:8000/v1",
        request_overrides=None,
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-plain")
    assert a.request_overrides == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_switch_to_different_model_same_endpoint_does_not_inherit():
    """Review regression: selecting a *different* model while naming a custom
    provider must NOT inherit that provider's extra_body when the models differ.

    'main-think' pins 'think-model'. Selecting 'plain-model' under
    custom:main-think must not carry enable_thinking=True — the model-aware
    matcher rejects the mismatch and the stale extra_body is cleared. (A
    name-only match would have wrongly carried it over.)
    """
    a = _agent(
        model="plain-model",  # differs from main-think's pinned 'think-model'
        base_url="http://10.0.0.1:8000/v1",
        request_overrides={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert "extra_body" not in a.request_overrides  # not inherited; stale cleared


def test_switch_endpoint_mismatch_does_not_inherit():
    """A matching provider *name* but a different base_url must not match either
    (endpoint identity is part of the condition)."""
    a = _agent(
        model="think-model",
        base_url="http://10.9.9.9:8000/v1",  # different endpoint than the entry
        request_overrides={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert "extra_body" not in a.request_overrides  # base_url mismatch -> cleared
