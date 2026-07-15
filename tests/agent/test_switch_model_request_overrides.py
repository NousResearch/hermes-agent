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
    """Switching to the matching provider+model applies its extra_body and
    preserves non-provider overrides (service_tier/speed from /fast)."""
    a = _agent(
        model="think-model",
        base_url="http://10.0.0.1:8000/v1",
        request_overrides={"service_tier": "priority"},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert a.request_overrides["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert a.request_overrides["service_tier"] == "priority"  # preserved


def test_switch_to_noncustom_clears_stale_extra_body():
    """Switching to a built-in provider clears the previous provider's extra_body."""
    a = _agent(
        model="claude-x",
        base_url="https://api.anthropic.com",
        request_overrides={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "service_tier": "priority",
        },
    )
    arh._apply_switched_provider_request_overrides(a, "anthropic")
    assert "extra_body" not in a.request_overrides  # stale extra_body cleared
    assert a.request_overrides["service_tier"] == "priority"  # preserved


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
