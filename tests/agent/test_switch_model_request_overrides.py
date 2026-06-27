"""Regression tests for the in-place /model switch (CLI/TUI) carrying a custom
provider's request_overrides (extra_body) — _apply_switched_provider_request_overrides.

Before the fix, agent_runtime_helpers.switch_model() swapped model/provider/
base_url/api_key in place but never touched request_overrides, so a /model
switch to a thinking-enabled custom provider in the TUI/CLI kept the old
provider's extra_body.
"""

import agent.agent_runtime_helpers as arh


class _Agent:
    pass


def test_switch_applies_new_provider_extra_body(monkeypatch):
    a = _Agent()
    a.request_overrides = {"service_tier": "priority"}  # pre-existing /fast override
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda name: {"name": "main-think",
                      "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main-think")
    assert a.request_overrides["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert a.request_overrides["service_tier"] == "priority"  # preserved


def test_switch_to_noncustom_clears_stale_extra_body(monkeypatch):
    a = _Agent()
    a.request_overrides = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "service_tier": "priority",
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider", lambda name: None
    )
    arh._apply_switched_provider_request_overrides(a, "anthropic")
    assert "extra_body" not in a.request_overrides  # stale extra_body cleared
    assert a.request_overrides["service_tier"] == "priority"  # preserved


def test_switch_from_none_overrides(monkeypatch):
    a = _Agent()
    a.request_overrides = None
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda name: {"name": "main", "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
    arh._apply_switched_provider_request_overrides(a, "custom:main")
    assert a.request_overrides == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
