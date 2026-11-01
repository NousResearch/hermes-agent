"""Regression tests for the named-custom-provider profile fallback.

A named custom provider defined in config.yaml's ``providers:`` section
resolves at runtime to the bare billing class ``custom`` (see
``_custom_runtime()`` in hermes_cli/runtime_provider_custom.py), so
``get_provider_profile(agent.provider)`` in
``agent/chat_completion_helpers.py`` lands on the bundled Ollama-leaning
``CustomProfile`` instead of the provider's own profile. Any per-provider
quirk that profile carries (e.g. an Ollama-style ``extra_body.think``)
is then silently applied to providers that reject it.

The fix: when the resolved profile is missing or is the generic
``CustomProfile``, retry once with ``agent.requested_provider`` (the
persistent config slug) before falling back.
"""
from types import SimpleNamespace
from unittest.mock import patch

import agent.chat_completion_helpers as C


class CustomProfile:  # name-matched by the helper's generic check
    """Stand-in for providers.CustomProfile (generic billing-class profile)."""

    def __init__(self):
        self.name = "custom"


class _FakeNamedProfile:
    """Stand-in for a provider-specific profile."""

    def __init__(self, name):
        self.name = name


def _agent(provider="custom", requested=""):
    return SimpleNamespace(provider=provider, requested_provider=requested)


def test_named_custom_provider_resolves_its_own_profile():
    """provider='custom' + requested_provider='tokenrhythm' → named profile wins."""
    with patch("providers.get_provider_profile") as gp:
        def _get(name):
            return {"custom": CustomProfile(), "tokenrhythm": _FakeNamedProfile("tokenrhythm")}.get(name)
        gp.side_effect = _get
        profile = C._resolve_agent_profile_for_test(_agent("custom", "tokenrhythm"))
    assert type(profile).__name__ == "_FakeNamedProfile"


def test_bare_custom_without_requested_keeps_generic_profile():
    """No requested_provider → generic CustomProfile is kept (unchanged behaviour)."""
    with patch("providers.get_provider_profile") as gp:
        gp.side_effect = lambda name: CustomProfile() if name == "custom" else None
        profile = C._resolve_agent_profile_for_test(_agent("custom", ""))
    assert type(profile).__name__ == "CustomProfile"


def test_builtin_provider_profile_not_overridden():
    """A registered provider (e.g. 'anthropic') never falls back — even if
    requested_provider carries a stale value."""
    with patch("providers.get_provider_profile") as gp:
        def _get(name):
            return {"anthropic": _FakeNamedProfile("anthropic"),
                    "tokenrhythm": _FakeNamedProfile("tokenrhythm")}.get(name)
        gp.side_effect = _get
        profile = C._resolve_agent_profile_for_test(_agent("anthropic", "tokenrhythm"))
    assert type(profile).__name__ == "_FakeNamedProfile"
    assert profile.name == "anthropic"


def test_requested_equal_to_custom_is_ignored():
    """requested_provider='custom' must not re-resolve the generic profile."""
    with patch("providers.get_provider_profile") as gp:
        calls = []
        def _get(name):
            calls.append(name)
            return CustomProfile() if name == "custom" else None
        gp.side_effect = _get
        profile = C._resolve_agent_profile_for_test(_agent("custom", "custom"))
    assert calls == ["custom"]
    assert type(profile).__name__ == "CustomProfile"
