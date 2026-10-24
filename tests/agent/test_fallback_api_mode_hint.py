"""Regression tests for `_fallback_api_mode_hint` (#143933).

A `fallback_providers` entry that names a plugin-registered provider inherits
the wire format declared on the registered `ProviderProfile`. Before #143933's
fix the hint only consulted the entry itself, `providers.<name>` config blocks,
and the Anthropic name/URL heuristics — so a provider like `commandcode-anthropic`
(`api_mode="anthropic_messages"` on its profile, behind a plain host the URL
heuristic cannot classify) was downgraded to `chat_completions`, and every
fallback hop failed with an HTTP 400 exactly when the primary was already down.
"""

from __future__ import annotations

from providers.base import ProviderProfile

from agent.chat_completion_helpers import _fallback_api_mode_hint


def _profile(name: str, api_mode: str) -> ProviderProfile:
    return ProviderProfile(name=name, api_mode=api_mode)


def _no_named_custom_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_named_custom_provider",
        lambda name: None,
    )


def test_plugin_provider_profile_declares_the_fallback_wire(monkeypatch):
    """A plugin-registered profile's api_mode resolves the hint (#143933)."""
    import providers as providers_registry

    _no_named_custom_provider(monkeypatch)
    monkeypatch.setattr(
        providers_registry,
        "get_provider_profile",
        lambda name: (
            _profile(name, "anthropic_messages")
            if name == "commandcode-anthropic"
            else None
        ),
    )

    explicit, api_mode = _fallback_api_mode_hint(
        {"provider": "commandcode-anthropic", "model": "claude-sonnet-5-5"},
        "commandcode-anthropic",
        None,
    )

    assert (explicit, api_mode) == (True, "anthropic_messages")


def test_default_chat_completions_profile_keeps_re_detection(monkeypatch):
    """A profile carrying the default wire must not pin the hint explicit —
    `_fallback_api_mode_resolved` re-detection stays available for it."""
    import providers as providers_registry

    _no_named_custom_provider(monkeypatch)
    monkeypatch.setattr(
        providers_registry,
        "get_provider_profile",
        lambda name: _profile(name, "chat_completions"),
    )

    explicit, api_mode = _fallback_api_mode_hint(
        {"provider": "some-openai-relay", "model": "gpt-5"},
        "some-openai-relay",
        "https://relay.example.com/v1",
    )

    assert (explicit, api_mode) == (False, "chat_completions")


def test_entry_api_mode_still_wins_over_the_profile(monkeypatch):
    """An explicit api_mode on the fallback entry overrides the profile."""
    import providers as providers_registry

    _no_named_custom_provider(monkeypatch)
    monkeypatch.setattr(
        providers_registry,
        "get_provider_profile",
        lambda name: _profile(name, "anthropic_messages"),
    )

    explicit, api_mode = _fallback_api_mode_hint(
        {"provider": "dual-wire-relay", "model": "m", "api_mode": "chat_completions"},
        "dual-wire-relay",
        None,
    )

    assert (explicit, api_mode) == (True, "chat_completions")


def test_unregistered_provider_is_unchanged(monkeypatch):
    """No profile and no config block: today's default + re-detection path."""
    import providers as providers_registry

    _no_named_custom_provider(monkeypatch)
    monkeypatch.setattr(providers_registry, "get_provider_profile", lambda name: None)

    explicit, api_mode = _fallback_api_mode_hint(
        {"provider": "custom", "model": "m"},
        "custom",
        None,
    )

    assert (explicit, api_mode) == (False, "chat_completions")
