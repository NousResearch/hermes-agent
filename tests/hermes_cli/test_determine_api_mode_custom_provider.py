"""``determine_api_mode`` honors a user-config provider's declared wire protocol.

A custom endpoint configured through ``hermes model`` stores its protocol in
``api_mode``. ``determine_api_mode`` used to ignore that and return
``chat_completions`` for any provider outside the built-in catalog, so an
anthropic_messages gateway rejected every request with HTTP 500
("... does not support request path /v1/chat/completions ..."). Regression for #126308.

These drive the real ``_save_custom_provider`` writer rather than hand-writing config,
because the wizard's output shape (``api_mode``) is the whole bug surface.
"""

from __future__ import annotations

import pytest

from hermes_cli.providers import determine_api_mode


@pytest.fixture
def saved_provider():
    """Persist custom endpoints through the real writer; yield (slug, base_url) pairs."""
    from hermes_cli.main_provider_setup import _save_custom_provider

    entries = [
        ("WiredGateway", "https://gateway.example.test/v1", "anthropic_messages"),
        ("CodexRelay", "https://relay.example.test/v1", "codex_responses"),
    ]
    for name, base_url, api_mode in entries:
        _save_custom_provider(base_url=base_url, api_key="sk-test", name=name, api_mode=api_mode)
    return [(f"custom:{name.lower()}", base_url) for name, base_url, _ in entries]


def test_declared_api_mode_selects_the_wire(saved_provider):
    # The wizard's own output must route to the protocol the user picked — not collapse to
    # chat_completions because the name is outside the built-in provider catalog.
    assert determine_api_mode(*saved_provider[0]) == "anthropic_messages"
    assert determine_api_mode(*saved_provider[1]) == "codex_responses"


def test_unknown_and_disabled_entries_fall_through(saved_provider, monkeypatch):
    from hermes_cli import config

    # An unknown name is untouched (no invented provider).
    assert determine_api_mode("custom:does-not-exist", "https://nowhere.test/v1") == "chat_completions"
    # An entry hidden from every other consumer (``enabled: false``, which empties the compat
    # view) must not be resurrected as a routing source here.
    monkeypatch.setattr(config, "get_compatible_custom_providers", lambda *_a, **_k: [])
    assert determine_api_mode(*saved_provider[0]) == "chat_completions"
