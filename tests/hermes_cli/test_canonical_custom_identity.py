"""``canonical_custom_identity`` must return the durable config-key identity.

A keyed ``providers:`` entry's identity is its config key, not its display
name — ``custom_provider_slug`` encodes that, and the endpoint- and
model-based recovery sources both honour it. The configured-provider fallback
built its slug from whatever string the caller had, so a display name that
differs from its key healed to ``custom:<display-name>``: a second identity
for the same endpoint that no longer matches what persistence and routing
store.

Both spellings match the entry (``_get_named_custom_provider`` accepts
either), so the test asserts they converge on one identity rather than
asserting any particular spelling is rejected.
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp

PROVIDER_KEY = "my-endpoint"
DISPLAY_NAME = "My Endpoint Display"
BASE_URL = "https://example.invalid/v1"
MODEL = "cool-model-1"

CANONICAL = f"custom:{PROVIDER_KEY}"


@pytest.fixture
def keyed_provider_config(monkeypatch):
    """A ``providers:`` entry whose display name differs from its config key."""
    config = {
        "providers": {
            PROVIDER_KEY: {
                "name": DISPLAY_NAME,
                "api": BASE_URL,
                "api_key": "sk-test",
                "default_model": MODEL,
                "models": [MODEL],
            }
        }
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    return config


def test_display_name_heals_to_the_config_key_identity(keyed_provider_config):
    """The regression: the display-name spelling must not mint a second identity."""
    assert rp.canonical_custom_identity(config_provider=DISPLAY_NAME) == CANONICAL


def test_config_model_provider_display_name_heals_too(keyed_provider_config, monkeypatch):
    """Same path reached through ``config.model.provider`` rather than an argument."""
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": DISPLAY_NAME})
    assert rp.canonical_custom_identity() == CANONICAL


def test_config_key_spelling_still_resolves(keyed_provider_config):
    """The spelling that already worked keeps working."""
    assert rp.canonical_custom_identity(config_provider=PROVIDER_KEY) == CANONICAL


def test_all_recovery_sources_agree_on_one_identity(keyed_provider_config):
    """Endpoint, model and configured-provider recovery must not disagree.

    Three sources feeding the same session-identity slot is only safe while
    they agree; a divergent one silently splits an endpoint in two.
    """
    by_url = rp.canonical_custom_identity(base_url=BASE_URL)
    by_model = rp.canonical_custom_identity(model=MODEL)
    by_config = rp.canonical_custom_identity(config_provider=DISPLAY_NAME)

    assert {by_url, by_model, by_config} == {CANONICAL}


def test_unconfigured_candidate_still_returns_none(keyed_provider_config):
    """Fail-closed contract: never invent an identity resolution can't honour."""
    assert rp.canonical_custom_identity(config_provider="not-a-configured-entry") is None


def test_legacy_unkeyed_entry_keeps_its_name_identity(monkeypatch):
    """``custom_providers:`` entries have no key, so the name stays the identity."""
    config = {
        "custom_providers": [
            {
                "name": "Legacy Endpoint",
                "base_url": "https://legacy.invalid/v1",
                "api_key": "sk-legacy",
                "models": ["legacy-model"],
            }
        ]
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})

    assert rp.canonical_custom_identity(config_provider="Legacy Endpoint") == "custom:legacy-endpoint"


# --- Issue #81789: two providers sharing one base_url, different api_keys ----
#
# The base_url reverse-lookup cannot disambiguate same-URL entries (it always
# returns the FIRST URL owner), so recovery must prefer the explicitly
# requested provider identity over the URL.

SHARED_URL = "https://relay.example.invalid/v1"

SHARED_URL_CONFIG = {
    "custom_providers": [
        {
            "name": "slomerex-grok",
            "base_url": SHARED_URL,
            "api_key": "keyA",
            "api_mode": "chat_completions",
        },
        {
            "name": "slomerex-alt",
            "base_url": SHARED_URL,
            "api_key": "keyB",
            "api_mode": "chat_completions",
        },
    ]
}


@pytest.fixture
def shared_url_config(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: SHARED_URL_CONFIG)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: SHARED_URL_CONFIG)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    return SHARED_URL_CONFIG


def test_requested_identity_disambiguates_shared_base_url(shared_url_config):
    """The explicitly requested provider must win over the first URL owner."""
    assert (
        rp.canonical_custom_identity(
            requested_provider="slomerex-alt", base_url=SHARED_URL
        )
        == "custom:slomerex-alt"
    )
    assert (
        rp.canonical_custom_identity(
            requested_provider="slomerex-grok", base_url=SHARED_URL
        )
        == "custom:slomerex-grok"
    )


def test_requested_identity_wins_even_when_url_matches_first_entry(shared_url_config):
    """The base_url alone would heal to the FIRST owner — the name must win."""
    assert (
        rp.canonical_custom_identity(
            requested_provider="custom:slomerex-alt", base_url=SHARED_URL
        )
        == "custom:slomerex-alt"
    )


def test_bare_url_still_heals_to_first_url_owner(shared_url_config):
    """Without a name the URL reverse-lookup keeps its legacy first-match
    behavior (callers that only have a base_url cannot do better)."""
    assert rp.canonical_custom_identity(base_url=SHARED_URL) == "custom:slomerex-grok"


def test_unconfigured_requested_identity_falls_through_to_url(shared_url_config):
    """A requested name that matches no entry must not mint a fake identity."""
    assert (
        rp.canonical_custom_identity(
            requested_provider="not-a-real-entry", base_url=SHARED_URL
        )
        == "custom:slomerex-grok"
    )


def test_requested_identity_heals_keyed_providers_dict(monkeypatch):
    """Keyed ``providers:`` entries keep their config-key slug, not the
    display name, when recovered by the requested identity."""
    config = {
        "providers": {
            "slomerex-alt": {
                "name": "Slomerex Alt Display",
                "api": SHARED_URL,
                "api_key": "keyB",
            }
        }
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})

    assert (
        rp.canonical_custom_identity(
            requested_provider="Slomerex Alt Display", base_url=SHARED_URL
        )
        == "custom:slomerex-alt"
    )
