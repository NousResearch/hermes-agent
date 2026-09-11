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


class TestIsRoutableProvider:
    """``is_routable_provider`` gates session-resume fallback: a persisted
    provider name that no longer resolves (renamed/removed) must be detected
    so recovery falls back instead of failing agent init with
    "Unknown provider '<name>'".
    """

    def test_empty_auto_and_builtin_are_routable(self, keyed_provider_config):
        assert rp.is_routable_provider(None) is True
        assert rp.is_routable_provider("") is True
        assert rp.is_routable_provider("auto") is True
        assert rp.is_routable_provider("openrouter") is True

    def test_bare_custom_is_not_routable(self, keyed_provider_config):
        # The resolved billing class, not a routable identity — restore
        # paths must heal it (canonical_custom_identity) or fall back.
        assert rp.is_routable_provider("custom") is False

    def test_registered_names_are_routable(self, keyed_provider_config):
        assert rp.is_routable_provider(PROVIDER_KEY) is True
        assert rp.is_routable_provider(CANONICAL) is True

    def test_stale_name_is_not_routable(self, keyed_provider_config):
        # Same endpoint family, but the OLD slug no longer matches any
        # configured entry — the regression this gate exists for.
        assert rp.is_routable_provider("stale-endpoint") is False
        assert rp.is_routable_provider("custom:stale-endpoint") is False

    def test_legacy_unkeyed_name_is_routable(self, monkeypatch):
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

        assert rp.is_routable_provider("legacy-endpoint") is True
        assert rp.is_routable_provider("custom:legacy-endpoint") is True
        assert rp.is_routable_provider("Legacy Endpoint") is True
