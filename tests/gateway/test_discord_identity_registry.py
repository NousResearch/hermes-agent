"""Slice 1.1 tests for the Discord v2 identity registry service."""

from __future__ import annotations

import json

import pytest

from gateway.config import DiscordNativeMultibotConfig
from gateway.discord_identity_registry import (
    DiscordIdentityRegistry,
    DiscordIdentityRegistryError,
    SecretResolutionError,
)
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store


RESOLVED_TOKEN = "resolved-discord-token-value-should-never-leak"


def _identity(**overrides):
    data = {
        "agent_id": "bohumil",
        "hermes_profile": "default",
        "discord_application_id": "111111111111111111",
        "discord_bot_user_id": "222222222222222222",
        "token_secret_ref": "secret://hermes/discord/bohumil-token",
        "capabilities": ["intake", "reply"],
        "allowed_scopes": {"guild_ids": ["333333333333333333"]},
        "enabled": True,
    }
    data.update(overrides)
    return data


def _karel_identity(**overrides):
    data = _identity(
        agent_id="karel",
        discord_application_id="444444444444444444",
        discord_bot_user_id="555555555555555555",
        token_secret_ref="secret://hermes/discord/karel-token",
        capabilities=["consult"],
    )
    data.update(overrides)
    return data


def _config(*, identities=None, **overrides) -> DiscordNativeMultibotConfig:
    data = {
        "enabled": True,
        "mode": "active",
        "guild_allowlist": ["333333333333333333"],
        "default_intake_agent_id": "bohumil",
        "identities": [_identity()] if identities is None else identities,
    }
    data.update(overrides)
    return DiscordNativeMultibotConfig.from_dict(data)


class FakeSecretResolver:
    def __init__(self, token: str = RESOLVED_TOKEN):
        self.token = token
        self.calls: list[str] = []

    def resolve(self, ref: str) -> str:
        self.calls.append(ref)
        return self.token


class LeakyFailingResolver:
    def resolve(self, ref: str) -> str:
        raise RuntimeError(f"backend failure included {RESOLVED_TOKEN}")


def _store(tmp_path) -> DiscordProtocolV2Store:
    return DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda config: setattr(config.identities[1], "agent_id", "bohumil"),
            "unique agent_id",
        ),
        (
            lambda config: setattr(
                config.identities[1], "discord_application_id", "111111111111111111"
            ),
            "unique discord_application_id",
        ),
        (
            lambda config: setattr(
                config.identities[1], "discord_bot_user_id", "222222222222222222"
            ),
            "unique discord_bot_user_id",
        ),
        (
            lambda config: setattr(
                config.identities[1],
                "token_secret_ref",
                "secret://hermes/discord/bohumil-token",
            ),
            "unique token_secret_ref",
        ),
    ],
)
def test_duplicate_registry_ids_fail_closed(mutate, message):
    config = _config(identities=[_identity(), _karel_identity()])
    mutate(config)

    with pytest.raises(DiscordIdentityRegistryError, match=message):
        DiscordIdentityRegistry.load(config, store=None, secret_resolver=FakeSecretResolver())


def test_duplicate_validation_does_not_partially_persist(tmp_path):
    config = _config(
        identities=[
            _identity(),
            _karel_identity(discord_bot_user_id="222222222222222222"),
        ]
    )

    with _store(tmp_path) as store:
        with pytest.raises(DiscordIdentityRegistryError, match="unique discord_bot_user_id"):
            DiscordIdentityRegistry.load(config, store, FakeSecretResolver())

        assert store.get_identity("bohumil") is None
        assert store.get_identity("karel") is None


def test_duplicate_token_secret_refs_can_be_explicitly_allowed_for_diagnostics():
    config = _config(
        identities=[
            _identity(),
            _karel_identity(token_secret_ref="secret://hermes/discord/bohumil-token"),
        ]
    )

    registry = DiscordIdentityRegistry.load(
        config,
        store=None,
        secret_resolver=FakeSecretResolver(),
        allow_duplicate_token_secret_refs=True,
    )

    assert registry.active_agent_ids == ("bohumil", "karel")


def test_plaintext_token_like_secret_ref_fails_validation():
    config = _config()
    config.identities[0].token_secret_ref = "Bot plaintext-discord-token-value"

    with pytest.raises(DiscordIdentityRegistryError, match="token_secret_ref"):
        DiscordIdentityRegistry.load(config, store=None, secret_resolver=FakeSecretResolver())


def test_load_persists_only_safe_metadata_and_does_not_resolve_token(tmp_path):
    resolver = FakeSecretResolver()
    with _store(tmp_path) as store:
        registry = DiscordIdentityRegistry.load(_config(), store, resolver)
        row = store.get_identity("bohumil")

    assert resolver.calls == []
    assert registry.active_agent_ids == ("bohumil",)
    assert row is not None
    assert row["agent_id"] == "bohumil"
    assert row["token_secret_ref"] == "secret://hermes/discord/bohumil-token"
    assert row["enabled"] == 1
    assert RESOLVED_TOKEN not in json.dumps(dict(row), sort_keys=True)


def test_resolve_token_returns_plaintext_only_from_runtime_resolver_and_redacts_logs(
    tmp_path, caplog
):
    resolver = FakeSecretResolver()
    caplog.set_level("DEBUG")
    with _store(tmp_path) as store:
        registry = DiscordIdentityRegistry.load(_config(), store, resolver)
        token = registry.resolve_token("bohumil")
        row = store.get_identity("bohumil")

    rendered_registry = repr(registry)
    rendered_identity = repr(registry.identities["bohumil"])
    rendered_snapshot = json.dumps(registry.redacted_snapshot(), sort_keys=True)

    assert token == RESOLVED_TOKEN
    assert resolver.calls == ["secret://hermes/discord/bohumil-token"]
    assert row is not None
    assert RESOLVED_TOKEN not in json.dumps(dict(row), sort_keys=True)
    assert RESOLVED_TOKEN not in rendered_registry
    assert RESOLVED_TOKEN not in rendered_identity
    assert RESOLVED_TOKEN not in rendered_snapshot
    assert RESOLVED_TOKEN not in caplog.text
    assert "secret://<redacted>" in rendered_snapshot
    assert "secret://hermes/discord/bohumil-token" not in rendered_snapshot


def test_secret_resolver_exceptions_are_redacted(caplog):
    registry = DiscordIdentityRegistry.load(
        _config(), store=None, secret_resolver=LeakyFailingResolver()
    )
    caplog.set_level("DEBUG")

    with pytest.raises(SecretResolutionError) as excinfo:
        registry.resolve_token("bohumil")

    assert RESOLVED_TOKEN not in str(excinfo.value)
    assert excinfo.value.__context__ is None
    assert RESOLVED_TOKEN not in caplog.text
    assert "secret://<redacted>" in str(excinfo.value)


def test_disabled_identities_are_loaded_but_excluded_from_active_lookup(tmp_path):
    config = _config(identities=[_identity(), _karel_identity(enabled=False)])

    with _store(tmp_path) as store:
        registry = DiscordIdentityRegistry.load(config, store, FakeSecretResolver())
        disabled_row = store.get_identity("karel")

    snapshot = registry.redacted_snapshot()
    assert registry.active_agent_ids == ("bohumil",)
    assert registry.get_identity("karel") is not None
    assert registry.get_identity("karel", include_disabled=False) is None
    assert any(identity["agent_id"] == "karel" for identity in snapshot["identities"])
    assert disabled_row is not None
    assert disabled_row["enabled"] == 0
    with pytest.raises(KeyError):
        registry.resolve_token("karel")


def test_off_config_with_planning_identities_is_loaded_but_inert(tmp_path):
    config = _config(enabled=False, mode="off", guild_allowlist=[])

    with _store(tmp_path) as store:
        registry = DiscordIdentityRegistry.load(config, store, FakeSecretResolver())
        row = store.get_identity("bohumil")

    assert registry.enabled is False
    assert registry.active_agent_ids == ()
    assert registry.get_identity("bohumil") is not None
    assert row is not None
    assert row["enabled"] == 0
    with pytest.raises(KeyError):
        registry.resolve_token("bohumil")


def test_absent_off_config_loads_empty_registry():
    registry = DiscordIdentityRegistry.load(
        DiscordNativeMultibotConfig.from_dict({}),
        store=None,
        secret_resolver=FakeSecretResolver(),
    )

    assert registry.enabled is False
    assert registry.identities == {}
    assert registry.active_agent_ids == ()
    assert registry.redacted_snapshot()["identities"] == []
