"""Slice 1.3 tests for Discord v2 secret resolver/redaction contracts."""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3

import pytest

from gateway.config import DiscordNativeMultibotConfig, GatewayConfig
from gateway.discord_identity_registry import DiscordIdentityRegistry, SecretResolutionError
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.secret_refs import (
    EnvSecretResolver,
    SecretRedactionFilter,
    SecretRefError,
    SensitiveToken,
    StaticSecretResolver,
    is_raw_discord_token_like,
    redact_secret_ref,
    redact_sensitive_data,
    redact_sensitive_text_value,
    validate_secret_ref,
)
from hermes_cli import discord_native

PLAINTEXT_TOKEN = "MTIz.fake.discord-token-value"
SECRET_REF = "secret://hermes/discord/bohumil-token"


def _identity(**overrides):
    data = {
        "agent_id": "bohumil",
        "hermes_profile": "default",
        "discord_application_id": "111111111111111111",
        "discord_bot_user_id": "222222222222222222",
        "token_secret_ref": SECRET_REF,
        "capabilities": ["intake", "reply"],
        "allowed_scopes": {"guild_ids": ["333333333333333333"]},
        "enabled": True,
    }
    data.update(overrides)
    return data


def _config(**overrides) -> DiscordNativeMultibotConfig:
    data = {
        "enabled": True,
        "mode": "active",
        "guild_allowlist": ["333333333333333333"],
        "default_intake_agent_id": "bohumil",
        "identities": [_identity()],
    }
    data.update(overrides)
    return DiscordNativeMultibotConfig.from_dict(data)


def test_sensitive_token_reveal_is_explicit_and_repr_str_are_redacted():
    token = SensitiveToken(PLAINTEXT_TOKEN)

    assert token.reveal() == PLAINTEXT_TOKEN
    assert PLAINTEXT_TOKEN not in str(token)
    assert PLAINTEXT_TOKEN not in repr(token)
    with pytest.raises(TypeError):
        pickle.dumps(token)
    with pytest.raises(TypeError):
        json.dumps({"token": token})


def test_secret_ref_validation_allows_secret_by_default_and_rejects_env_without_policy():
    assert validate_secret_ref(SECRET_REF) == SECRET_REF
    assert redact_secret_ref(SECRET_REF) == "secret://<redacted>"

    with pytest.raises(SecretRefError, match="secret_ref"):
        validate_secret_ref("env://DISCORD_BOT_TOKEN")
    with pytest.raises(SecretRefError, match="secret_ref"):
        validate_secret_ref(PLAINTEXT_TOKEN)


def test_env_resolver_requires_explicit_allow_env_policy(monkeypatch):
    monkeypatch.setenv("DISCORD_TEST_TOKEN", PLAINTEXT_TOKEN)

    locked = EnvSecretResolver()
    with pytest.raises(SecretRefError):
        locked.resolve("env://DISCORD_TEST_TOKEN")

    resolver = EnvSecretResolver(allow_env=True)
    token = resolver.resolve("env://DISCORD_TEST_TOKEN")

    assert isinstance(token, SensitiveToken)
    assert token.reveal() == PLAINTEXT_TOKEN
    assert resolver.calls == ["env://DISCORD_TEST_TOKEN"]


def test_static_fake_resolver_returns_runtime_only_sensitive_token():
    resolver = StaticSecretResolver({SECRET_REF: PLAINTEXT_TOKEN})

    token = resolver.resolve(SECRET_REF)

    assert isinstance(token, SensitiveToken)
    assert token.reveal() == PLAINTEXT_TOKEN
    assert PLAINTEXT_TOKEN not in repr(token)
    assert resolver.calls == [SECRET_REF]


def test_v2_identity_config_rejects_forbidden_raw_token_keys():
    for key in ("token", "bot_token", "discord_token", "DISCORD_BOT_TOKEN"):
        bad_identity = _identity()
        bad_identity[key] = PLAINTEXT_TOKEN
        with pytest.raises(ValueError, match="forbidden credential key"):
            _config(identities=[bad_identity])

    bad_identity = _identity(notes=f"raw credential {PLAINTEXT_TOKEN}")
    with pytest.raises(ValueError, match="raw Discord token-looking value"):
        _config(identities=[bad_identity])


def test_v2_identity_config_rejects_raw_token_looking_ref_and_env_ref():
    assert is_raw_discord_token_like(PLAINTEXT_TOKEN) is True

    for bad_ref in (PLAINTEXT_TOKEN, f"Bot {PLAINTEXT_TOKEN}", "env://DISCORD_TEST_TOKEN"):
        with pytest.raises(ValueError, match="token_secret_ref"):
            _config(identities=[_identity(token_secret_ref=bad_ref)])


def test_registry_unwraps_sensitive_token_without_persisting_or_serializing_plaintext(tmp_path):
    store_path = tmp_path / "discord-v2.sqlite3"
    resolver = StaticSecretResolver({SECRET_REF: PLAINTEXT_TOKEN})

    with DiscordProtocolV2Store(store_path) as store:
        registry = DiscordIdentityRegistry.load(_config(), store, resolver)
        resolved = registry.resolve_token("bohumil")
        row = store.get_identity("bohumil")

    config_json = json.dumps(GatewayConfig.from_dict({"discord_native_multibot": _config().to_dict()}).to_dict())
    snapshot_json = json.dumps(registry.redacted_snapshot(), sort_keys=True)
    raw_db = store_path.read_bytes().decode("latin1", errors="ignore")

    assert resolved == PLAINTEXT_TOKEN
    assert row is not None
    assert row["token_secret_ref"] == SECRET_REF
    assert PLAINTEXT_TOKEN not in json.dumps(dict(row), sort_keys=True)
    assert PLAINTEXT_TOKEN not in raw_db
    assert PLAINTEXT_TOKEN not in config_json
    assert PLAINTEXT_TOKEN not in snapshot_json
    assert SECRET_REF not in snapshot_json
    assert "secret://<redacted>" in snapshot_json


def test_registry_resolver_exceptions_do_not_leak_plaintext(caplog):
    class LeakyResolver:
        def resolve(self, ref: str) -> SensitiveToken:
            raise RuntimeError(f"backend leaked {PLAINTEXT_TOKEN} for {ref}")

    registry = DiscordIdentityRegistry.load(_config(), store=None, secret_resolver=LeakyResolver())
    caplog.set_level("DEBUG")

    with pytest.raises(SecretResolutionError) as excinfo:
        registry.resolve_token("bohumil")

    assert PLAINTEXT_TOKEN not in str(excinfo.value)
    assert SECRET_REF not in str(excinfo.value)
    assert PLAINTEXT_TOKEN not in caplog.text
    assert "secret://<redacted>" in str(excinfo.value)
    assert excinfo.value.__context__ is None


def test_redaction_helpers_cover_refs_token_values_and_sensitive_keys():
    payload = {
        "token_secret_ref": SECRET_REF,
        "nested": [
            {"status": f"token={PLAINTEXT_TOKEN}"},
            {"bot_token": PLAINTEXT_TOKEN},
        ],
    }

    rendered = json.dumps(redact_sensitive_data(payload), sort_keys=True)
    text = redact_sensitive_text_value(f"ref {SECRET_REF} value {PLAINTEXT_TOKEN}")

    assert PLAINTEXT_TOKEN not in rendered
    assert SECRET_REF not in rendered
    assert "secret://<redacted>" in rendered
    assert PLAINTEXT_TOKEN not in text
    assert SECRET_REF not in text
    assert "secret://<redacted>" in text


def test_gateway_secret_redaction_filter_scrubs_caplog(caplog):
    logger = logging.getLogger("gateway.discord.secret_refs.test")
    filt = SecretRedactionFilter()
    logger.addFilter(filt)
    try:
        caplog.set_level(logging.INFO, logger=logger.name)
        logger.info("starting ref=%s token=%s", SECRET_REF, PLAINTEXT_TOKEN)
        logger.info(
            f"literal ref {SECRET_REF} token {PLAINTEXT_TOKEN} arg=%s",
            "ok",
        )
    finally:
        logger.removeFilter(filt)

    assert PLAINTEXT_TOKEN not in caplog.text
    assert SECRET_REF not in caplog.text
    assert "secret://<redacted>" in caplog.text
    assert "<redacted>" in caplog.text
    assert "arg=ok" in caplog.text


def test_cli_stdout_stderr_and_db_do_not_contain_plaintext_token(tmp_path, capsys):
    class LeakyFakeClient:
        def verify_identity(self, identity):
            return {
                "agent_id": identity["agent_id"],
                "status": f"saw {PLAINTEXT_TOKEN}",
                "token": PLAINTEXT_TOKEN,
                "token_secret_ref": SECRET_REF,
            }

        def list_guild_bots(self, guild_id):  # pragma: no cover - protocol stub
            return []

    config = _config(enabled=False, mode="off", guild_allowlist=[])
    store_path = tmp_path / "discord-v2.sqlite3"

    verify_result = discord_native.verify_identities(config, client=LeakyFakeClient())
    discord_native._print_json(verify_result)
    sync_result = discord_native.sync_metadata(config, store_path=store_path)
    discord_native._print_json(sync_result)
    captured = capsys.readouterr()

    with sqlite3.connect(store_path) as conn:
        rows = conn.execute("SELECT * FROM identity_registry").fetchall()

    assert rows
    assert PLAINTEXT_TOKEN not in captured.out
    assert PLAINTEXT_TOKEN not in captured.err
    assert PLAINTEXT_TOKEN not in json.dumps(verify_result, sort_keys=True)
    assert PLAINTEXT_TOKEN not in store_path.read_bytes().decode("latin1", errors="ignore")
    assert SECRET_REF not in captured.out
