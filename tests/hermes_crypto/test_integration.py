"""Integration tests: the credential readers transparently handle encryption."""

from __future__ import annotations

import json

from hermes_constants import get_hermes_home
from hermes_crypto import detect, migrate

FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}


def test_load_env_decrypts_transparently():
    import hermes_cli.config as config

    env_path = get_hermes_home() / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-roundtrip\nREGION=us\n", encoding="utf-8")

    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    assert detect.is_encrypted(env_path.read_bytes())

    config.invalidate_env_cache()
    loaded = config.load_env()
    assert loaded["OPENAI_API_KEY"] == "sk-roundtrip"
    assert loaded["REGION"] == "us"


def test_save_env_value_keeps_file_encrypted():
    import hermes_cli.config as config

    env_path = get_hermes_home() / ".env"
    env_path.write_text("EXISTING=1\n", encoding="utf-8")
    migrate.enable("keyfile")

    config.save_env_value("ADDED_KEY", "added-value")
    assert detect.is_encrypted(env_path.read_bytes())
    config.invalidate_env_cache()
    assert config.load_env()["ADDED_KEY"] == "added-value"


def test_auth_store_round_trips_through_encryption():
    import hermes_cli.auth as auth

    auth_path = get_hermes_home() / "auth.json"
    auth_path.write_text(
        json.dumps(
            {"version": 1, "providers": {}, "credential_pool": {"openrouter": [{"access_token": "or-xyz"}]}}
        ),
        encoding="utf-8",
    )

    migrate.enable("keyfile")
    assert detect.is_encrypted(auth_path.read_bytes())

    store = auth._load_auth_store()
    assert store["credential_pool"]["openrouter"][0]["access_token"] == "or-xyz"

    # A write back through _save_auth_store must keep the file encrypted.
    auth._save_auth_store(store)
    assert detect.is_encrypted(auth_path.read_bytes())
    assert auth._load_auth_store()["version"] == 1


def test_plaintext_reads_unchanged_when_encryption_disabled():
    """Backward-compat: with no keystore, plaintext files read exactly as before."""
    import hermes_cli.auth as auth
    import hermes_cli.config as config

    (get_hermes_home() / ".env").write_text("PLAIN_KEY=plain-value\n", encoding="utf-8")
    (get_hermes_home() / "auth.json").write_text(
        json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
    )

    config.invalidate_env_cache()
    assert config.load_env()["PLAIN_KEY"] == "plain-value"
    assert auth._load_auth_store()["version"] == 1


def test_disable_then_read_returns_plaintext():
    import hermes_cli.config as config

    env_path = get_hermes_home() / ".env"
    env_path.write_text("TOKEN_KEY=abc123\n", encoding="utf-8")
    migrate.enable("keyfile")
    migrate.disable()

    assert not detect.is_encrypted(env_path.read_bytes())
    config.invalidate_env_cache()
    assert config.load_env()["TOKEN_KEY"] == "abc123"


def test_stash_passphrase_removes_env_var_and_still_unlocks(monkeypatch):
    """The startup stash moves the passphrase out of os.environ (so no child
    process can inherit it) while get_data_key() keeps unlocking from the
    in-memory copy."""
    import os

    import hermes_crypto
    from hermes_crypto import keystore

    env_path = get_hermes_home() / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-stash\n", encoding="utf-8")
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    keystore.lock()

    monkeypatch.setenv("HERMES_ENCRYPTION_PASSPHRASE", "pw")
    hermes_crypto.stash_passphrase_from_env()

    assert "HERMES_ENCRYPTION_PASSPHRASE" not in os.environ
    assert hermes_crypto.get_data_key()  # unlocks from the stash

    # A later explicit lock()/get_data_key() cycle still works headlessly.
    keystore.lock()
    assert hermes_crypto.get_data_key()


def test_env_var_unlock_survives_relock(monkeypatch):
    """Direct env-var unlock (no startup stash) keeps the proven passphrase
    in memory so a re-lock does not strand a headless process."""
    import os

    import hermes_crypto
    from hermes_crypto import keystore

    env_path = get_hermes_home() / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-relock\n", encoding="utf-8")
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2)
    keystore.lock()

    monkeypatch.setenv("HERMES_ENCRYPTION_PASSPHRASE", "pw")
    assert hermes_crypto.get_data_key()
    assert "HERMES_ENCRYPTION_PASSPHRASE" not in os.environ

    keystore.lock()
    assert hermes_crypto.get_data_key()
