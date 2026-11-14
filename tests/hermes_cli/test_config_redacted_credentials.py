"""Writers refuse to persist a redacted display mask as a credential (#42727).

The agent sees credentials only through the redactor and ``mask_secret``; writing one of those
back used to replace a working key with a dead string. Each test runs a real writer against a
temp ``HERMES_HOME`` and checks what lands on disk.
"""

from __future__ import annotations

import pytest

import hermes_yaml as yaml

REAL = "sk-proj-" + "a1B2c3D4" * 6


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text(f"OPENAI_API_KEY={REAL}\n", encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "auxiliary": {"vision": {"provider": "custom", "api_key": REAL}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("OPENAI_API_KEY", "FAL_KEY", "HERMES_MODEL_API_KEY", "AUXILIARY_VISION_API_KEY",
                 "SUDO_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    return home


_SAMPLES = {
    "OPENAI_API_KEY": REAL,
    "TELEGRAM_BOT_TOKEN": "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
    "SLACK_BOT_TOKEN": "xo" + "xb-1234567890-1234567890-abcdefghijABCDEFGHIJ",
    "GITHUB_TOKEN": "ghp_" + "abcdefghij" * 3 + "abcdef",
}


def _redactor_values():
    """Every value Hermes's own redactor/maskers show the agent for the sample credentials."""
    from agent.redact import _mask_token, _mask_token_nonreusable, mask_secret, redact_sensitive_text
    shown = {"***", "sk-...", "****redacted****", "[REDACTED]"}
    for name, secret in _SAMPLES.items():
        shown |= {mask_secret(secret), _mask_token(secret), _mask_token_nonreusable(secret),
                  "Bearer " + _mask_token(secret)}
        for mode in ({}, {"file_read": True}):
            for doc, sep in ((f"{name}={secret}", "="), (f"token: {secret}", ": "),
                             (f"Authorization: Bearer {secret}", ": ")):
                out = redact_sensitive_text(doc, force=True, **mode)
                assert secret not in out
                shown.add(out.split(sep, 1)[1])
    return sorted(shown)


def _masks():
    masks = _redactor_values()
    # The shapes that slipped past the first predicate; pinned so a redactor change can't drop them.
    assert {"Bearer ***", "123456789:***"} <= set(masks)
    return masks + ["xoxb-***", "Bearer sk-pro...WXYZ", "********", "sk-****WXYZ"]


def test_human_passwords_are_not_masks():
    from hermes_cli.config_credential_values import is_redacted_credential_value
    for pw in ("Hello...World1", "a.b.c...defgh", "correct horse battery", "Bearer-token-2024!",
               "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw", "${OPENAI_API_KEY}", REAL,
               "MyRedactedPassword123!", "Unredacted2024", "redacted-pass-99"):
        assert not is_redacted_credential_value(pw), pw


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "FAL_KEY"])
def test_save_env_value_refuses_masks(home, key):
    from hermes_cli.config import save_env_value
    for mask in _masks():
        with pytest.raises(ValueError, match="redacted display mask"):
            save_env_value(key, mask)
    assert (home / ".env").read_text(encoding="utf-8") == f"OPENAI_API_KEY={REAL}\n"


def test_save_env_value_keeps_real_and_non_credential_values(home):
    from hermes_cli.config import load_env, save_env_value
    save_env_value("OPENAI_API_KEY", "sk-new-" + "Z9" * 20)
    save_env_value("TELEGRAM_ALLOWED_USERS", "***")  # not a credential name
    save_env_value("FAL_KEY", "")  # clearing stays possible
    env = load_env()
    assert env["OPENAI_API_KEY"] == "sk-new-" + "Z9" * 20
    assert env["TELEGRAM_ALLOWED_USERS"] == "***"


def test_save_config_refuses_mask_on_nested_credential(home):
    from hermes_cli.config import read_raw_config, save_config
    from hermes_cli.config_backend import ConfigValueError
    cfg = read_raw_config()
    for mask in _masks():
        cfg["auxiliary"]["vision"]["api_key"] = mask
        with pytest.raises(ConfigValueError, match="auxiliary.vision.api_key"):
            save_config(cfg)
    assert REAL in (home / "config.yaml").read_text(encoding="utf-8")


def test_save_config_allows_env_ref_and_preexisting_mask(home):
    from hermes_cli.config import read_raw_config, save_config
    cfg = read_raw_config()
    cfg["auxiliary"]["vision"]["api_key"] = "${AUXILIARY_VISION_API_KEY}"
    save_config(cfg)
    # A mask already on disk does not block unrelated saves.
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {"api_key": "sk-...", "default": "a"}}))
    cfg = read_raw_config()
    cfg["model"]["default"] = "b"
    save_config(cfg)
    assert yaml.safe_load((home / "config.yaml").read_text())["model"]["default"] == "b"


@pytest.mark.parametrize("key", ["model.api_key", "auxiliary.vision.api_key", "providers.x.token"])
def test_config_set_refuses_masks(home, key):
    from hermes_cli.config import set_config_value
    from hermes_cli.config_backend import ConfigWriteError
    before = ((home / "config.yaml").read_text(), (home / ".env").read_text())
    with pytest.raises((SystemExit, ConfigWriteError)):
        set_config_value(key, "sk-pro...aaaa")
    assert ((home / "config.yaml").read_text(), (home / ".env").read_text()) == before


def test_save_config_allows_deleting_entry_beside_preexisting_mask(home):
    from hermes_cli.config import read_raw_config, save_config
    (home / "config.yaml").write_text(yaml.safe_dump({"custom_providers": [
        {"name": "a", "api_key": REAL}, {"name": "b", "api_key": "***"}]}))
    cfg = read_raw_config()
    del cfg["custom_providers"][0]  # the old mask moves from index 1 to 0
    save_config(cfg)
    assert yaml.safe_load((home / "config.yaml").read_text())["custom_providers"] == [
        {"name": "b", "api_key": "***"}]
    cfg["custom_providers"].append({"name": "c", "api_key": "sk-pro...aaaa"})
    from hermes_cli.config_backend import ConfigValueError
    with pytest.raises(ConfigValueError, match="custom_providers.1.api_key"):
        save_config(cfg)


@pytest.mark.parametrize("target", ["model", "provider_a", "auxiliary"])
def test_stale_mask_elsewhere_does_not_exempt_a_new_mask(home, target):
    """A ``***`` already on disk at one credential path must not let ``***`` overwrite another."""
    from hermes_cli.config import read_raw_config, save_config
    from hermes_cli.config_backend import ConfigValueError
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"api_key": REAL},
        "auxiliary": {"vision": {"api_key": REAL}},
        "custom_providers": [{"name": "a", "api_key": REAL}, {"name": "b", "api_key": "***"}]}))
    before = (home / "config.yaml").read_text()
    cfg = read_raw_config()
    {"model": lambda: cfg["model"].__setitem__("api_key", "***"),
     "provider_a": lambda: cfg["custom_providers"][0].__setitem__("api_key", "***"),
     "auxiliary": lambda: cfg["auxiliary"]["vision"].__setitem__("api_key", "***")}[target]()
    with pytest.raises(ConfigValueError):
        save_config(cfg)
    assert (home / "config.yaml").read_text() == before


def test_save_config_allows_reordering_entries_beside_preexisting_mask(home):
    from hermes_cli.config import read_raw_config, save_config
    (home / "config.yaml").write_text(yaml.safe_dump({"custom_providers": [
        {"name": "a", "api_key": REAL}, {"name": "b", "api_key": "***"}]}))
    cfg = read_raw_config()
    cfg["custom_providers"].reverse()
    save_config(cfg)
    assert [p["name"] for p in yaml.safe_load((home / "config.yaml").read_text())["custom_providers"]] == ["b", "a"]


# Human-chosen passwords: mask-SHAPED values are real passwords unless they are a mask of the
# password stored now (the write-back). Built by concatenation: ``"Hello" + "..." + "2026"``.
_SHAPED_PASSWORDS = ["Hello" + "..." + "2026", "p" + "..." + "word", "wait" + "\u2026" + "1234", "*" * 4]


@pytest.mark.parametrize("password", _SHAPED_PASSWORDS)
def test_mask_shaped_sudo_password_can_be_saved(home, password):
    from hermes_cli.config import load_env, save_env_value
    save_env_value("SUDO_PASSWORD", password)  # not refused; .env keeps ASCII (_check_non_ascii_credential)
    assert load_env()["SUDO_PASSWORD"] == password.encode("ascii", "ignore").decode()


def test_sudo_password_refuses_a_mask_of_the_stored_password(home):
    from agent.redact import mask_secret, redact_sensitive_text
    from hermes_cli.config import load_env, save_env_value
    stored = "correct-horse-" + "battery9"
    save_env_value("SUDO_PASSWORD", stored)
    shown = redact_sensitive_text(f"SUDO_PASSWORD={stored}", force=True).split("=", 1)[1]
    for mask in {shown, mask_secret(stored), "[REDACTED]", "«redacted-secret»"}:
        with pytest.raises(ValueError, match="redacted display mask"):
            save_env_value("SUDO_PASSWORD", mask)
    assert load_env()["SUDO_PASSWORD"] == stored


def test_config_password_leaf_refuses_only_a_mask_of_the_stored_password(home):
    from agent.redact import mask_secret
    from hermes_cli.config import read_raw_config, save_config
    from hermes_cli.config_backend import ConfigValueError
    stored = "correct-horse-" + "battery9"
    cfg = read_raw_config()
    cfg["dashboard"] = {"basic_auth": {"username": "op", "password": stored}}
    save_config(cfg)
    for mask in (mask_secret(stored), "***"):
        cfg["dashboard"]["basic_auth"]["password"] = mask
        with pytest.raises(ConfigValueError, match="dashboard.basic_auth.password"):
            save_config(cfg)
    for password in _SHAPED_PASSWORDS:
        cfg["dashboard"]["basic_auth"]["password"] = password
        save_config(cfg)
        assert read_raw_config()["dashboard"]["basic_auth"]["password"] == password


def test_provider_keys_still_refuse_every_mask_shape(home):
    """The password carve-out is by name: a provider key keeps the full shape refusal."""
    from hermes_cli.config import save_env_value
    for mask in _SHAPED_PASSWORDS:
        with pytest.raises(ValueError, match="redacted display mask"):
            save_env_value("OPENAI_API_KEY", mask)


@pytest.mark.parametrize("key", ["model.api_key", "mcp_servers.s.headers.X-API-Key"])
def test_keyed_writer_refuses_masks(home, key):
    """``write_config_key(s)`` (per-key round-trip, not the document writer) runs the same check."""
    from hermes_cli.config import get_config_path
    from hermes_cli.config_backend import ConfigValueError, write_config_key, write_config_keys
    before = (home / "config.yaml").read_text()
    for mask in ("***", "*** ***", "sk-pro" + "...WXYZ"):
        with pytest.raises(ConfigValueError, match=key.replace(".", r"\.")):
            write_config_key(get_config_path(), key, mask)
        with pytest.raises(ConfigValueError):
            write_config_keys(get_config_path(), {"display.personality": "kawaii", key: mask})
    assert (home / "config.yaml").read_text() == before
    write_config_key(get_config_path(), key, REAL)
    write_config_key(get_config_path(), "display.personality", "***")  # not a credential
    assert yaml.safe_load((home / "config.yaml").read_text())["display"]["personality"] == "***"
