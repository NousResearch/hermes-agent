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
    for name in ("OPENAI_API_KEY", "FAL_KEY", "HERMES_MODEL_API_KEY", "AUXILIARY_VISION_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return home


def _masks():
    from agent.redact import _mask_token, _mask_token_nonreusable, mask_secret
    return ["***", "sk-...", "****redacted****", "[REDACTED]", mask_secret(REAL),
            _mask_token(REAL), _mask_token_nonreusable(REAL)]


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
