"""Config writers store credentials in ``.env`` and only a reference in ``config.yaml`` (#77471).

Each test runs a real writer against a temp ``HERMES_HOME``, then checks the contract: the secret
is not in ``config.yaml``, it is in the profile's ``.env``, and the runtime reader still resolves it.
"""

from __future__ import annotations

import pytest

SECRET = "sk-77471-synthetic-secret-value"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").touch()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("HERMES_MODEL_API_KEY", "HERMES_DELEGATION_API_KEY", "AUXILIARY_VISION_API_KEY",
                 "HERMES_CUSTOM_LLM_EXAMPLE_COM_API_KEY", "HERMES_CUSTOM_LLM_EXAMPLE_COM_8443_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return home


def _config_text(home) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


def _assert_secret_only_in_env(home) -> None:
    assert SECRET not in _config_text(home)
    assert SECRET in (home / ".env").read_text(encoding="utf-8")


def test_config_set_model_api_key_stores_a_reference(home, capsys):
    from hermes_cli.config import load_config, set_config_value
    from hermes_cli.runtime_provider import resolve_runtime_provider

    set_config_value("model.provider", "custom")
    set_config_value("model.base_url", "https://llm.example.com/v1")
    set_config_value("model.default", "m")
    set_config_value("model.api_key", SECRET)

    _assert_secret_only_in_env(home)
    assert "${HERMES_MODEL_API_KEY}" in _config_text(home)
    assert SECRET not in capsys.readouterr().out
    assert load_config()["model"]["api_key"] == SECRET
    assert resolve_runtime_provider(requested="custom")["api_key"] == SECRET


@pytest.mark.parametrize("key,env_name", [
    ("auxiliary.vision.api_key", "AUXILIARY_VISION_API_KEY"),
    ("delegation.api_key", "HERMES_DELEGATION_API_KEY"),
])
def test_config_set_aux_and_delegation_keys_store_a_reference(home, key, env_name):
    from hermes_cli.config import load_config, set_config_value

    set_config_value(key, SECRET)

    _assert_secret_only_in_env(home)
    assert f"${{{env_name}}}" in _config_text(home)
    section, *rest = key.split(".")
    node = load_config()[section]
    for part in rest[:-1]:
        node = node[part]
    assert node["api_key"] == SECRET


def test_config_set_keeps_an_explicit_reference_and_inline_values_still_resolve(home):
    """A ``${VAR}`` the user types is stored verbatim; an inline value written by hand (no migration)
    keeps resolving."""
    from hermes_cli.config import load_config, set_config_value

    set_config_value("model.api_key", "${MY_OWN_KEY}")
    assert "${MY_OWN_KEY}" in _config_text(home)
    assert "MY_OWN_KEY" not in (home / ".env").read_text(encoding="utf-8")

    (home / "config.yaml").write_text(f"model:\n  provider: custom\n  api_key: {SECRET}\n", encoding="utf-8")
    assert load_config()["model"]["api_key"] == SECRET


def test_dashboard_main_custom_endpoint_stores_a_reference(home, monkeypatch):
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **k: {"accepted": True, "persist": True, "recognized": True, "message": None})
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: x\n", encoding="utf-8")

    _apply_model_assignment_sync("main", "custom", "m", "", "https://llm.example.com:8443/v1", SECRET)

    _assert_secret_only_in_env(home)
    cfg = load_config()
    assert cfg["model"]["api_key"] == SECRET
    row = next(e for e in cfg["custom_providers"] if e["base_url"] == "https://llm.example.com:8443/v1")
    assert row["key_env"] == "HERMES_CUSTOM_LLM_EXAMPLE_COM_8443_API_KEY" and "api_key" not in row
    assert resolve_runtime_provider(requested="custom")["api_key"] == SECRET


def test_dashboard_aux_slot_stores_a_reference(home):
    from agent.auxiliary_client import _resolve_task_provider_model
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: x\n", encoding="utf-8")

    _apply_model_assignment_sync("auxiliary", "custom", "m", "vision", "https://llm.example.com/v1", SECRET)

    _assert_secret_only_in_env(home)
    assert load_config()["auxiliary"]["vision"]["api_key"] == SECRET
    assert _resolve_task_provider_model("vision")[3] == SECRET


def test_hermes_model_aux_flow_stores_a_reference(home):
    from agent.auxiliary_client import _resolve_task_provider_model
    from hermes_cli.main_provider_setup import _save_aux_choice

    _save_aux_choice("vision", provider="custom", model="m", base_url="https://llm.example.com/v1", api_key=SECRET)

    _assert_secret_only_in_env(home)
    assert _resolve_task_provider_model("vision")[3] == SECRET


def test_new_custom_provider_entry_stores_key_env(home):
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.main_provider_setup import _save_custom_provider

    _save_custom_provider("https://llm.example.com/v1", SECRET, "m", name="Example")

    _assert_secret_only_in_env(home)
    entry = load_config()["custom_providers"][0]
    assert entry["key_env"] == "HERMES_CUSTOM_LLM_EXAMPLE_COM_API_KEY" and "api_key" not in entry
    assert resolve_runtime_provider(requested="custom:example")["api_key"] == SECRET


def test_endpoint_names_separate_gateways_behind_one_host():
    from hermes_cli.config_secret_refs import endpoint_api_key_env

    assert endpoint_api_key_env("http://localhost:11434/v1") == "HERMES_CUSTOM_LOCALHOST_11434_API_KEY"
    assert endpoint_api_key_env("http://h:9/old/v1") != endpoint_api_key_env("http://h:9/new/v1")


def test_webhook_env_secret_is_never_persisted_to_config_yaml(home, monkeypatch):
    """The issue's webhook claim: ``WEBHOOK_SECRET`` from the environment is folded into the
    in-memory gateway config only; config saves (home channel, ``config set``) never write it."""
    from gateway.config import HomeChannel, Platform, load_gateway_config, persist_home_channel
    from hermes_cli.config import set_config_value

    (home / "config.yaml").write_text("platforms:\n  webhook:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)

    assert load_gateway_config().platforms[Platform.WEBHOOK].extra["secret"] == SECRET
    persist_home_channel(HomeChannel(platform=Platform.WEBHOOK, chat_id="c1", name="Home"))
    set_config_value("display.skin", "default")

    assert SECRET not in _config_text(home)
