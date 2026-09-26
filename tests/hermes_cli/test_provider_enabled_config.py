"""Regression coverage for ``providers.<name>.enabled`` provider gating."""

import logging
import os
from unittest import mock

import pytest
import yaml


def _write_config(tmp_path, monkeypatch, config: dict) -> dict:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Every test gets a fresh home; clear both signature caches so the behavioral
    # read used by the gate cannot retain another test's profile.
    from hermes_cli import config as config_mod

    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    return config


@pytest.fixture(autouse=True)
def _strip_provider_environment(monkeypatch):
    """Keep developer/CI credentials out of provider-discovery assertions."""
    for key in list(os.environ):
        if key.endswith(("_API_KEY", "_TOKEN")) or key in {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_PROFILE",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "OPENAI_BASE_URL",
        }:
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("entry", [{}, {"enabled": True}, {"enabled": "true"}])
def test_absent_or_true_keeps_provider_enabled(entry):
    from hermes_cli.config import is_provider_id_enabled

    assert is_provider_id_enabled("copilot", {"copilot": entry}) is True


def test_false_disables_profile_and_all_of_its_aliases():
    from hermes_cli.config import is_provider_id_enabled

    providers = {"copilot": {"enabled": False}}
    assert is_provider_id_enabled("copilot", providers) is False
    assert is_provider_id_enabled("github-copilot", providers) is False
    assert is_provider_id_enabled("deepseek", providers) is True


def test_named_custom_provider_disable_is_isolated():
    from hermes_cli.config import is_provider_id_enabled

    providers = {"relay": {"enabled": False}}
    assert is_provider_id_enabled("relay", providers) is False
    assert is_provider_id_enabled("custom:relay", providers) is False
    assert is_provider_id_enabled("custom:other", providers) is True
    assert is_provider_id_enabled("ollama", {"vllm": {"enabled": False}}) is True


def test_provider_gate_follows_profile_switches_without_cache_leak(tmp_path, monkeypatch):
    """The active profile's config wins across an A -> B -> A switch."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for home, enabled in ((home_a, False), (home_b, True)):
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({"providers": {"copilot": {"enabled": enabled}}}),
            encoding="utf-8",
        )

    from hermes_cli import config as config_mod
    from hermes_cli.auth import _is_provider_enabled

    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    for home, expected in ((home_a, False), (home_b, True), (home_a, False)):
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert _is_provider_enabled("github-copilot") is expected


def test_enabled_is_a_known_provider_key(caplog):
    from hermes_cli.config import _PROVIDER_NORMALIZE_WARNED, _normalize_custom_provider_entry

    _PROVIDER_NORMALIZE_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        result = _normalize_custom_provider_entry(
            {"api": "https://api.example.com", "enabled": False},
            provider_key="relay",
        )
    assert result is not None
    assert not [record for record in caplog.records if "unknown config keys" in record.message.lower()]


def test_disabled_api_key_provider_never_resolves_copilot_token(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import resolve_api_key_provider_credentials

    with mock.patch(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        side_effect=AssertionError("disabled provider probed Copilot credentials"),
    ):
        credentials = resolve_api_key_provider_credentials("copilot")

    assert credentials == {
        "provider": "copilot",
        "api_key": "",
        "base_url": "",
        "source": "disabled",
    }


def test_disabled_external_process_provider_never_resolves_command(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"copilot-acp": {"enabled": False}}})

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "_external_process_spec",
        lambda _config: (_ for _ in ()).throw(AssertionError("disabled provider resolved its process")),
    )
    assert auth_mod.get_external_process_provider_status("copilot-acp") == {
        "configured": False,
        "logged_in": False,
        "provider": "copilot-acp",
        "disabled": True,
    }
    credentials = auth_mod.resolve_external_process_provider_credentials("copilot-acp")
    assert credentials == {
        "provider": "copilot-acp",
        "api_key": "",
        "base_url": "",
        "command": "",
        "args": [],
        "source": "disabled",
    }


def test_auth_status_and_explicit_discovery_ignore_stale_credentials(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"deepseek": {"enabled": False}}})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-stale")

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "_resolve_api_key_provider_secret",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled provider status probed credentials")
        ),
    )
    expected = {
        "logged_in": False,
        "configured": False,
        "provider": "deepseek",
        "disabled": True,
    }
    assert auth_mod.get_api_key_provider_status("deepseek") == expected
    assert auth_mod.get_auth_status("deepseek") == expected
    assert auth_mod.is_provider_explicitly_configured("deepseek") is False


def test_explicit_disabled_provider_fails_closed_through_alias(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"copilot": {"enabled": False}}})

    from hermes_cli.auth import AuthError, resolve_provider

    with pytest.raises(AuthError) as caught:
        resolve_provider("github-copilot")
    assert caught.value.code == "provider_disabled"


def test_disabled_config_pin_is_not_revived_by_stale_active_login(tmp_path, monkeypatch):
    _write_config(
        tmp_path,
        monkeypatch,
        {
            "model": {"provider": "copilot", "default": "gpt-5"},
            "providers": {"copilot": {"enabled": False}},
        },
    )

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "_load_auth_store",
        lambda: {"active_provider": "copilot", "providers": {"copilot": {"token": "stale"}}},
    )
    with pytest.raises(auth_mod.AuthError) as caught:
        auth_mod.resolve_provider("auto")
    assert caught.value.code == "provider_disabled"


def test_auto_env_discovery_skips_disabled_provider(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"deepseek": {"enabled": False}}})

    from hermes_cli.auth import _env_key_auto_detected

    def read_env(name):
        return "sk-stale" if name == "DEEPSEEK_API_KEY" else ""

    assert _env_key_auto_detected(read_env, None) is None


def test_absent_flag_preserves_api_key_resolution(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from hermes_cli.auth import resolve_api_key_provider_credentials

    credentials = resolve_api_key_provider_credentials("deepseek")
    assert credentials["api_key"] == "sk-test"
    assert credentials["source"] == "DEEPSEEK_API_KEY"


def test_disabled_primary_cannot_be_revived_by_configured_fallback(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path,
        monkeypatch,
        {
            "providers": {"copilot": {"enabled": False}},
            "fallback_providers": [{"provider": "deepseek", "model": "deepseek-chat"}],
        },
    )
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback

    with pytest.raises(ValueError, match="disabled"):
        resolve_runtime_with_fallback(config, requested="copilot")


def test_picker_skips_disabled_profile_before_credential_probe(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"copilot": {"enabled": False}}})

    import hermes_cli.model_switch_providers as picker_mod

    build = picker_mod._PickerBuild(
        current_provider="copilot",
        current_base_url="",
        current_model="gpt-5",
        max_models=None,
        for_picker=True,
        force_fresh_nous_tier=False,
        probe_custom_providers=False,
        probe_current_custom_provider=False,
        refresh=True,
        excluded=set(),
        curated={"copilot": ["gpt-5"]},
    )
    probed = []

    def has_credentials(_build, _pid, hermes_slug, _overlay):
        probed.append(hermes_slug)
        if hermes_slug == "copilot":
            raise AssertionError("disabled provider reached credential discovery")
        return False

    monkeypatch.setattr(picker_mod, "_overlay_has_creds", has_credentials)
    picker_mod._lap_overlay_rows(
        build,
        {},
        {"copilot": {"enabled": False}},
    )
    assert "copilot" not in probed
    assert not [row for row in build.results if row["slug"] == "copilot"]


def test_picker_alias_disable_filters_existing_row():
    from hermes_cli.model_switch_providers import _finalize_picker_rows

    rows = [{
        "slug": "copilot",
        "provider_id": "github-copilot",
        "name": "Copilot",
        "is_current": True,
        "models": ["gpt-5"],
        "total_models": 1,
    }]
    assert _finalize_picker_rows(
        rows,
        {"github-copilot": {"enabled": False}},
        "gpt-5",
    ) == []


def test_disabled_provider_is_not_offered_by_model_inventory(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"copilot": {"enabled": False}}})

    import hermes_cli.models as models_mod

    probed = []

    def has_credentials(provider_id):
        probed.append(provider_id)
        if provider_id == "copilot":
            raise AssertionError("disabled provider reached inventory credential discovery")
        return False

    monkeypatch.setattr(models_mod, "_provider_has_credentials", has_credentials)
    rows = models_mod.list_available_providers()
    assert "copilot" not in {row["id"] for row in rows}
    assert "copilot" not in probed


def test_disabled_provider_is_not_switchable_with_stale_credentials(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"providers": {"deepseek": {"enabled": False}}})

    import hermes_cli.auth as auth_mod
    from hermes_cli.models_detect import provider_has_credentials

    monkeypatch.setattr(
        auth_mod,
        "get_auth_status",
        lambda _provider: (_ for _ in ()).throw(
            AssertionError("disabled provider reached model-switch credential discovery")
        ),
    )
    assert provider_has_credentials("deepseek") is False


def test_disabled_custom_provider_is_not_routable():
    from hermes_cli.providers import resolve_user_provider

    providers = {
        "relay": {
            "enabled": False,
            "base_url": "https://relay.example/v1",
            "key_env": "RELAY_API_KEY",
        }
    }
    assert resolve_user_provider("relay", providers) is None
