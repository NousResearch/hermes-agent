"""Fresh profiles retain the selected custom endpoint, without copying its neighbors."""

from pathlib import Path

import pytest

from hermes_cli.config import (
    DEFAULT_CONFIG,
    atomic_config_write,
    get_compatible_custom_providers,
    read_user_config_raw,
)
from hermes_cli.profiles import create_profile
from hermes_cli.profile_model_seed import launch_model_seed
from hermes_cli import runtime_provider


@pytest.mark.parametrize(
    "schema,provider,url_key",
    [
        ("providers", "lab", "base_url"),
        ("providers", "custom:lab", "base_url"),
        ("providers", "custom:display-name", "base_url"),
        ("custom_providers", "custom:display-name", "base_url"),
        ("providers", "custom:lab", "baseUrl"),
    ],
)
def test_fresh_profile_retains_resolvable_custom_endpoint(
    tmp_path, monkeypatch, schema, provider, url_key
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LAB_TOKEN", "synthetic-secret")
    endpoint = {
        "name": "Display Name",
        url_key: "http://127.0.0.1:11434/v1",
        "api_key": "${LAB_TOKEN}",
        "api_mode": "chat_completions",
    }
    unrelated = {"name": "Unrelated", "base_url": "http://127.0.0.1:11435/v1"}
    catalog = (
        {"lab": endpoint, "other": unrelated}
        if schema == "providers"
        else [endpoint, unrelated]
    )
    source = {
        "model": {"provider": provider, "default": "synthetic-model"},
        schema: catalog,
        "_config_version": DEFAULT_CONFIG["_config_version"],
    }
    atomic_config_write(home / "config.yaml", source)
    monkeypatch.setattr(runtime_provider, "load_config", lambda: source)
    assert runtime_provider._get_named_custom_provider(provider) is not None

    profile = create_profile("sample-bot", no_alias=True, no_skills=True)
    created = read_user_config_raw(profile / "config.yaml")

    assert created["model"] == source["model"]
    copied = get_compatible_custom_providers(created)
    assert [entry["base_url"] for entry in copied] == [endpoint[url_key]]
    assert copied[0]["api_key"] == endpoint["api_key"]
    monkeypatch.setattr(runtime_provider, "load_config", lambda: created)
    resolved = runtime_provider._get_named_custom_provider(provider)
    assert resolved is not None
    assert resolved["base_url"] == endpoint[url_key]
    source["model"] = {"provider": "nous", "default": "other-model"}
    atomic_config_write(home / "config.yaml", source)
    assert read_user_config_raw(profile / "config.yaml") == created


@pytest.mark.parametrize("schema", ["providers", "custom_providers"])
@pytest.mark.parametrize("provider", ["openrouter", "custom:openrouter"])
def test_builtin_name_collision_only_copies_explicit_custom_selection(
    tmp_path, monkeypatch, schema, provider
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    endpoint = {
        "name": "OpenRouter",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "synthetic-other-secret",
    }
    catalog = {"other-endpoint": endpoint} if schema == "providers" else [endpoint]
    source = {
        "model": {"provider": provider, "default": "synthetic-model"},
        schema: catalog,
        "_config_version": DEFAULT_CONFIG["_config_version"],
    }
    atomic_config_write(home / "config.yaml", source)
    monkeypatch.setattr(runtime_provider, "load_config", lambda: source)
    is_custom = provider.startswith("custom:")
    assert (
        runtime_provider._get_named_custom_provider(provider) is not None
    ) is is_custom

    profile = create_profile("sample-bot", no_alias=True, no_skills=True)
    created = read_user_config_raw(profile / "config.yaml")

    assert created["model"] == source["model"]
    if is_custom:
        copied = get_compatible_custom_providers(created)
        assert [entry["base_url"] for entry in copied] == [endpoint["base_url"]]
        assert copied[0]["api_key"] == endpoint["api_key"]
        monkeypatch.setattr(runtime_provider, "load_config", lambda: created)
        assert runtime_provider._get_named_custom_provider(provider) is not None
    else:
        assert "providers" not in created
        assert "custom_providers" not in created


def test_fresh_profile_keeps_explicit_builtin_provider_settings(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = {
        "model": {"provider": "openrouter", "default": "synthetic-model"},
        "providers": {"openrouter": {"request_timeout_seconds": 150}},
        "_config_version": DEFAULT_CONFIG["_config_version"],
    }
    atomic_config_write(home / "config.yaml", source)

    profile = create_profile("sample-bot", no_alias=True, no_skills=True)

    assert (
        read_user_config_raw(profile / "config.yaml")["providers"]
        == source["providers"]
    )


@pytest.mark.parametrize(
    "providers,legacy",
    [
        (
            {"lab": {"request_timeout_seconds": 150}},
            [{"name": "Lab", "base_url": "http://127.0.0.1:11434/v1"}],
        ),
        (
            {
                "other": {"name": "Lab", "base_url": "http://127.0.0.1:11434/v1"},
                "lab": {"base_url": "http://127.0.0.1:11435/v1"},
            },
            [],
        ),
    ],
)
def test_model_seed_keeps_runtime_endpoint_selection(monkeypatch, providers, legacy):
    source = {
        "model": {"provider": "lab", "default": "synthetic-model"},
        "providers": providers,
        "custom_providers": legacy,
    }
    monkeypatch.setattr(runtime_provider, "load_config", lambda: source)
    selected = runtime_provider._get_named_custom_provider("lab")
    assert selected is not None

    seed = launch_model_seed(source)

    monkeypatch.setattr(runtime_provider, "load_config", lambda: seed)
    inherited = runtime_provider._get_named_custom_provider("lab")
    assert inherited is not None
    assert inherited["base_url"] == selected["base_url"]


def test_model_seed_uses_supplied_config_without_reading_credentials(monkeypatch):
    import hermes_cli.runtime_provider_custom as custom_runtime

    def unexpected_read(*args, **kwargs):
        pytest.fail(
            "model seed must use the supplied config without resolving credentials"
        )

    monkeypatch.setattr(runtime_provider, "load_config", unexpected_read)
    monkeypatch.setattr(custom_runtime, "get_secret_str", unexpected_read)
    source = {
        "model": {"provider": "custom:lab", "default": "synthetic-model"},
        "providers": {
            "lab": {
                "api": "http://127.0.0.1:11434/v1",
                "key_env": "LAB_TOKEN",
                "api_key": "${LAB_TOKEN}",
            }
        },
    }

    seed = launch_model_seed(source)

    assert seed == source
