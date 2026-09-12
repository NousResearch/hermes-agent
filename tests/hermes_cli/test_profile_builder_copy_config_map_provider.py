"""Regression for #106643 — profile builder must carry a config-map provider's
``providers.<name>`` entry into the target profile, else agent init fails with a
misleading ``Unknown provider '<name>'``.

The desktop/dashboard profile path (``_write_profile_model`` in
``hermes_cli/web_routers/profiles.py``) used to write only ``model:`` (provider +
default + key_env pointer) into the target profile's ``config.yaml`` and never the
``providers.<name>`` block that *defines* a config-map (non-built-in) provider. The
target profile then looked fully configured but resolved to ``Unknown provider`` at
init, because config-map providers are not in ``PROVIDER_REGISTRY``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Standalone — no hermes_cli imports at module top to avoid binding HERMES_HOME early.


def _write_dashboard_config(home: Path, *, provider_name: str, base_url: str,
                            model: str, key_env: str) -> None:
    """Simulate a dashboard session whose active profile already has the config-map
    provider defined (the source the builder reads from)."""
    from hermes_cli.config import save_config

    save_config({
        "model": {"default": model, "provider": provider_name,
                  "base_url": base_url, "key_env": key_env},
        "providers": {
            provider_name: {
                "name": provider_name,
                "base_url": base_url,
                "model": model,
                "discover_models": True,
                "key_env": key_env,
            },
        },
    })


def _read_target_config(profile_dir: Path) -> dict:
    from hermes_cli.config import read_raw_config

    with open(profile_dir / "config.yaml", "w") as fh:
        pass  # ensure an empty file exists; _write_profile_model populates it
    raw = read_raw_config()
    return raw if isinstance(raw, dict) else {}


def test_write_profile_model_copies_config_map_provider_entry(tmp_path, monkeypatch):
    """A config-map provider assigned to a NEW profile must land its
    ``providers.<name>`` entry in the target profile, not only ``model:``."""
    from hermes_cli.config_providers import find_provider_entry
    from hermes_cli.web_routers import profiles as profiles_mod
    from hermes_cli.web_server_profiles import _hermes_home_scope

    # Dashboard session owns a home with the config-map provider defined.
    dashboard_home = tmp_path / "dashboard"
    dashboard_home.mkdir()
    provider_name = "scnet"
    base_url = "https://api.scnet.cn/api/llm/v1"
    model = "GLM-5.3-Flash"
    key_env = "HERMES_CUSTOM_SCNET_API_KEY"

    # Scope config reads/writes to the dashboard home first (the builder's session).
    with _hermes_home_scope(dashboard_home):
        _write_dashboard_config(dashboard_home, provider_name=provider_name,
                                base_url=base_url, model=model, key_env=key_env)

    # The target profile dir the builder will write into.
    target_profile = tmp_path / "profiles" / "procure_helper"
    target_profile.mkdir(parents=True)

    # Drive the same code path the dashboard endpoints use. The dashboard process
    # is itself scoped to its own home, so _write_profile_model reads the source
    # providers map from the dashboard session's config (HERMES_HOME=dashboard_home)
    # and re-scopes to the target profile only for the write.
    with _hermes_home_scope(dashboard_home):
        profiles_mod._write_profile_model(target_profile, provider_name, model)

    # Read the written target config (scope to the target profile).
    from hermes_cli.config import read_user_config_raw
    written = read_user_config_raw(target_profile / "config.yaml")

    # model: block was always written (the existing behavior).
    assert written.get("model", {}).get("provider") == provider_name
    assert written.get("model", {}).get("default") == model

    # THE FIX: the providers.<name> entry must now be present in the target profile,
    # carrying the base_url + key_env pointer so agent init can resolve it.
    providers = written.get("providers")
    assert isinstance(providers, dict), f"providers map missing entirely: {written!r}"
    stored_key, entry = find_provider_entry(providers, provider_name)
    assert stored_key is not None, f"provider {provider_name!r} not in providers map"
    assert isinstance(entry, dict)
    assert entry.get("base_url") == base_url
    assert entry.get("key_env") == key_env
    # Never copy a resolved secret — only the pointer (the reporter noted the .env
    # key line already exists in the target profile).
    assert "api_key" not in entry or not str(entry.get("api_key", "")).strip()

    # User-visible contract: the target profile now resolves the selected provider
    # through the real runtime chain, rather than merely containing plausible YAML.
    monkeypatch.setenv(key_env, "target-profile-token")
    with _hermes_home_scope(target_profile):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=provider_name, target_model=model)
    assert runtime["provider"] == "custom"
    assert runtime["base_url"] == base_url
    assert runtime["api_key"] == "target-profile-token"
    assert runtime["model"] == model


def test_write_profile_model_resolves_durable_custom_slug(tmp_path, monkeypatch):
    """The dashboard may submit ``custom:<key>``; preserve the stored providers key."""
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.web_routers import profiles as profiles_mod
    from hermes_cli.web_server_profiles import _hermes_home_scope

    dashboard_home = tmp_path / "dashboard"
    dashboard_home.mkdir()
    with _hermes_home_scope(dashboard_home):
        _write_dashboard_config(
            dashboard_home, provider_name="scnet", base_url="https://api.scnet.cn/api/llm/v1",
            model="GLM-5.3-Flash", key_env="HERMES_CUSTOM_SCNET_API_KEY")

    target_profile = tmp_path / "profiles" / "procure_helper"
    target_profile.mkdir(parents=True)
    with _hermes_home_scope(dashboard_home):
        profiles_mod._write_profile_model(target_profile, "custom:scnet", "GLM-5.3-Flash")

    written = read_user_config_raw(target_profile / "config.yaml")
    assert "scnet" in written["providers"]
    assert "custom:scnet" not in written["providers"]


def test_write_profile_model_does_not_copy_inline_provider_secrets(tmp_path):
    """Supported aliases and credential-bearing headers must not cross profile boundaries."""
    from hermes_cli.config import read_user_config_raw, save_config
    from hermes_cli.web_routers import profiles as profiles_mod
    from hermes_cli.web_server_profiles import _hermes_home_scope

    dashboard_home = tmp_path / "dashboard"
    dashboard_home.mkdir()
    with _hermes_home_scope(dashboard_home):
        save_config({
            "providers": {
                "scnet": {
                    "name": "scnet",
                    "baseUrl": "https://api.scnet.cn/api/llm/v1",
                    "keyEnv": "HERMES_CUSTOM_SCNET_API_KEY",
                    "apiMode": "openai_chat",
                    "apiKey": "literal-camel-secret",
                    "api_key": "literal-snake-secret",
                    "extra_headers": {
                        "Authorization": "Bearer literal-secret",
                        "CF-Access-Client-Secret": "literal-header-secret",
                        "X-Tenant": "procurement",
                    },
                    "key_cmd": "printf literal-command-secret",
                    "extra_body": {"token": "literal-body-secret"},
                },
            },
        })

    target_profile = tmp_path / "profiles" / "procure_helper"
    target_profile.mkdir(parents=True)
    with _hermes_home_scope(dashboard_home):
        profiles_mod._write_profile_model(target_profile, "scnet", "GLM-5.3-Flash")

    entry = read_user_config_raw(target_profile / "config.yaml")["providers"]["scnet"]
    assert "apiKey" not in entry
    assert "api_key" not in entry
    assert "extra_headers" not in entry
    assert "key_cmd" not in entry
    assert "extra_body" not in entry
    assert entry["base_url"] == "https://api.scnet.cn/api/llm/v1"
    assert entry["key_env"] == "HERMES_CUSTOM_SCNET_API_KEY"
    assert entry["api_mode"] == "openai_chat"


def test_write_profile_model_preserves_existing_target_provider_entry(tmp_path):
    """A target profile's explicit customization wins over the source definition."""
    from hermes_cli.config import read_user_config_raw, save_config
    from hermes_cli.web_routers import profiles as profiles_mod
    from hermes_cli.web_server_profiles import _hermes_home_scope

    dashboard_home = tmp_path / "dashboard"
    dashboard_home.mkdir()
    with _hermes_home_scope(dashboard_home):
        _write_dashboard_config(
            dashboard_home, provider_name="scnet", base_url="https://source.example/v1",
            model="GLM-5.3-Flash", key_env="SOURCE_KEY")

    target_profile = tmp_path / "profiles" / "procure_helper"
    target_profile.mkdir(parents=True)
    with _hermes_home_scope(target_profile):
        save_config({"providers": {"scnet": {
            "name": "scnet", "base_url": "https://target.example/v1", "key_env": "TARGET_KEY"}}})
    with _hermes_home_scope(dashboard_home):
        profiles_mod._write_profile_model(target_profile, "scnet", "GLM-5.3-Flash")

    entry = read_user_config_raw(target_profile / "config.yaml")["providers"]["scnet"]
    assert entry["base_url"] == "https://target.example/v1"
    assert entry["key_env"] == "TARGET_KEY"


def test_write_profile_model_keeps_builtin_provider_path(tmp_path):
    """A non-registry built-in such as OpenRouter must not be mistaken for custom."""
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.web_routers import profiles as profiles_mod
    from hermes_cli.web_server_profiles import _hermes_home_scope

    dashboard_home = tmp_path / "dashboard"
    dashboard_home.mkdir()
    target_profile = tmp_path / "profiles" / "openrouter_profile"
    target_profile.mkdir(parents=True)
    with _hermes_home_scope(dashboard_home):
        profiles_mod._write_profile_model(
            target_profile, "openrouter", "anthropic/claude-sonnet-4.6")

    written = read_user_config_raw(target_profile / "config.yaml")
    assert written["model"]["provider"] == "openrouter"
    assert "providers" not in written


def test_source_config_read_failure_does_not_write_dangling_model(tmp_path, monkeypatch):
    """A source read failure must fail the assignment before target config is changed."""
    from hermes_cli import config as config_mod
    from hermes_cli.web_routers import profiles as profiles_mod

    target_profile = tmp_path / "profiles" / "procure_helper"
    target_profile.mkdir(parents=True)
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: (_ for _ in ()).throw(OSError("unreadable")))

    with pytest.raises(OSError, match="unreadable"):
        profiles_mod._write_profile_model(target_profile, "scnet", "GLM-5.3-Flash")
    assert not (target_profile / "config.yaml").exists()
