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
from hermes_cli import runtime_provider


@pytest.mark.parametrize(
    "schema,provider",
    [
        ("providers", "lab"),
        ("providers", "custom:lab"),
        ("providers", "custom:display-name"),
        ("custom_providers", "custom:display-name"),
    ],
)
def test_fresh_profile_retains_resolvable_custom_endpoint(
    tmp_path, monkeypatch, schema, provider
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LAB_TOKEN", "synthetic-secret")
    endpoint = {
        "name": "Display Name",
        "base_url": "http://127.0.0.1:11434/v1",
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
    assert [entry["base_url"] for entry in copied] == [endpoint["base_url"]]
    assert copied[0]["api_key"] == endpoint["api_key"]
    monkeypatch.setattr(runtime_provider, "load_config", lambda: created)
    resolved = runtime_provider._get_named_custom_provider(provider)
    assert resolved is not None
    assert resolved["base_url"] == endpoint["base_url"]
    source["model"] = {"provider": "nous", "default": "other-model"}
    atomic_config_write(home / "config.yaml", source)
    assert read_user_config_raw(profile / "config.yaml") == created
