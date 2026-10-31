"""Numeric config paths must not create invalid list-shaped mappings (#78370)."""

from pathlib import Path

import pytest
import yaml

from hermes_cli.config import set_config_value


@pytest.mark.parametrize(
    "config,key,error",
    [
        ({}, "custom_providers.0.api_key", "not an existing list"),
        ({"custom_providers": None}, "custom_providers.0.api_key", "not an existing list"),
        ({"custom_providers": "old"}, "custom_providers.0.api_key", "not an existing list"),
        ({"custom_providers": {}}, "custom_providers.0.api_key", "not an existing list"),
        ({}, "custom_providers.-1.api_key", "not an existing list"),
        ({}, "agent.disabled_toolsets.0", "not an existing list"),
        ({"custom_providers": []}, "custom_providers.0.api_key", "out of range"),
        ({"custom_providers": [{}]}, "custom_providers.5.api_key", "out of range"),
        ({"custom_providers": [{}]}, "custom_providers.5", "out of range"),
        ({"custom_providers": [{}]}, "custom_providers.abc.api_key", "not a numeric"),
        ({"custom_providers": [{}]}, "custom_providers.abc", "not a numeric"),
    ],
)
def test_invalid_list_path_exits_without_writing(config, key, error, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config_path = tmp_path / "config.yaml"
    original = yaml.safe_dump(config)
    config_path.write_text(original)

    with pytest.raises(SystemExit) as exc:
        set_config_value(key, "new-value")

    assert exc.value.code == 1
    assert error in capsys.readouterr().err
    assert config_path.read_text() == original


@pytest.mark.parametrize(
    "config,key,expected",
    [
        (
            {"custom_providers": [{"name": "old"}]},
            "custom_providers.0.name",
            {"custom_providers": [{"name": "new-value"}]},
        ),
        (
            {"custom_providers": [{"name": "first"}, {"name": "old"}]},
            "custom_providers.-1.name",
            {"custom_providers": [{"name": "first"}, {"name": "new-value"}]},
        ),
        (
            {"telegram": {"channel_overrides": {"123456": {"model": "old"}}}},
            "telegram.channel_overrides.123456.model",
            {"telegram": {"channel_overrides": {"123456": {"model": "new-value"}}}},
        ),
        (
            {"telegram": {"channel_overrides": {}}},
            "telegram.channel_overrides.123456.model",
            {"telegram": {"channel_overrides": {"123456": {"model": "new-value"}}}},
        ),
        (
            {"telegram": {}},
            "telegram.channel_overrides.123456.model",
            {"telegram": {"channel_overrides": {"123456": {"model": "new-value"}}}},
        ),
        (
            {},
            "platforms.telegram.channel_overrides.123456.model",
            {"platforms": {"telegram": {"channel_overrides": {"123456": {"model": "new-value"}}}}},
        ),
        (
            {},
            "platforms.telegram.channel_overrides.-123456.model",
            {"platforms": {"telegram": {"channel_overrides": {"-123456": {"model": "new-value"}}}}},
        ),
        (
            {"agent.disabled_toolsets": {}},
            "agent.disabled_toolsets.entry",
            {"agent.disabled_toolsets": {"entry": "new-value"}},
        ),
        (
            {"quick_commands": {"x": "list"}},
            "quick_commands.x.command",
            {"quick_commands": {"x": {"command": "new-value"}}},
        ),
    ],
)
def test_existing_containers_keep_their_type(config, key, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    set_config_value(key, "new-value")

    assert yaml.safe_load(config_path.read_text()) == expected
