"""Tests for group/channel-scoped /model persistence."""

import yaml

from gateway.config import GatewayConfig, Platform
from gateway.config_loader import load_yaml_layer
from hermes_cli.model_switch import ModelSwitchResult, persist_channel_model_selection


def test_persist_channel_model_selection_writes_only_nonsecret_route(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  default: gpt-5.6-luna\n"
        "telegram:\n  allowed_chats: '-1000000000001'\n"
        "  channel_overrides:\n"
        "    '-111':\n"
        "      model: existing\n",
        encoding="utf-8",
    )
    result = ModelSwitchResult(
        success=True,
        new_model="channel/model",
        target_provider="opencode-go",
        api_key="secret-must-not-be-written",
        base_url="https://example.invalid",
        api_mode="chat_completions",
    )

    persist_channel_model_selection(result, config_path, platform="telegram", channel_id="-1000000000001")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["model"]["default"] == "gpt-5.6-luna"
    assert data["telegram"]["allowed_chats"] == "-1000000000001"
    assert data["telegram"]["channel_overrides"]["-111"]["model"] == "existing"
    assert data["telegram"]["channel_overrides"]["-1000000000001"] == {
        "model": "channel/model",
        "provider": "opencode-go",
        "enforce": True,
    }
    assert "secret-must-not-be-written" not in config_path.read_text(encoding="utf-8")


def test_persisted_nested_platform_route_round_trips_through_gateway_loader(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "gateway": {
                    "platforms": {
                        "telegram": {
                            "channel_overrides": {
                                "group-1": {
                                    "model": "old/model",
                                    "system_prompt": "Keep this prompt",
                                }
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = ModelSwitchResult(
        success=True,
        new_model="new/model",
        target_provider="openrouter",
        api_key="secret-must-not-be-written",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
    )

    persist_channel_model_selection(result, config_path, platform="telegram", channel_id="group-1")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["gateway"]["platforms"]["telegram"]["channel_overrides"]["group-1"] == {
        "model": "new/model",
        "provider": "openrouter",
        "system_prompt": "Keep this prompt",
        "enforce": True,
    }

    monkeypatch.setattr("gateway.config_loader.read_yaml_layers", lambda _home: raw)
    gateway_data = {}
    load_yaml_layer(tmp_path, gateway_data)
    config = GatewayConfig.from_dict(gateway_data)
    override = config.platforms[Platform.TELEGRAM].channel_overrides["group-1"]
    assert override.model == "new/model"
    assert override.provider == "openrouter"
    assert override.system_prompt == "Keep this prompt"
    assert override.enforce is True


def test_persist_channel_model_selection_updates_top_level_platforms_block(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "platforms": {
                    "telegram": {
                        "channel_overrides": {
                            "other-group": {"model": "existing/model"}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = ModelSwitchResult(
        success=True,
        new_model="new/model",
        target_provider="openrouter",
        api_key="secret-must-not-be-written",
    )

    persist_channel_model_selection(result, config_path, platform="telegram", channel_id="group-1")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["platforms"]["telegram"]["channel_overrides"]["group-1"] == {
        "model": "new/model",
        "provider": "openrouter",
        "enforce": True,
    }
    assert "telegram" not in data


def test_persist_channel_model_selection_strips_runtime_secrets_from_existing_entry(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "telegram": {
                    "channel_overrides": {
                        "group-1": {
                            "model": "old/model",
                            "system_prompt": "Keep this prompt",
                            "api_key": "preexisting-secret",
                            "base_url": "https://old.example/v1",
                            "api_mode": "chat_completions",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = ModelSwitchResult(
        success=True,
        new_model="new/model",
        target_provider="openrouter",
        api_key="new-secret-must-not-be-written",
    )

    persist_channel_model_selection(result, config_path, platform="telegram", channel_id="group-1")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["telegram"]["channel_overrides"]["group-1"] == {
        "model": "new/model",
        "provider": "openrouter",
        "system_prompt": "Keep this prompt",
        "enforce": True,
    }
    assert "preexisting-secret" not in config_path.read_text(encoding="utf-8")
