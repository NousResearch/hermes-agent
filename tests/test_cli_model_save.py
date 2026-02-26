"""Regression tests for CLI config persistence helpers."""

from pathlib import Path

import yaml

import cli


def test_save_config_value_updates_active_profile_when_setting_model_default(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".hermes"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "legacy-old",
                    "provider": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                    "profiles": [
                        {
                            "name": "work",
                            "provider": "zai",
                            "model": "glm-old",
                            "base_url": "https://profile.example/v1",
                            "enabled": True,
                        },
                        {
                            "name": "fallback",
                            "provider": "openrouter",
                            "model": "anthropic/claude-opus-4.6",
                            "base_url": "https://openrouter.ai/api/v1",
                            "enabled": True,
                        },
                    ],
                    "active_profile": "work",
                    "scoped_profiles": ["work", "fallback"],
                }
            },
            sort_keys=False,
        )
    )

    monkeypatch.setattr(Path, "home", lambda: fake_home)

    assert cli.save_config_value("model.default", "glm-new") is True

    persisted = yaml.safe_load(config_path.read_text())
    model_cfg = persisted["model"]

    active = next(p for p in model_cfg["profiles"] if p["name"] == model_cfg["active_profile"])
    assert active["name"] == "work"
    assert active["model"] == "glm-new"
    assert model_cfg["default"] == "glm-new"
    assert model_cfg["provider"] == "zai"
    assert model_cfg["base_url"] == "https://profile.example/v1"
