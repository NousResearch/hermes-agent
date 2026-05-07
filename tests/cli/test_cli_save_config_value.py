"""Tests for save_config_value() in cli.py — atomic write behavior."""

import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestSaveConfigValueAtomic:
    """save_config_value() must use atomic_yaml_write to avoid data loss."""

    @pytest.fixture
    def config_env(self, tmp_path, monkeypatch):
        """Isolated config environment with a writable config.yaml."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(yaml.dump({
            "model": {"default": "test-model", "provider": "openrouter"},
            "display": {"skin": "default"},
        }))
        monkeypatch.setattr("cli._hermes_home", hermes_home)
        return config_path

    def test_calls_atomic_yaml_write(self, config_env, monkeypatch):
        """save_config_value must route through atomic_yaml_write, not bare open()."""
        mock_atomic = MagicMock()
        monkeypatch.setattr("utils.atomic_yaml_write", mock_atomic)

        from cli import save_config_value
        save_config_value("display.skin", "mono")

        mock_atomic.assert_called_once()
        written_path, written_data = mock_atomic.call_args[0]
        assert Path(written_path) == config_env
        assert written_data["display"]["skin"] == "mono"

    def test_preserves_existing_keys(self, config_env):
        """Writing a new key must not clobber existing config entries."""
        from cli import save_config_value
        save_config_value("agent.max_turns", 50)

        result = yaml.safe_load(config_env.read_text())
        assert result["model"]["default"] == "test-model"
        assert result["model"]["provider"] == "openrouter"
        assert result["display"]["skin"] == "default"
        assert result["agent"]["max_turns"] == 50

    def test_creates_nested_keys(self, config_env):
        """Dot-separated paths create intermediate dicts as needed."""
        from cli import save_config_value
        save_config_value("auxiliary.compression.model", "google/gemini-3-flash-preview")

        result = yaml.safe_load(config_env.read_text())
        assert result["auxiliary"]["compression"]["model"] == "google/gemini-3-flash-preview"

    def test_overwrites_existing_value(self, config_env):
        """Updating an existing key replaces the value."""
        from cli import save_config_value
        save_config_value("display.skin", "ares")

        result = yaml.safe_load(config_env.read_text())
        assert result["display"]["skin"] == "ares"

    def test_preserves_env_ref_templates_in_unrelated_fields(self, config_env):
        """The /model --global persistence path must not inline env-backed secrets."""
        config_env.write_text(yaml.dump({
            "custom_providers": [{
                "name": "tuzi",
                "api_key": "${TU_ZI_API_KEY}",
                "model": "claude-opus-4-6",
            }],
            "model": {"default": "test-model", "provider": "openrouter"},
        }))

        from cli import save_config_value
        save_config_value("model.default", "doubao-pro")

        result = yaml.safe_load(config_env.read_text())
        assert result["model"]["default"] == "doubao-pro"
        assert result["custom_providers"][0]["api_key"] == "${TU_ZI_API_KEY}"

    def test_file_not_truncated_on_error(self, config_env, monkeypatch):
        """If atomic_yaml_write raises, the original file is untouched."""
        original_content = config_env.read_text()

        def exploding_write(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("utils.atomic_yaml_write", exploding_write)

        from cli import save_config_value
        result = save_config_value("display.skin", "broken")

        assert result is False
        assert config_env.read_text() == original_content

    def test_missing_user_config_writes_to_user_home_not_project_fallback(self, tmp_path, monkeypatch):
        """First-run writes should target ~/.hermes/config.yaml, not repo cli-config.yaml."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_config = project_dir / "cli-config.yaml"
        project_config.write_text(yaml.dump({
            "model": {"default": "fallback-model", "provider": "openrouter"},
            "display": {"skin": "default"},
        }))
        original_project = project_config.read_text()

        import cli
        monkeypatch.setattr(cli, "_hermes_home", hermes_home)
        monkeypatch.setattr(cli, "__file__", str(project_dir / "cli.py"))

        mock_atomic = MagicMock()
        monkeypatch.setattr("utils.atomic_yaml_write", mock_atomic)

        from cli import save_config_value
        result = save_config_value("display.skin", "ares")

        assert result is True
        written_path, written_data = mock_atomic.call_args[0]
        assert Path(written_path) == hermes_home / "config.yaml"
        assert written_data["model"]["default"] == "fallback-model"
        assert written_data["model"]["provider"] == "openrouter"
        assert written_data["display"]["skin"] == "ares"
        assert project_config.read_text() == original_project
