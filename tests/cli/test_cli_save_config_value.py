"""Tests for save_config_value() in cli.py — atomic write behavior."""

from pathlib import Path
from unittest.mock import MagicMock

import yaml

import pytest


class TestSaveConfigValueAtomic:
    """save_config_value() must use atomic round-trip YAML updates."""

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
        # save_config_value resolves the target live via get_hermes_home(), so
        # point HERMES_HOME at the temp dir (the _hermes_home import-time
        # constant is no longer consulted).
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr("cli._hermes_home", hermes_home)
        return config_path

    def test_calls_roundtrip_yaml_update(self, config_env, monkeypatch):
        """save_config_value must preserve user-edited YAML structure."""
        mock_update = MagicMock()
        monkeypatch.setattr("utils.atomic_roundtrip_yaml_update", mock_update)

        from cli import save_config_value
        save_config_value("display.skin", "mono")

        mock_update.assert_called_once_with(config_env, "display.skin", "mono")


    def test_creates_nested_keys(self, config_env):
        """Dot-separated paths create intermediate dicts as needed."""
        from cli import save_config_value
        save_config_value("auxiliary.compression.model", "google/gemini-3-flash-preview")

        result = yaml.safe_load(config_env.read_text())
        assert result["auxiliary"]["compression"]["model"] == "google/gemini-3-flash-preview"



    def test_model_write_runs_shared_cron_drift_warning(self, config_env, monkeypatch):
        warning = MagicMock()
        monkeypatch.setattr(
            "hermes_cli.config.warn_unpinned_cron_jobs_after_model_config_change",
            warning,
        )

        from cli import save_config_value

        assert save_config_value("model.default", "new-model") is True
        warning.assert_called_once_with("model.default", "new-model")



    def test_file_not_truncated_on_error(self, config_env, monkeypatch):
        """If atomic_yaml_write raises, the original file is untouched."""
        original_content = config_env.read_text()

        def exploding_write(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("utils.atomic_roundtrip_yaml_update", exploding_write)

        from cli import save_config_value
        result = save_config_value("display.skin", "broken")

        assert result is False
        assert config_env.read_text() == original_content


class TestSaveConfigValueTargetsUserConfig:
    """Regression: persisted runtime settings must land in HERMES_HOME/config.yaml
    (which config readers actually read), never the repo's cli-config.yaml.

    This was the "wake-word ear reverts to disabled after restart" bug: on an
    install whose HERMES_HOME/config.yaml did not exist yet, save_config_value
    fell back to the checked-in cli-config.yaml. The toggle reported success, but
    startup read HERMES_HOME/config.yaml and never saw the setting."""

    def test_creates_user_config_when_absent(self, tmp_path, monkeypatch):
        # Fresh HERMES_HOME with NO config.yaml (managed/desktop first launch).
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from cli import save_config_value

        assert save_config_value("wake_word.enabled", True) is True

        config_path = hermes_home / "config.yaml"
        assert config_path.exists(), "user config.yaml must be created, not skipped"
        result = yaml.safe_load(config_path.read_text())
        assert result["wake_word"]["enabled"] is True

    def test_does_not_write_repo_cli_config(self, tmp_path, monkeypatch):
        # Even when the repo's cli-config.yaml exists, the write goes to the
        # user config, so a runtime setting is never buried in the shipped file.
        import cli as cli_module

        repo_cli_config = Path(cli_module.__file__).parent / "cli-config.yaml"
        before = repo_cli_config.read_text() if repo_cli_config.exists() else None

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from cli import save_config_value

        save_config_value("wake_word.enabled", True)

        # The repo template is untouched…
        after = repo_cli_config.read_text() if repo_cli_config.exists() else None
        assert after == before
        # …and the value landed in the user config.
        result = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert result["wake_word"]["enabled"] is True
