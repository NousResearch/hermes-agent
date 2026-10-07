"""Tests for save_config_value() in cli.py — atomic write behavior."""

from pathlib import Path

import rabbit_yaml as yaml

import pytest


class TestSaveConfigValueAtomic:
    """save_config_value() must use atomic round-trip YAML updates."""

    @pytest.fixture
    def config_env(self, tmp_path, monkeypatch):
        """Isolated config environment with a writable config.yaml."""
        rabbit_home = tmp_path / ".rabbit"
        rabbit_home.mkdir()
        config_path = rabbit_home / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "model": {"default": "test-model", "provider": "openrouter"},
            "display": {"skin": "default"},
        }))
        # save_config_value resolves the target live via get_rabbit_home(), so
        # point RABBIT_HOME at the temp dir (the _rabbit_home import-time
        # constant is no longer consulted).
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        monkeypatch.setattr("cli._rabbit_home", rabbit_home)
        return config_path



    def test_creates_nested_keys(self, config_env):
        """Dot-separated paths create intermediate dicts as needed."""
        from cli import save_config_value
        save_config_value("auxiliary.compression.model", "google/gemini-3-flash-preview")

        result = yaml.safe_load(config_env.read_text())
        assert result["auxiliary"]["compression"]["model"] == "google/gemini-3-flash-preview"

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
    """Regression: persisted runtime settings must land in RABBIT_HOME/config.yaml
    (which config readers actually read), never the repo's cli-config.yaml.

    This was the "wake-word ear reverts to disabled after restart" bug: on an
    install whose RABBIT_HOME/config.yaml did not exist yet, save_config_value
    fell back to the checked-in cli-config.yaml. The toggle reported success, but
    startup read RABBIT_HOME/config.yaml and never saw the setting."""

    def test_creates_user_config_when_absent(self, tmp_path, monkeypatch):
        # Fresh RABBIT_HOME with NO config.yaml (managed/desktop first launch).
        rabbit_home = tmp_path / ".rabbit"
        rabbit_home.mkdir()
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))

        from cli import save_config_value

        assert save_config_value("wake_word.enabled", True) is True

        config_path = rabbit_home / "config.yaml"
        assert config_path.exists(), "user config.yaml must be created, not skipped"
        result = yaml.safe_load(config_path.read_text())
        assert result["wake_word"]["enabled"] is True

    def test_does_not_write_repo_cli_config(self, tmp_path, monkeypatch):
        # Even when the repo's cli-config.yaml exists, the write goes to the
        # user config, so a runtime setting is never buried in the shipped file.
        import cli as cli_module

        repo_cli_config = Path(cli_module.__file__).parent / "cli-config.yaml"
        before = repo_cli_config.read_text() if repo_cli_config.exists() else None

        rabbit_home = tmp_path / ".rabbit"
        rabbit_home.mkdir()
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))

        from cli import save_config_value

        save_config_value("wake_word.enabled", True)

        # The repo template is untouched…
        after = repo_cli_config.read_text() if repo_cli_config.exists() else None
        assert after == before
        # …and the value landed in the user config.
        result = yaml.safe_load((rabbit_home / "config.yaml").read_text())
        assert result["wake_word"]["enabled"] is True
