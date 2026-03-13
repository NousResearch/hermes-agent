"""Tests for hermes_cli configuration management."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml

from hermes_cli.config import (
    DEFAULT_CONFIG,
    get_hermes_home,
    ensure_hermes_home,
    load_config,
    load_env,
    save_config,
    save_env_value,
    save_env_value_secure,
)


class TestGetHermesHome:
    def test_default_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HOME", None)
            home = get_hermes_home()
            assert home == Path.home() / ".hermes"

    def test_env_override(self):
        with patch.dict(os.environ, {"HERMES_HOME": "/custom/path"}):
            home = get_hermes_home()
            assert home == Path("/custom/path")


class TestEnsureHermesHome:
    def test_creates_subdirs(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            ensure_hermes_home()
            assert (tmp_path / "cron").is_dir()
            assert (tmp_path / "sessions").is_dir()
            assert (tmp_path / "logs").is_dir()
            assert (tmp_path / "memories").is_dir()


class TestLoadConfigDefaults:
    def test_returns_defaults_when_no_file(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            assert config["model"] == DEFAULT_CONFIG["model"]
            assert config["agent"]["max_turns"] == DEFAULT_CONFIG["agent"]["max_turns"]
            assert "max_turns" not in config
            assert "terminal" in config
            assert config["terminal"]["backend"] == "local"

    def test_legacy_root_level_max_turns_migrates_to_agent_config(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config_path = tmp_path / "config.yaml"
            config_path.write_text("max_turns: 42\n")

            config = load_config()
            assert config["agent"]["max_turns"] == 42
            assert "max_turns" not in config


class TestSaveAndLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["model"] = "test/custom-model"
            config["agent"]["max_turns"] = 42
            save_config(config)

            reloaded = load_config()
            assert reloaded["model"] == "test/custom-model"
            assert reloaded["agent"]["max_turns"] == 42

            saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert saved["agent"]["max_turns"] == 42
            assert "max_turns" not in saved

    def test_save_config_normalizes_legacy_root_level_max_turns(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_config({"model": "test/custom-model", "max_turns": 37})

            saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
            assert saved["agent"]["max_turns"] == 37
            assert "max_turns" not in saved

    def test_nested_values_preserved(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["terminal"]["timeout"] = 999
            save_config(config)

            reloaded = load_config()
            assert reloaded["terminal"]["timeout"] == 999


class TestSaveEnvValueSecure:
    def test_save_env_value_writes_without_stdout(self, tmp_path, capsys):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("TENOR_API_KEY", "sk-test-secret")
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err == ""

            env_values = load_env()
            assert env_values["TENOR_API_KEY"] == "sk-test-secret"

    def test_secure_save_returns_metadata_only(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            result = save_env_value_secure("GITHUB_TOKEN", "ghp_test_secret")
            assert result == {
                "success": True,
                "stored_as": "GITHUB_TOKEN",
                "validated": False,
            }
            assert "secret" not in str(result).lower()

    def test_save_env_value_updates_process_environment(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
            os.environ.pop("TENOR_API_KEY", None)
            save_env_value("TENOR_API_KEY", "sk-test-secret")
            assert os.environ["TENOR_API_KEY"] == "sk-test-secret"

    def test_save_env_value_hardens_file_permissions_on_posix(self, tmp_path):
        if os.name == "nt":
            return

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            save_env_value("TENOR_API_KEY", "sk-test-secret")
            env_mode = (tmp_path / ".env").stat().st_mode & 0o777
            assert env_mode == 0o600


class TestSaveConfigAtomicity:
    """Verify save_config uses atomic writes (tempfile + os.replace)."""

    def test_no_partial_write_on_crash(self, tmp_path):
        """If save_config crashes mid-write, the previous file stays intact."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            # Write an initial config
            config = load_config()
            config["model"] = "original-model"
            save_config(config)

            config_path = tmp_path / "config.yaml"
            assert config_path.exists()

            # Simulate a crash during yaml.dump by making atomic_yaml_write's
            # yaml.dump raise after the temp file is created but before replace.
            with patch("utils.yaml.dump", side_effect=OSError("disk full")):
                try:
                    config["model"] = "should-not-persist"
                    save_config(config)
                except OSError:
                    pass

            # Original file must still be intact
            reloaded = load_config()
            assert reloaded["model"] == "original-model"

    def test_no_leftover_temp_files(self, tmp_path):
        """Failed writes must clean up their temp files."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            save_config(config)

            with patch("utils.yaml.dump", side_effect=OSError("disk full")):
                try:
                    save_config(config)
                except OSError:
                    pass

            # No .tmp files should remain
            tmp_files = list(tmp_path.glob(".*config*.tmp"))
            assert tmp_files == []

    def test_atomic_write_creates_valid_yaml(self, tmp_path):
        """The written file must be valid YAML matching the input."""
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["model"] = "test/atomic-model"
            config["agent"]["max_turns"] = 77
            save_config(config)

            # Read raw YAML to verify it's valid and correct
            config_path = tmp_path / "config.yaml"
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            assert raw["model"] == "test/atomic-model"
            assert raw["agent"]["max_turns"] == 77
