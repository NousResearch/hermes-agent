"""Tests for hermes_cli configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

from hermes_cli.config import (
    DEFAULT_CONFIG,
    get_hermes_home,
    ensure_hermes_home,
    load_config,
    save_config,
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
            assert config["max_turns"] == DEFAULT_CONFIG["max_turns"]
            assert "terminal" in config
            assert config["terminal"]["backend"] == "local"


class TestSaveAndLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["model"] = "test/custom-model"
            config["max_turns"] = 42
            save_config(config)

            reloaded = load_config()
            assert reloaded["model"] == "test/custom-model"
            assert reloaded["max_turns"] == 42

    def test_nested_values_preserved(self, tmp_path):
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            config = load_config()
            config["terminal"]["timeout"] = 999
            save_config(config)

            reloaded = load_config()
            assert reloaded["terminal"]["timeout"] == 999
