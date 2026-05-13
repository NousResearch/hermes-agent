"""Tests for Honcho client configuration."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from plugins.memory.honcho.client import HonchoClientConfig


class TestHonchoClientConfigAutoEnable:
    """Test auto-enable behavior when API key is present."""

    def test_auto_enables_when_api_key_present_no_explicit_enabled(self, tmp_path):
        """When API key exists and enabled is not set, should auto-enable."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            # Note: no "enabled" field
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is True  # Auto-enabled because API key exists

    def test_respects_explicit_enabled_false(self, tmp_path):
        """When enabled is explicitly False, should stay disabled even with API key."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            "enabled": False,  # Explicitly disabled
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is False  # Respects explicit setting

    def test_respects_explicit_enabled_true(self, tmp_path):
        """When enabled is explicitly True, should be enabled."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-api-key-12345",
            "enabled": True,
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "test-api-key-12345"
        assert cfg.enabled is True

    def test_disabled_when_no_api_key_and_no_explicit_enabled(self, tmp_path):
        """When no API key and enabled not set, should be disabled."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "workspace": "test",
            # No apiKey, no enabled
        }))

        # Clear env var if set
        env_key = os.environ.pop("HONCHO_API_KEY", None)
        try:
            cfg = HonchoClientConfig.from_global_config(config_path=config_path)
            assert cfg.api_key is None
            assert cfg.enabled is False  # No API key = not enabled
        finally:
            if env_key:
                os.environ["HONCHO_API_KEY"] = env_key

    def test_auto_enables_with_env_var_api_key(self, tmp_path, monkeypatch):
        """When API key is in env var (not config), should auto-enable."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "workspace": "test",
            # No apiKey in config
        }))

        monkeypatch.setenv("HONCHO_API_KEY", "env-api-key-67890")

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)

        assert cfg.api_key == "env-api-key-67890"
        assert cfg.enabled is True  # Auto-enabled from env var API key

    def test_from_env_always_enabled(self, monkeypatch):
        """from_env() should always set enabled=True."""
        monkeypatch.setenv("HONCHO_API_KEY", "env-test-key")

        cfg = HonchoClientConfig.from_env()

        assert cfg.api_key == "env-test-key"
        assert cfg.enabled is True

    def test_falls_back_to_env_when_no_config_file(self, tmp_path, monkeypatch):
        """When config file doesn't exist, should fall back to from_env()."""
        nonexistent = tmp_path / "nonexistent.json"
        monkeypatch.setenv("HONCHO_API_KEY", "fallback-key")

        cfg = HonchoClientConfig.from_global_config(config_path=nonexistent)

        assert cfg.api_key == "fallback-key"
        assert cfg.enabled is True  # from_env() sets enabled=True


import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
import sys
import plugins.memory.honcho.client as _honcho_mod


class TestGetHonchoClientToctouRace:
    """Regression: get_honcho_client() must construct exactly one Honcho instance
    even under concurrent load (double-checked locking guard)."""

    def setup_method(self):
        _honcho_mod._honcho_client = None

    def teardown_method(self):
        _honcho_mod._honcho_client = None

    def _fake_cfg(self):
        cfg = MagicMock()
        cfg.api_key = "test-key"
        cfg.base_url = None
        cfg.workspace_id = "ws-test"
        cfg.environment = "test"
        cfg.timeout = 30.0
        cfg.host = "app.honcho.dev"
        cfg.raw = {}
        return cfg

    def _fake_honcho_module(self, counter: list):
        class SpyHoncho:
            def __init__(self, **kwargs):
                counter.append(1)
        fake_mod = type(sys)("honcho")
        fake_mod.Honcho = SpyHoncho
        return fake_mod

    def test_concurrent_calls_return_same_instance(self):
        n = 50
        barrier = threading.Barrier(n)
        results: list = []
        counter: list = []
        cfg = self._fake_cfg()
        fake_mod = self._fake_honcho_module(counter)

        with patch.dict(sys.modules, {"honcho": fake_mod}):
            def worker():
                barrier.wait()
                results.append(_honcho_mod.get_honcho_client(cfg))

            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(lambda _: worker(), range(n)))

        assert len(results) == n
        assert all(r is results[0] for r in results), "multiple Honcho instances returned"

    def test_only_one_honcho_constructed(self):
        n = 50
        barrier = threading.Barrier(n)
        counter: list = []
        cfg = self._fake_cfg()
        fake_mod = self._fake_honcho_module(counter)

        with patch.dict(sys.modules, {"honcho": fake_mod}):
            def worker():
                barrier.wait()
                _honcho_mod.get_honcho_client(cfg)

            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(lambda _: worker(), range(n)))

        assert len(counter) == 1, f"Honcho constructed {len(counter)} times, expected 1"
