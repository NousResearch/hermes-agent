"""Tests for Mem0 custom API URL support in PlatformBackend and config loading."""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_mem0_module():
    """Ensure 'mem0' is importable even when the package is not installed."""
    fake_mem0 = MagicMock()
    with patch.dict(sys.modules, {"mem0": fake_mem0}):
        yield fake_mem0


class TestPlatformBackendApiUrl:
    """PlatformBackend.__init__ passes the correct kwargs to MemoryClient."""

    def test_platform_backend_default_api_url(self, _mock_mem0_module):
        """When api_url is None, MemoryClient receives only api_key."""
        from plugins.memory.mem0._backend import PlatformBackend

        mock_client_cls = _mock_mem0_module.MemoryClient

        PlatformBackend("sk-test-key", api_url=None)

        mock_client_cls.assert_called_once_with(api_key="sk-test-key")

    def test_platform_backend_custom_api_url(self, _mock_mem0_module):
        """When a custom api_url is provided, MemoryClient receives host."""
        from plugins.memory.mem0._backend import PlatformBackend

        mock_client_cls = _mock_mem0_module.MemoryClient

        PlatformBackend("sk-test-key", api_url="https://custom.mem0.example.com")

        mock_client_cls.assert_called_once_with(
            api_key="sk-test-key", host="https://custom.mem0.example.com"
        )

    def test_platform_backend_empty_string_api_url(self, _mock_mem0_module):
        """When api_url is empty string, host is NOT passed (same as None)."""
        from plugins.memory.mem0._backend import PlatformBackend

        mock_client_cls = _mock_mem0_module.MemoryClient

        PlatformBackend("sk-test-key", api_url="")

        mock_client_cls.assert_called_once_with(api_key="sk-test-key")


class TestConfigLoadsApiUrl:
    """_load_config() reads MEM0_API_URL from environment."""

    def test_config_loads_api_url_from_env(self, tmp_path):
        """MEM0_API_URL env var is picked up by _load_config()."""
        env = {
            "MEM0_API_KEY": "sk-from-env",
            "MEM0_API_URL": "https://my-proxy.example.com",
        }
        with patch.dict("os.environ", env, clear=False), \
             patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            from plugins.memory.mem0 import _load_config

            config = _load_config()

        assert config["api_url"] == "https://my-proxy.example.com"
        assert config["api_key"] == "sk-from-env"
