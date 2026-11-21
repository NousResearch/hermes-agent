"""MATRIX_REPLY_TO_MODE bridges onto PlatformConfig.reply_to_mode."""
import os
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides


class TestMatrixReplyToModeEnv:
    def _make_config(self):
        config = GatewayConfig()
        config.platforms[Platform.MATRIX] = PlatformConfig(enabled=True, token="test")
        return config

    def test_off_sets_matrix_reply_to_mode(self):
        config = self._make_config()
        with patch.dict(os.environ, {"MATRIX_REPLY_TO_MODE": "off"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].reply_to_mode == "off"

    def test_invalid_value_does_not_change_reply_to_mode(self):
        config = self._make_config()
        assert config.platforms[Platform.MATRIX].reply_to_mode == "first"
        with patch.dict(os.environ, {"MATRIX_REPLY_TO_MODE": "nope"}, clear=False):
            _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].reply_to_mode == "first"
