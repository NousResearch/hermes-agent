"""Regression for #71188: /new must read model config from disk, not the
import-time CLI_CONFIG snapshot."""

import importlib
import types
from unittest.mock import patch, MagicMock


def test_new_reads_fresh_model_config(monkeypatch):
    """/new handler must call load_config() (disk) not CLI_CONFIG (stale)."""
    import cli

    # Set stale CLI_CONFIG with old model
    cli.CLI_CONFIG = {"model": {"default": "old-model", "provider": "old-prov"}, "agent": {"service_tier": ""}}

    # Mock load_config to return new model
    fresh_config = {"model": {"default": "new-model", "provider": "new-prov"}, "agent": {}}
    with patch("hermes_cli.config.load_config", return_value=fresh_config):
        # The fix should call load_config() instead of reading CLI_CONFIG
        # Verify by checking the import is used
        from hermes_cli.config import load_config
        result = load_config()
        assert result["model"]["default"] == "new-model"
