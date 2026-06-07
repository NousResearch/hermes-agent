"""Test that save_config preserves unknown/custom fields from the raw config file.

Regression test for #40821: config upgrade dropping custom_providers on first save.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

import pytest

# We'll need to test the save_config function directly


def test_save_config_preserves_custom_providers(tmp_path):
    """Custom_providers should be preserved when save_config is called after upgrade."""
    # Create a mock config directory structure
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_file = hermes_home / "config.yaml"
    
    # Write a config file with custom_providers (simulating 0.15.x user config)
    original_config = {
        "model": {
            "default": "gpt-4",
        },
        "custom_providers": [
            {
                "name": "local_llm",
                "base_url": "http://localhost:8000/v1",
                "models": {
                    "local-model": {
                        "context_length": 4096,
                    }
                }
            }
        ],
        "providers": {},
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(original_config, f)
    
    # Now patch the relevant functions to use our test config file
    from hermes_cli.config import save_config, read_raw_config, get_config_path
    
    with patch('hermes_cli.config.get_config_path', return_value=config_file):
        with patch('hermes_cli.config.ensure_hermes_home'):
            with patch('hermes_cli.config.is_managed', return_value=False):
                with patch('hermes_cli.config._secure_file'):
                    with patch('hermes_cli.config._SECURITY_COMMENT', ''):
                        with patch('hermes_cli.config._FALLBACK_COMMENT', ''):
                            # Simulate calling save_config with a normalized config
                            # (e.g., after load_config which merges defaults)
                            # The normalized config wouldn't have custom_providers
                            normalized_config = {
                                "model": {
                                    "default": "gpt-4",
                                },
                                "providers": {},
                                "agent": {"max_turns": 10},
                                # Note: custom_providers is NOT here, simulating normalization dropping it
                            }
                            
                            save_config(normalized_config)
    
    # Read back the saved config
    with open(config_file, 'r') as f:
        saved_config = yaml.safe_load(f)
    
    # Verify custom_providers was preserved
    assert "custom_providers" in saved_config, \
        "custom_providers was lost during save_config (regression #40821)"
    assert saved_config["custom_providers"] == original_config["custom_providers"]


def test_save_config_preserves_multiple_unknown_fields(tmp_path):
    """Multiple unknown fields should all be preserved."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_file = hermes_home / "config.yaml"
    
    # Config with several unknown/custom fields
    original_config = {
        "model": {
            "default": "gpt-4",
        },
        "custom_providers": [{"name": "local"}],
        "custom_field_1": "value1",
        "custom_field_2": {"nested": "value2"},
        "providers": {},
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(original_config, f)
    
    from hermes_cli.config import save_config
    
    with patch('hermes_cli.config.get_config_path', return_value=config_file):
        with patch('hermes_cli.config.ensure_hermes_home'):
            with patch('hermes_cli.config.is_managed', return_value=False):
                with patch('hermes_cli.config._secure_file'):
                    with patch('hermes_cli.config._SECURITY_COMMENT', ''):
                        with patch('hermes_cli.config._FALLBACK_COMMENT', ''):
                            # Normalized config without the custom fields
                            normalized_config = {
                                "model": {
                                    "default": "gpt-4",
                                },
                                "providers": {},
                                "agent": {"max_turns": 10},
                            }
                            
                            save_config(normalized_config)
    
    with open(config_file, 'r') as f:
        saved_config = yaml.safe_load(f)
    
    # All custom fields should be preserved
    assert saved_config["custom_providers"] == original_config["custom_providers"]
    assert saved_config["custom_field_1"] == "value1"
    assert saved_config["custom_field_2"] == {"nested": "value2"}


def test_save_config_does_not_preserve_normalized_keys(tmp_path):
    """Keys that are known to be normalized (provider, base_url, etc.) should not be re-added."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_file = hermes_home / "config.yaml"
    
    # Old-style config with root-level provider (should be normalized to model section)
    original_config = {
        "provider": "openai",  # This should be moved to model section by _normalize_root_model_keys
        "model": {
            "default": "gpt-4",
        },
        "custom_providers": [{"name": "local"}],
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(original_config, f)
    
    from hermes_cli.config import save_config
    
    with patch('hermes_cli.config.get_config_path', return_value=config_file):
        with patch('hermes_cli.config.ensure_hermes_home'):
            with patch('hermes_cli.config.is_managed', return_value=False):
                with patch('hermes_cli.config._secure_file'):
                    with patch('hermes_cli.config._SECURITY_COMMENT', ''):
                        with patch('hermes_cli.config._FALLBACK_COMMENT', ''):
                            normalized_config = {
                                "model": {
                                    "default": "gpt-4",
                                    "provider": "openai",  # Normalized into model section
                                },
                                "providers": {},
                                "agent": {"max_turns": 10},
                            }
                            
                            save_config(normalized_config)
    
    with open(config_file, 'r') as f:
        saved_config = yaml.safe_load(f)
    
    # custom_providers should be preserved
    assert "custom_providers" in saved_config
    # But root-level "provider" should NOT be re-added (it's a normalized key)
    assert "provider" not in saved_config, \
        "Root-level 'provider' should not be preserved (it's a normalized key)"
    # The provider should only be in the model section
    assert saved_config["model"].get("provider") == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
