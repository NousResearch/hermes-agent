import pytest
from unittest.mock import patch, mock_open
from hermes_cli.runtime_provider import _get_named_custom_provider


def test_get_named_custom_provider_finds_correct_provider():
    """Test that custom:or-api resolves to correct configuration"""
    
    # Mock config with custom provider
    mock_config = {
        "custom_providers": [
            {
                "name": "or-api",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-openrouter-key",
                "model": "anthropic/claude-opus-4.8",
                "key_env": "OR_API_KEY"
            },
            {
                "name": "test-provider",
                "base_url": "https://api.example.com/v1",
                "api_key": "test-key",
                "model": "test-model"
            }
        ]
    }
    
    with patch('hermes_cli.runtime_provider.load_config', return_value=mock_config):
        # Test that custom:or-api returns the correct provider
        result = _get_named_custom_provider("custom:or-api")
        
        assert result is not None, "Should find the custom:or-api provider"
        assert result["name"] == "or-api"
        assert result["base_url"] == "https://openrouter.ai/api/v1"
        assert result["api_key"] == "test-openrouter-key"
        assert result["model"] == "anthropic/claude-opus-4.8"
        assert result.get("key_env") == "OR_API_KEY"


def test_get_named_custom_provider_not_found():
    """Test that non-existent provider returns None"""
    
    mock_config = {
        "custom_providers": [
            {
                "name": "or-api",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
                "model": "test-model"
            }
        ]
    }
    
    with patch('hermes_cli.runtime_provider.load_config', return_value=mock_config):
        result = _get_named_custom_provider("custom:nonexistent")
        assert result is None


def test_get_named_custom_provider_with_provider_key():
    """Test provider with provider_key field"""
    
    mock_config = {
        "custom_providers": []
    }
    
    mock_compatible = [
        {
            "name": "test-provider",
            "provider_key": "tp-key",
            "base_url": "https://api.example.com/v1",
            "api_key": "test-key"
        }
    ]
    
    with patch('hermes_cli.runtime_provider.load_config', return_value=mock_config), \
         patch('hermes_cli.runtime_provider.get_compatible_custom_providers', return_value=mock_compatible):
        
        # Test matching by display name to get provider_key
        result = _get_named_custom_provider("custom:test-provider")
        
        assert result is not None
        assert result["name"] == "test-provider"
        assert result["provider_key"] == "tp-key"

        # Test matching by provider_key directly
        result_direct = _get_named_custom_provider("custom:tp-key")
        assert result_direct is not None
        assert result_direct["name"] == "test-provider"
        assert result_direct["provider_key"] == "tp-key"


def test_get_named_custom_provider_falls_through_when_not_found():
    """Ensure function returns None for non-matching providers (doesn't incorrectly match)"""
    
    mock_config = {
        "custom_providers": [
            {
                "name": "or-api",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
                "model": "anthropic/claude-opus-4.8"
            },
            {
                "name": "another-provider",
                "base_url": "https://api.another.com/v1",
                "api_key": "another-key",
                "model": "another-model"
            }
        ]
    }
    
    with patch('hermes_cli.runtime_provider.load_config', return_value=mock_config):
        # This should NOT match either provider
        result = _get_named_custom_provider("custom:wrong-name")
        assert result is None, "Should not match non-existent provider"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])