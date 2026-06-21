"""Tests for hermes_cli.doctor module."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_hermes_home():
    """Create a temporary HERMES_HOME directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hermes_home = Path(tmpdir) / "hermes_test"
        hermes_home.mkdir()
        (hermes_home / "sessions").mkdir()
        (hermes_home / "cron").mkdir()
        (hermes_home / "memories").mkdir()
        (hermes_home / "skills").mkdir()
        yield hermes_home


@pytest.fixture
def mock_config_file(temp_hermes_home):
    """Create a minimal config.yaml for testing."""
    config_path = temp_hermes_home / "config.yaml"
    config_path.write_text(
        """
model:
  provider: custom:litellm
  base_url: https://litellm.example.com/v1
  default: model-name
"""
    )
    return config_path


def test_probe_openrouter_custom_provider(mock_config_file, temp_hermes_home, monkeypatch):
    """Test that _probe_openrouter skips check for custom providers."""
    # Set up the test environment
    monkeypatch.setenv("HERMES_HOME", str(temp_hermes_home))
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("TZ", "UTC")
    
    # Import after setting env vars
    from hermes_cli.doctor import _probe_openrouter
    
    # Test with custom provider - should return empty result (no warning)
    result = _probe_openrouter()
    assert result.label == "OpenRouter API"
    # For custom providers, the result should have no issues (empty list)
    assert len(result.issues) == 0


def test_probe_openrouter_openrouter_provider(mock_config_file, temp_hermes_home, monkeypatch):
    """Test that _probe_openrouter checks when provider is openrouter."""
    # Set up the test environment with openrouter provider
    config_path = temp_hermes_home / "config.yaml"
    config_path.write_text(
        """
model:
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
  default: model-name
"""
    )
    monkeypatch.setenv("HERMES_HOME", str(temp_hermes_home))
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("TZ", "UTC")
    
    # Import after setting env vars
    from hermes_cli.doctor import _probe_openrouter
    
    # Test with openrouter provider - should check for API key
    result = _probe_openrouter()
    assert result.label == "OpenRouter API"
    # Without API key, should show warning
    assert len(result.issues) > 0


def test_probe_openrouter_auto_provider(mock_config_file, temp_hermes_home, monkeypatch):
    """Test that _probe_openrouter checks when provider is auto."""
    # Set up the test environment with auto provider
    config_path = temp_hermes_home / "config.yaml"
    config_path.write_text(
        """
model:
  provider: auto
  default: model-name
"""
    )
    monkeypatch.setenv("HERMES_HOME", str(temp_hermes_home))
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("TZ", "UTC")
    
    # Import after setting env vars
    from hermes_cli.doctor import _probe_openrouter
    
    # Test with auto provider - should check for API key
    result = _probe_openrouter()
    assert result.label == "OpenRouter API"


def test_probe_openrouter_empty_provider(mock_config_file, temp_hermes_home, monkeypatch):
    """Test that _probe_openrouter checks when provider is empty (default behavior)."""
    # Set up the test environment with no provider
    config_path = temp_hermes_home / "config.yaml"
    config_path.write_text(
        """
model:
  default: model-name
"""
    )
    monkeypatch.setenv("HERMES_HOME", str(temp_hermes_home))
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("TZ", "UTC")
    
    # Import after setting env vars
    from hermes_cli.doctor import _probe_openrouter
    
    # Test with empty provider - should check for API key (backward compatibility)
    result = _probe_openrouter()
    assert result.label == "OpenRouter API"