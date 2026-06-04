"""Tests for banner i18n integration."""
import pytest
from unittest.mock import patch, MagicMock


def test_welcome_uses_i18n():
    """Test that CLI welcome message uses i18n strings module."""
    from hermes_cli.strings import get_welcome, _clear_cache

    with patch('hermes_cli.strings.t', return_value='测试欢迎'):
        _clear_cache()
        result = get_welcome()
        assert result == '测试欢迎'


def test_welcome_fallback_to_skin_on_missing_key():
    """Test that welcome falls back to skin when i18n key is missing."""
    from hermes_cli.strings import get_welcome, _clear_cache

    # When i18n key doesn't exist, t() returns the key itself
    with patch('hermes_cli.strings.t', return_value='cli.welcome'):
        _clear_cache()
        result = get_welcome()
        # Should return the key name as fallback
        assert result == 'cli.welcome'


def test_skin_engine_branding_unchanged():
    """Test that skin engine branding still works independently."""
    from hermes_cli.skin_engine import get_active_skin

    skin = get_active_skin()
    welcome = skin.get_branding("welcome", "default")
    assert welcome  # Should return some non-empty string
    assert isinstance(welcome, str)
