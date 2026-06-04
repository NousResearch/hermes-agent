"""Tests for extended i18n functionality."""
import pytest
from unittest.mock import patch, MagicMock
import locale


def test_get_system_locale_returns_detected_language():
    """Test that get_system_locale returns the detected system language."""
    from agent.i18n import get_system_locale

    with patch('agent.i18n.locale.getdefaultlocale', return_value=('zh_CN', 'UTF-8')):
        result = get_system_locale()
        assert result == 'zh'


def test_get_system_locale_fallback_to_default():
    """Test that get_system_locale falls back to default when detection fails."""
    from agent.i18n import get_system_locale

    with patch('agent.i18n.locale.getdefaultlocale', return_value=(None, None)):
        result = get_system_locale()
        assert result == 'en'


def test_set_language_changes_current_language():
    """Test that set_language changes the current language."""
    from agent.i18n import set_language, get_language, reset_language_cache

    reset_language_cache()
    set_language('zh')
    assert get_language() == 'zh'

    # Reset for other tests
    set_language('en')
    reset_language_cache()


def test_get_language_priority_order():
    """Test language priority: command > env > config > system > default."""
    from agent.i18n import get_language, set_language, reset_language_cache

    reset_language_cache()

    # Test command priority (highest)
    set_language('zh')
    assert get_language() == 'zh'

    # Test env priority
    with patch.dict('os.environ', {'HERMES_LANGUAGE': 'ja'}):
        # Command should still take priority
        assert get_language() == 'zh'

    # Reset
    set_language('en')
    reset_language_cache()
