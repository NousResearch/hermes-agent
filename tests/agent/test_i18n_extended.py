"""Tests for extended i18n functionality."""
import pytest
from unittest.mock import patch, MagicMock
import locale


def test_get_system_locale_returns_detected_language():
    """Test that get_system_locale returns the detected system language."""
    from agent.i18n import get_system_locale

    with patch('agent.i18n.locale.getlocale', return_value=('zh_CN', 'UTF-8')):
        result = get_system_locale()
        assert result == 'zh'


def test_get_system_locale_fallback_to_default():
    """Test that get_system_locale falls back to default when detection fails."""
    from agent.i18n import get_system_locale

    with patch('agent.i18n.locale.getlocale', return_value=(None, None)):
        result = get_system_locale()
        assert result == 'en'


def test_set_language_changes_current_language():
    """Test that set_language changes the current language."""
    from agent.i18n import set_language, get_language, reset_language_cache

    reset_language_cache()
    try:
        set_language('zh')
        assert get_language() == 'zh'
    finally:
        # Reset for other tests
        set_language('en')
        reset_language_cache()


def test_get_system_locale_handles_exception():
    """Test that get_system_locale handles exceptions gracefully."""
    from agent.i18n import get_system_locale

    with patch('agent.i18n.locale.getlocale', side_effect=Exception("locale broken")):
        assert get_system_locale() == 'en'


def test_language_aliases():
    """Test that language aliases are correctly resolved."""
    from agent.i18n import _normalize_lang

    # Original aliases
    assert _normalize_lang('zh') == 'zh'
    assert _normalize_lang('ja') == 'ja'
    assert _normalize_lang('de') == 'de'
    assert _normalize_lang('es') == 'es'
    assert _normalize_lang('uk') == 'uk'
    assert _normalize_lang('tr') == 'tr'

    # New aliases
    assert _normalize_lang('zh-hant') == 'zh-hant'
    assert _normalize_lang('ko') == 'ko'
    assert _normalize_lang('it') == 'it'
    assert _normalize_lang('ga') == 'ga'
    assert _normalize_lang('pt') == 'pt'
    assert _normalize_lang('ru') == 'ru'
    assert _normalize_lang('hu') == 'hu'

    # Test case-insensitive and regional variants
    assert _normalize_lang('Korean') == 'ko'
    assert _normalize_lang('Italian') == 'it'
    assert _normalize_lang('Irish') == 'ga'
    assert _normalize_lang('Portuguese') == 'pt'
    assert _normalize_lang('Russian') == 'ru'
    assert _normalize_lang('Hungarian') == 'hu'
    assert _normalize_lang('Traditional-Chinese') == 'zh-hant'


def test_get_language_priority_order():
    """Test language priority: command > env > config > system > default."""
    import agent.i18n as i18n_mod
    from agent.i18n import get_language, set_language, reset_language_cache

    reset_language_cache()

    try:
        # Test command priority (highest)
        set_language('zh')
        assert get_language() == 'zh'

        # Test env priority - command should still take priority
        with patch.dict('os.environ', {'HERMES_LANGUAGE': 'ja'}):
            assert get_language() == 'zh'

        # Test env > config: when no command set, env should override config
        with patch.object(i18n_mod, '_current_language', None):
            reset_language_cache()
            with patch.dict('os.environ', {'HERMES_LANGUAGE': 'ja'}):
                with patch('agent.i18n._config_language_cached', return_value='de'):
                    assert get_language() == 'ja'

        # Test config > system: when no env, config should override system
        with patch.object(i18n_mod, '_current_language', None):
            reset_language_cache()
            with patch.dict('os.environ', {}, clear=True):
                with patch('agent.i18n._config_language_cached', return_value='de'):
                    with patch('agent.i18n.get_system_locale', return_value='fr'):
                        assert get_language() == 'de'

        # Test system > default: when no config, system should override default
        with patch.object(i18n_mod, '_current_language', None):
            reset_language_cache()
            with patch.dict('os.environ', {}, clear=True):
                with patch('agent.i18n._config_language_cached', return_value=None):
                    with patch('agent.i18n.get_system_locale', return_value='fr'):
                        assert get_language() == 'fr'

        # Test default fallback: when system returns default, should return 'en'
        with patch.object(i18n_mod, '_current_language', None):
            reset_language_cache()
            with patch.dict('os.environ', {}, clear=True):
                with patch('agent.i18n._config_language_cached', return_value=None):
                    with patch('agent.i18n.get_system_locale', return_value='en'):
                        assert get_language() == 'en'

    finally:
        i18n_mod._current_language = None
        reset_language_cache()
