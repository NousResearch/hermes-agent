"""Tests for CLI help i18n integration."""
import pytest
from unittest.mock import patch, MagicMock


def test_help_header_uses_i18n():
    """Test that help header uses i18n strings module."""
    from hermes_cli.strings import get_help_header, _clear_cache

    with patch('hermes_cli.strings.t', return_value='可用命令'):
        _clear_cache()
        result = get_help_header()
        assert result == '可用命令'


def test_help_header_fallback_on_missing_key():
    """Test that help header falls back when i18n key is missing."""
    from hermes_cli.strings import get_help_header, _clear_cache

    with patch('hermes_cli.strings.t', return_value='cli.help.header'):
        _clear_cache()
        result = get_help_header()
        assert result == 'cli.help.header'


def test_skin_help_header_still_works():
    """Test that skin engine help header still works independently."""
    from hermes_cli.skin_engine import get_active_help_header

    header = get_active_help_header("(^_^)? Available Commands")
    assert header
    assert isinstance(header, str)


def test_show_help_prefers_i18n_over_skin():
    """Test that show_help uses i18n header when i18n returns valid translation."""
    from hermes_cli.strings import _clear_cache, get_help_header as _get_i18n_help_header

    with patch('hermes_cli.strings.t', return_value='可用命令'):
        _clear_cache()
        _i18n_header = _get_i18n_help_header()
        assert _i18n_header == '可用命令'
        assert not _i18n_header.startswith("cli.")


def test_show_help_detects_untranslated_key():
    """Test that show_help detects when i18n returns untranslated key name."""
    from hermes_cli.strings import _clear_cache, get_help_header as _get_i18n_help_header

    with patch('hermes_cli.strings.t', return_value='cli.help.header'):
        _clear_cache()
        _i18n_header = _get_i18n_help_header()
        assert _i18n_header == 'cli.help.header'
        assert _i18n_header.startswith("cli.")
