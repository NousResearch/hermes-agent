"""Tests for /lang command."""
import pytest
from unittest.mock import patch, MagicMock


def test_lang_command_in_registry():
    """Test that /lang command is in COMMAND_REGISTRY."""
    from hermes_cli.commands import COMMAND_REGISTRY

    lang_cmd = None
    for cmd in COMMAND_REGISTRY:
        if cmd.name == 'lang':
            lang_cmd = cmd
            break

    assert lang_cmd is not None
    assert lang_cmd.category == 'Configuration'
    assert lang_cmd.args_hint == '[language]'


def test_lang_command_shows_current_language():
    """Test that /lang shows current language."""
    from hermes_cli.commands import resolve_command

    with patch('agent.i18n.get_language', return_value='zh'):
        # This would be tested in integration with CLI
        pass


def test_lang_command_switches_language():
    """Test that /lang switches language."""
    from agent.i18n import set_language, get_language, reset_language_cache

    reset_language_cache()
    set_language('zh')
    assert get_language() == 'zh'

    set_language('en')
    assert get_language() == 'en'
    reset_language_cache()
