"""Tests for centralized string management."""
import pytest
from unittest.mock import patch


def test_strings_module_imports():
    """Test that strings module can be imported."""
    from hermes_cli.strings import WELCOME, HELP_HEADER, TIPS
    assert WELCOME is not None
    assert HELP_HEADER is not None
    assert isinstance(TIPS, list)


def test_strings_use_i18n():
    """Test that strings use i18n translations."""
    from hermes_cli.strings import WELCOME

    with patch('hermes_cli.strings.t', return_value='测试欢迎消息'):
        from hermes_cli import strings
        strings._clear_cache()
        result = strings.get_welcome()
        assert result == '测试欢迎消息'


def test_dynamic_strings_with_parameters():
    """Test that dynamic strings support parameters."""
    from hermes_cli.strings import ERROR_TEMPLATE

    with patch('hermes_cli.strings.t', return_value='错误：{error}'):
        from hermes_cli import strings
        strings._clear_cache()
        result = strings.get_error_template().format(error='文件未找到')
        assert result == '错误：文件未找到'
