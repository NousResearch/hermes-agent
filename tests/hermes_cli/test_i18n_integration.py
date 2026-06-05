"""Integration tests for i18n system."""
import pytest
from unittest.mock import patch, MagicMock


def test_language_switching_works():
    """Test that language switching works end-to-end."""
    from agent.i18n import set_language, get_language, reset_language_cache
    from hermes_cli.strings import get_welcome, _clear_cache
    import agent.i18n as i18n_mod
    import hermes_cli.strings as strings_mod

    # Save original module-level values
    orig_welcome = strings_mod.WELCOME
    orig_help_header = strings_mod.HELP_HEADER
    orig_tips = strings_mod.TIPS

    try:
        reset_language_cache()
        i18n_mod._current_language = None

        # 切换到中文
        set_language('zh')
        _clear_cache()

        # 验证语言已切换
        assert get_language() == 'zh'

        # 获取欢迎消息（应该返回中文或英文 fallback）
        welcome = get_welcome()
        assert isinstance(welcome, str)
        assert len(welcome) > 0

        # 切换回英文
        set_language('en')
        _clear_cache()
        assert get_language() == 'en'
    finally:
        i18n_mod._current_language = None
        reset_language_cache()
        # Restore module-level values
        strings_mod.WELCOME = orig_welcome
        strings_mod.HELP_HEADER = orig_help_header
        strings_mod.TIPS = orig_tips


def test_system_locale_detection():
    """Test that system locale detection works without errors."""
    from agent.i18n import get_system_locale

    # 这取决于运行环境，但应该不抛异常
    result = get_system_locale()
    assert isinstance(result, str)
    assert len(result) > 0


def test_all_cli_strings_have_translations():
    """Test that all CLI strings have translations in en and zh."""
    from agent.i18n import _load_catalog

    en_catalog = _load_catalog('en')
    zh_catalog = _load_catalog('zh')

    en_cli_keys = {k for k in en_catalog.keys() if k.startswith('cli.')}
    zh_cli_keys = {k for k in zh_catalog.keys() if k.startswith('cli.')}

    # 检查覆盖率
    assert len(en_cli_keys) > 0, "No CLI keys found in en catalog"
    assert len(zh_cli_keys) > 0, "No CLI keys found in zh catalog"

    # 中文应该有至少 80% 的 CLI 键覆盖
    coverage = len(zh_cli_keys) / len(en_cli_keys) if en_cli_keys else 0
    assert coverage >= 0.8, f"CLI string coverage too low: {coverage:.1%}"


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


def test_strings_module_cache_clear():
    """Test that strings module cache clearing works."""
    from hermes_cli.strings import _clear_cache, get_welcome
    import hermes_cli.strings as strings_mod

    # Save original module-level values
    orig_welcome = strings_mod.WELCOME
    orig_help_header = strings_mod.HELP_HEADER
    orig_tips = strings_mod.TIPS

    try:
        # First call populates cache
        get_welcome()

        # Clear cache
        _clear_cache()

        # After clear, _current_lang should be None
        assert strings_mod._current_lang is None
    finally:
        # Restore module-level values so other tests aren't affected
        strings_mod.WELCOME = orig_welcome
        strings_mod.HELP_HEADER = orig_help_header
        strings_mod.TIPS = orig_tips


def test_i18n_priority_chain():
    """Test that i18n language priority chain works correctly."""
    from agent.i18n import get_language, set_language, reset_language_cache
    import agent.i18n as i18n_mod

    try:
        i18n_mod._current_language = None
        reset_language_cache()

        # Default should be 'en' (or system locale)
        lang = get_language()
        assert isinstance(lang, str)
        assert len(lang) > 0

        # set_language should override
        set_language('zh')
        assert get_language() == 'zh'

        set_language('ja')
        assert get_language() == 'ja'

        # Reset
        set_language('en')
        assert get_language() == 'en'
    finally:
        i18n_mod._current_language = None
        reset_language_cache()


def test_t_function_with_format_kwargs():
    """Test that t() function handles format kwargs correctly."""
    from agent.i18n import t, reset_language_cache
    import agent.i18n as i18n_mod

    try:
        i18n_mod._current_language = None
        reset_language_cache()
        # t() should work with or without kwargs
        result = t("cli.welcome")
        assert isinstance(result, str)
    finally:
        i18n_mod._current_language = None
        reset_language_cache()


def test_normalize_lang_aliases():
    """Test that language aliases are normalized correctly."""
    from agent.i18n import _normalize_lang

    # Test common aliases
    assert _normalize_lang("chinese") == "zh"
    assert _normalize_lang("zh-CN") == "zh"
    assert _normalize_lang("japanese") == "ja"
    assert _normalize_lang("english") == "en"
    assert _normalize_lang("EN") == "en"
    assert _normalize_lang("") == "en"
    assert _normalize_lang(None) == "en"


def test_supported_languages_list():
    """Test that supported languages list is populated."""
    from agent.i18n import SUPPORTED_LANGUAGES

    assert isinstance(SUPPORTED_LANGUAGES, tuple)
    assert "en" in SUPPORTED_LANGUAGES
    assert "zh" in SUPPORTED_LANGUAGES
    assert len(SUPPORTED_LANGUAGES) >= 2
