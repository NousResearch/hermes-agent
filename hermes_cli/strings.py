"""集中管理所有用户可见字符串。

所有字符串通过 i18n 系统翻译，支持运行时语言切换。
"""
from __future__ import annotations

from agent.i18n import t

# 缓存翻译结果（语言切换时清除）
_cache = {}
_current_lang = None


def _clear_cache():
    """清除翻译缓存（语言切换时调用）"""
    global _cache, _current_lang, WELCOME, HELP_HEADER, TIPS
    _cache.clear()
    _current_lang = None
    WELCOME = None
    HELP_HEADER = None
    TIPS = []


def _get(key, **kwargs):
    """获取翻译后的字符串（带缓存）"""
    from agent.i18n import get_language

    lang = get_language()
    global _current_lang
    if _current_lang != lang:
        _cache.clear()
        _current_lang = lang

    cache_key = "{}:{}".format(lang, key)
    if cache_key not in _cache:
        _cache[cache_key] = t(key, **kwargs)

    return _cache[cache_key]


# 欢迎消息
def get_welcome():
    return _get("cli.welcome")


def get_welcome_tip():
    return _get("cli.welcome_tip")


# 帮助文本
def get_help_header():
    return _get("cli.help.header")


def get_help_tip():
    return _get("cli.help.tip")


# 错误消息模板
def get_error_template():
    return _get("cli.error.template")


def get_error_invalid_input():
    return _get("cli.error.invalid_input")


def get_error_file_not_found():
    return _get("cli.error.file_not_found")


# 提示信息
def get_tips():
    """获取所有提示信息"""
    tips = []
    for i in range(1, 50):
        try:
            tip = _get("cli.tips.{}".format(i))
            if tip == "cli.tips.{}".format(i):
                break
            tips.append(tip)
        except Exception:
            break
    return tips


# 命令描述
def get_cmd_description(cmd):
    return _get("cli.cmd.{}".format(cmd))


# 模块级变量（兼容旧代码，模块加载时初始化）
WELCOME = get_welcome()
HELP_HEADER = get_help_header()
ERROR_TEMPLATE = get_error_template()
TIPS = get_tips()
