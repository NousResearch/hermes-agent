"""Helpers for optional cheap-vs-strong model routing."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from utils import is_truthy_value

_COMPLEX_KEYWORDS = {
    "debug",
    "debugging",
    "implement",
    "implementation",
    "refactor",
    "patch",
    "traceback",
    "stacktrace",
    "exception",
    "error",
    "analyze",
    "analysis",
    "investigate",
    "architecture",
    "design",
    "compare",
    "benchmark",
    "optimize",
    "optimise",
    "review",
    "terminal",
    "shell",
    "tool",
    "tools",
    "pytest",
    "test",
    "tests",
    "plan",
    "planning",
    "delegate",
    "subagent",
    "cron",
    "docker",
    "kubernetes",
}

# CJK Unicode character ranges (Chinese, Japanese kanji, Korean hangul).
_CJK_CHAR_RE = re.compile(
    r"[\u4e00-\u9fff"    # CJK Unified Ideographs
    r"\u3400-\u4dbf"     # CJK Extension A
    r"\uf900-\ufaff"     # CJK Compatibility Ideographs
    r"\u3040-\u309f"     # Hiragana
    r"\u30a0-\u30ff"     # Katakana
    r"\uac00-\ud7af]"    # Hangul Syllables
)

# CJK task/action keywords that indicate the user needs the strong model.
# These are matched as substrings in the original text — CJK languages
# don't use spaces between words, so split()-based matching fails (#8516).
_CJK_COMPLEX_KEYWORDS = {
    # Task/action verbs (Chinese)
    "帮忙", "帮我", "研究", "分析", "调查", "排查", "检查", "诊断",
    "实现", "开发", "修复", "修改", "优化", "重构", "设计",
    # Memory/knowledge
    "记住", "记得", "记忆", "记录",
    # Search/lookup
    "查询", "搜索", "搜一下", "查一下",
    # Tool-invoking
    "设置", "配置", "安装", "部署", "定时", "提醒",
    # Technical
    "终端", "命令", "脚本", "代码", "接口",
    # Japanese task keywords
    "調べて", "分析して", "実装", "デバッグ", "検索",
    # Korean task keywords
    "분석", "검색", "구현", "디버그", "설치",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Conservative by design: if the message has signs of code/tool/debugging/
    long-form work, keep the primary model.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    cheap_model = cfg.get("cheap_model") or {}
    if not isinstance(cheap_model, dict):
        return None
    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 160)
    max_words = _coerce_int(cfg.get("max_simple_words"), 28)

    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None
    if text.count("\n") > 1:
        return None
    if "```" in text or "`" in text:
        return None
    if _URL_RE.search(text):
        return None

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return None

    # CJK languages don't use spaces between words, so split() produces a
    # single token for the entire message.  Use substring matching against
    # known CJK task/action keywords instead (#8516).
    if _CJK_CHAR_RE.search(text):
        for kw in _CJK_COMPLEX_KEYWORDS:
            if kw in text:
                return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    return route


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.
    """
    route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=route.get("base_url"),
        )
    except Exception:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
                "command": primary.get("command"),
                "args": list(primary.get("args") or []),
                "credential_pool": primary.get("credential_pool"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
                primary.get("command"),
                tuple(primary.get("args") or ()),
            ),
        }

    # Prefer cheap_model config values for api_mode, command, and args
    # over the runtime-resolved values.  resolve_runtime_provider reads
    # api_mode from the PRIMARY model config, not the cheap_model config,
    # so it can return the wrong value for the routed model (#8515).
    effective_api_mode = route.get("api_mode") or runtime.get("api_mode")
    effective_command = route.get("command") or runtime.get("command")
    effective_args = list(route.get("args") or runtime.get("args") or [])

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": effective_api_mode,
            "command": effective_command,
            "args": effective_args,
            "credential_pool": runtime.get("credential_pool"),
        },
        "label": f"smart route \u2192 {route.get('model')} ({runtime.get('provider')})",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            effective_api_mode,
            effective_command,
            tuple(effective_args),
        ),
    }
