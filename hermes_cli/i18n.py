"""i18n — Lightweight translation framework for Hermes CLI.

Provides a single ``_()`` function that translates user-facing strings
based on ``display.language`` in the Hermes config.

Usage::

    from hermes_cli.i18n import _

    print_info(_("Hello, world!"))
    print(_("Settings:"), config_path)

Falls back to the original English string when no translation is available.
"""

import os
from typing import Dict, Optional

_TRANSLATION_CACHE: Optional[Dict[str, str]] = None
_CURRENT_LANG: Optional[str] = None


def _load_config_language() -> str:
    """Read ``display.language`` from config.yaml, defaulting to ``"en"``."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        lang = (config.get("display") or {}).get("language", "en")
        if not isinstance(lang, str) or not lang.strip():
            return "en"
        return lang.strip().lower()
    except Exception:
        return "en"


def _load_translations(lang: str) -> Dict[str, str]:
    """Load translation dict for a language, returning {} when not found.

    Supports ISO codes (``zh``, ``en``) — the corresponding module is
    ``hermes_cli.translations.<lang>``.
    """
    if lang in ("en", "english") or not lang:
        return {}
    try:
        import importlib
        mod = importlib.import_module(f"hermes_cli.translations.{lang}")
        table: Dict[str, str] = getattr(mod, "translations", {})
        if not isinstance(table, dict):
            return {}
        return table
    except (ImportError, ModuleNotFoundError, AttributeError):
        return {}


def _flush():
    """Force-reload language on next ``_()`` call (e.g. after config change)."""
    global _TRANSLATION_CACHE, _CURRENT_LANG
    _TRANSLATION_CACHE = None
    _CURRENT_LANG = None


def _(text: str) -> str:
    """Translate a single user-facing string.

    The language is read once from config on first call and cached.
    Call ``_flush()`` to re-read if config changes at runtime.
    Untranslated strings pass through as-is (English fallback).
    """
    global _TRANSLATION_CACHE, _CURRENT_LANG

    if _TRANSLATION_CACHE is None:
        _CURRENT_LANG = _load_config_language()
        _TRANSLATION_CACHE = _load_translations(_CURRENT_LANG)

    if not _TRANSLATION_CACHE:
        return text
    return _TRANSLATION_CACHE.get(text, text)


def current_language() -> str:
    """Return the active language code (e.g. ``"zh"``, ``"en"``)."""
    global _CURRENT_LANG
    if _CURRENT_LANG is None:
        _CURRENT_LANG = _load_config_language()
    return _CURRENT_LANG
