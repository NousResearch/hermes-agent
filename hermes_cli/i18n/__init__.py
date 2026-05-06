"""Gettext-based i18n loader for Hermes CLI.

Usage:
    from hermes_cli.i18n import setup_i18n, _

    setup_i18n("zh_CN")
    print(_("Hello world"))   # → "你好世界"

The translation files live alongside this module in locale subdirectories:

    hermes_cli/i18n/
        en/LC_MESSAGES/setup.po
        zh_CN/LC_MESSAGES/setup.po
        ja_JP/LC_MESSAGES/setup.po
        ...

If the .mo file is missing, gettext falls back to .po automatically
on most Python versions.  If both are absent, NullTranslations is used
and msgid (the original English string) is returned as-is.
"""

import gettext
import os
import logging

logger = logging.getLogger(__name__)

_LOCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
_DOMAIN = "setup"

_current_translations = gettext.NullTranslations()


def setup_i18n(language: str) -> None:
    """Initialize translations for the given locale code.

    Call this once at startup (e.g. at the top of run_setup_wizard).
    Falls back to English (msgid as-is) when the locale is unknown,
    empty, or when no translation file is found.
    """
    global _current_translations
    if not language or language == "en":
        _current_translations = gettext.NullTranslations()
        return

    try:
        _current_translations = gettext.translation(
            _DOMAIN,
            localedir=_LOCALE_DIR,
            languages=[language],
            fallback=True,  # silently returns NullTranslations on failure
        )
        # Verify translation file was actually loaded
        _current_translations.gettext("")
    except Exception as exc:
        logger.debug("i18n: failed to load '%s': %s — falling back to English", language, exc)
        _current_translations = gettext.NullTranslations()


def _(msg: str) -> str:
    """Translate a message string.  Falls back to msgid (English)."""
    return _current_translations.gettext(msg)


def ngettext(msgid1: str, msgid2: str, n: int) -> str:
    """Translate a plural form (for future use)."""
    return _current_translations.ngettext(msgid1, msgid2, n)
