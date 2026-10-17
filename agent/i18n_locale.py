"""Host-locale detection for the auto-detect form of ``display.language``.

An absent or empty ``display.language`` means auto-detect: match the control host's own locale
against the languages some layer actually supplies, so a Brazilian on Windows opens Hermes in pt-BR
without configuring anything. An explicit ``HERMES_LANGUAGE`` or ``display.language`` always wins.

This deliberately does not live in ``hermes_platform.host``: those facts are cached per process and
take **no environment-variable input**, while a POSIX locale *is* environment input. Precedence here
is the explicit environment first (``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``), then the platform API,
so a shell that sets ``LANG`` still gets what it asked for.

Detection never raises and never yields a language no layer supplies; a miss falls through to the
caller's own default (English).
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)

# posix-locale(7) precedence for the message locale.
_LOCALE_ENV = ("LC_ALL", "LC_MESSAGES", "LANG")
# gettext's "no localization" tags: they name no language to match.
_LOCALE_NONE = {"c", "posix"}


def _clean_tag(raw: str) -> str | None:
    """``pt_BR.UTF-8@latin`` -> ``pt-br``; ``None`` for empty, ``C`` or ``POSIX`` values."""
    tag = raw.split(".", 1)[0].split("@", 1)[0].strip().replace("_", "-").lower()
    if not tag or tag in _LOCALE_NONE or tag.split("-", 1)[0] in _LOCALE_NONE:
        return None
    return tag


def _env_locale() -> str | None:
    for name in _LOCALE_ENV:
        cleaned = _clean_tag(os.environ.get(name) or "")
        if cleaned:
            return cleaned
    return None


def _windows_locale() -> str | None:
    """``GetUserDefaultLocaleName`` — the user's own locale, not the system UI one."""
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(85)  # LOCALE_NAME_MAX_LENGTH
        if ctypes.windll.kernel32.GetUserDefaultLocaleName(buffer, len(buffer)):
            return _clean_tag(buffer.value or "")
    except Exception as exc:  # a missing or blocked API must not break startup
        logger.debug("GetUserDefaultLocaleName unavailable: %s", exc)
    return None


def _python_locale() -> str | None:
    try:
        import locale as _locale

        return _clean_tag(_locale.getlocale()[0] or "")
    except Exception as exc:
        logger.debug("locale.getlocale unavailable: %s", exc)
        return None


def os_locale() -> str | None:
    """The control host's locale as a lowercase BCP-47-style tag (``pt-br``), or ``None``."""
    return _env_locale() or (_windows_locale() if sys.platform == "win32" else _python_locale())


@lru_cache(maxsize=8)
def _detect_cached(hermes_home: str) -> str | None:
    """Host locale -> a supplied language id, cached per home (``t()`` is a hot path)."""
    tag = os_locale()
    if not tag:
        return None
    from agent.i18n import resolve_language_id

    resolved = resolve_language_id(tag, hermes_home)
    if resolved is None:
        logger.debug("i18n auto-detect: host locale %r matches no supplied language", tag)
    return resolved


def detect_language(hermes_home: str) -> str | None:
    """Language the host locale maps to when a layer supplies one, else ``None``."""
    return _detect_cached(hermes_home)


def reset_detection_cache() -> None:
    """Drop the cached host-locale match (called by :func:`agent.i18n.reset_language_cache`)."""
    _detect_cached.cache_clear()


__all__ = ["detect_language", "os_locale", "reset_detection_cache"]
