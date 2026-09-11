"""Lightweight i18n for Hermes' static user-facing strings (approval prompts, a few gateway replies).

Catalogs are ``locales/<lang>.yaml`` flattened to dotted keys. Missing keys
fall back to English, then to the key itself, so a broken catalog never crashes.
Language resolution: explicit ``lang=`` > ``HERMES_LANGUAGE`` > ``display.language`` > ``en``.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "ru", "hu", "ar",
)
DEFAULT_LANGUAGE = "en"

# Natural aliases so "chinese" / "zh-CN" / "jp" hit the right catalog instead of
# silently falling back to English. Bare "chinese" defaults to Simplified;
# Taiwan/HK/Macau tags route to the distinct Traditional catalog. pt-br shares
# the pt catalog (no separate br one).
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh",
    "traditional-chinese": "zh-hant", "traditional_chinese": "zh-hant",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
    "japanese": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de", "de-at": "de", "de-ch": "de",
    "spanish": "es", "español": "es", "espanol": "es", "es-es": "es", "es-mx": "es", "es-ar": "es",
    "french": "fr", "français": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "українська": "uk", "uk-ua": "uk", "ua": "uk",
    "turkish": "tr", "türkçe": "tr", "tr-tr": "tr",
    "afrikaans": "af", "af-za": "af",
    "korean": "ko", "한국어": "ko", "ko-kr": "ko",
    "italian": "it", "italiano": "it", "it-it": "it", "it-ch": "it",
    "irish": "ga", "gaeilge": "ga", "ga-ie": "ga",
    "portuguese": "pt", "português": "pt", "portugues": "pt",
    "pt-pt": "pt", "pt-br": "pt", "brazilian": "pt", "brasileiro": "pt",
    "russian": "ru", "русский": "ru", "ru-ru": "ru",
    "hungarian": "hu", "magyar": "hu", "hu-hu": "hu",
    "arabic": "ar", "العربية": "ar",
    "ar-sa": "ar", "ar-eg": "ar", "ar-ae": "ar", "ar-ma": "ar", "ar-dz": "ar",
}

_catalog_cache: dict[str, dict[str, str]] = {}
_catalog_lock = threading.Lock()


def _locales_dir() -> Path:
    """Locale dir: ``HERMES_BUNDLED_LOCALES`` (sealed packaging, e.g. Nix) if it exists, else ``<repo-root>/locales``.

    The source path is returned even when missing so ``_load_catalog`` can log
    the path it looked at rather than raise.
    """
    override = os.getenv("HERMES_BUNDLED_LOCALES", "").strip()
    if override and Path(override).is_dir():
        return Path(override)
    if override:
        logger.warning(
            "HERMES_BUNDLED_LOCALES points to a non-directory path (%s); "
            "falling back to bundled/source locale resolution", override,
        )
    return Path(__file__).resolve().parent.parent / "locales"


def _normalize_lang(value: Any) -> str:
    """Map a user-supplied value (code, alias, or regional tag like ``zh-CN``) to a supported code, else default."""
    key = value.strip().lower() if isinstance(value, str) else ""
    if key in SUPPORTED_LANGUAGES:
        return key
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    base = key.split("-", 1)[0]  # strip region suffix
    return base if base in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def _cache_catalog(lang: str, flat: dict[str, str]) -> dict[str, str]:
    with _catalog_lock:
        _catalog_cache[lang] = flat
    return flat


def _load_catalog(lang: str) -> dict[str, str]:
    """Load one locale YAML flattened to dotted keys; cached per language (empty dict on any failure)."""
    with _catalog_lock:
        cached = _catalog_cache.get(lang)
        if cached is not None:
            return cached

    path = _locales_dir() / f"{lang}.yaml"
    flat: dict[str, str] = {}
    if not path.is_file():
        logger.debug("i18n catalog missing for %s at %s", lang, path)
        return _cache_catalog(lang, flat)
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            _flatten_into(yaml.safe_load(f) or {}, "", flat)
    except Exception as exc:
        logger.warning("Failed to load i18n catalog %s: %s", path, exc)
        flat = {}
    return _cache_catalog(lang, flat)


def _flatten_into(node: Any, prefix: str, out: dict[str, str]) -> None:
    # Non-string, non-dict leaves are ignored -- catalogs are text-only.
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten_into(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, str):
        out[prefix] = node


@lru_cache(maxsize=1)
def _config_language_cached() -> str | None:
    """``display.language`` from config.yaml, read once per process (``t()`` is a hot path)."""
    try:
        from hermes_cli.config import load_config_readonly
        lang = (load_config_readonly().get("display") or {}).get("language")
        return _normalize_lang(lang) if lang else None
    except Exception as exc:
        logger.debug("Could not read display.language from config: %s", exc)
        return None


def reset_language_cache() -> None:
    """Invalidate cached language resolution and catalogs (call after ``save_config`` changes ``display.language``)."""
    _config_language_cached.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()


def get_language() -> str:
    """Resolve the active language using env > config > default order."""
    env_lang = os.environ.get("HERMES_LANGUAGE")
    return _normalize_lang(env_lang) if env_lang else _config_language_cached() or DEFAULT_LANGUAGE


def t(key: str, lang: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted catalog key to the active (or explicit ``lang``) language.

    ``format_kwargs`` are applied with ``str.format``. Falls back to English,
    then to the bare key; a format failure returns the unformatted string.
    """
    target = _normalize_lang(lang) if lang else get_language()
    value = _load_catalog(target).get(key)
    if value is None and target != DEFAULT_LANGUAGE:
        value = _load_catalog(DEFAULT_LANGUAGE).get(key)
    if value is None:
        logger.debug("i18n miss: key=%r lang=%r", key, target)
        value = key
    if not format_kwargs:
        return value
    try:
        return value.format(**format_kwargs)
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("i18n format failed for key=%r lang=%r kwargs=%r: %s", key, target, format_kwargs, exc)
        return value


__all__ = ["SUPPORTED_LANGUAGES", "DEFAULT_LANGUAGE", "t", "get_language", "reset_language_cache"]
