# ── Hermes Agent i18n ─────────────────────────────────────────────────────────
# Two-tier locale system for static messages:
#   1. SUPPORTED_LANGUAGES — canonical codes (en, zh, pt, etc.)
#   2. _LANGUAGE_ALIASES — user-friendly inputs ("chinese" → "zh", "pt-br" → "pt", etc.)
#
# Each language has a YAML file under locales/<code>.yaml (or locales/<code>.yml).
# Missing keys fall back to English.

import functools
import os
import re
from pathlib import Path
from typing import Optional

import yaml

# ── Static catalogue ───────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()

# The base locale directory ships inside the agent package. Users can override
# individual keys by placing YAML files of the same name under HERMES_HOME/locales/.
_BUILTIN_LOCALE_DIR = Path(__file__).resolve().parent.parent / "locales"

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "pt-BR", "ru", "hu",
)
DEFAULT_LANGUAGE = "en"

# Accept a few natural aliases so users who type "chinese" / "zh-CN" / "jp"
# get the right catalog instead of silently falling back to English.
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    # Simplified Chinese — explicit codes route here; bare "chinese" / "mandarin"
    # also default to Simplified since that's the larger user base.
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh",
    # Traditional Chinese — distinct catalog.  Cover Taiwan / Hong Kong / Macau
    # locale tags plus the common "traditional" alias.
    "traditional-chinese": "zh-hant", "traditional_chinese": "zh-hant",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
    "japanese": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de", "de-at": "de", "de-ch": "de",
    "spanish": "es", "español": "es", "espanol": "es", "es-es": "es", "es-mx": "es", "es-ar": "es",
    "french": "fr", "français": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "українська": "uk", "uk-ua": "uk", "ua": "uk",
    "turkish": "tr", "türkçe": "tr", "tr-tr": "tr",
    # Afrikaans — South African Dutch-derived language; "af-ZA" is the common BCP-47 tag.
    "afrikaans": "af", "af-za": "af",
    # Korean
    "korean": "ko", "한국어": "ko", "ko-kr": "ko",
    # Italian
    "italian": "it", "italiano": "it", "it-it": "it", "it-ch": "it",
    # Irish (Gaeilge) — ga is the BCP-47 code
    "irish": "ga", "gaeilge": "ga", "ga-ie": "ga",
    # Portuguese — "portuguese" routes to European Portuguese;
    # "pt-br" and "brazilian" route to Brazilian Portuguese catalog.
    "portuguese": "pt", "português": "pt", "portugues": "pt",
    "pt-pt": "pt", "pt-br": "pt-BR", "brazilian": "pt-BR", "brasileiro": "pt-BR",
    # Russian
    "russian": "ru", "русский": "ru", "ru-ru": "ru",
    # Hungarian
    "hungarian": "hu", "magyar": "hu", "hu-hu": "hu",
}


# ── Combined locale resolution (alias → catalog check) ────────────────────────

def _resolve_language_code(key: str) -> Optional[str]:
    """Normalise a user-supplied language key to a supported catalog code.

    Accepts canonical codes ("en", "zh"), aliases ("chinese", "zh-cn"), and
    display names ("English").  Returns None when the key is not recognised.
    """
    key = key.strip().lower().replace("_", "-")
    # 1. Exact canonical match
    if key in SUPPORTED_LANGUAGES:
        return key
    # 2. Alias lookup (case-insensitive; already lowercased)
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    # 3. Try matching display names (e.g. "English" → en, "中文" → zh)
    for code, aliases in _LANGUAGE_ALIASES.items():
        if isinstance(aliases, str) and code.replace("_", "-").lower() == key:
            return aliases
    return None


# ── Catalog loading ────────────────────────────────────────────────────────────

# Cache of per-language catalogs: {locale_code: dict}
_CATALOG_CACHE: dict[str, dict] = {}


def _load_catalog(lang: str) -> dict:
    """Load a locale catalog from the builtin directory, then overlay user overrides."""
    if lang in _CATALOG_CACHE:
        return _CATALOG_CACHE[lang]

    catalog: dict = {}

    # 1. Builtin (ships with the agent)
    builtin_path = _BUILTIN_LOCALE_DIR / f"{lang}.yaml"
    if builtin_path.is_file():
        with open(builtin_path, encoding="utf-8") as f:
            catalog = yaml.safe_load(f) or {}

    # 2. User overrides (HERMES_HOME/locales/<lang>.yaml)
    user_path = HERMES_HOME / "locales" / f"{lang}.yaml"
    if user_path.is_file():
        with open(user_path, encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
            _deep_merge(catalog, overrides)

    _CATALOG_CACHE[lang] = catalog
    return catalog


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge ``overrides`` into ``base`` (mutates base)."""
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


# ── Public API ─────────────────────────────────────────────────────────────────

# Sentinel for "not cached yet"
_T = object()


def t(key: str, *, lang: Optional[str] = None, fallback: Optional[str] = None) -> str:
    """Look up a localised string by dotted key.

    Args:
        key: Dotted path into the catalog, e.g. "approval.denied".
        lang: Target language code.  Falls back to config or env then English.
        fallback: Explicit fallback when neither the target language nor English
                  has the key.  If omitted, returns the raw key as a last resort.

    Returns:
        The localised string, or *fallback* if provided, or the raw *key*.
    """
    if lang is None:
        lang = _get_language_from_config_or_env()

    resolved = _resolve_language_code(lang) or "en"
    catalog = _load_catalog(resolved)

    parts = key.split(".")
    val: object = catalog
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part, _T)
        else:
            val = _T
            break

    if val is not _T and isinstance(val, str):
        return val

    # Fallback chain: English → raw key
    if resolved != "en":
        return t(key, lang="en", fallback=fallback)

    return fallback if fallback is not None else key


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_language_from_config_or_env() -> str:
    """Read ``display.language`` from config or ``HERMES_LANGUAGE`` env var."""
    try:
        from hermes_cli.config import load_hermes_config
        cfg = load_hermes_config()
        val = cfg.get("display", {}).get("language", "")
        if val and isinstance(val, str):
            return val
    except Exception:
        pass
    return os.environ.get("HERMES_LANGUAGE", "en")


def get_supported_languages() -> tuple[str, ...]:
    """Return the tuple of all supported language codes."""
    return SUPPORTED_LANGUAGES


def clear_catalog_cache() -> None:
    """Clear the in-memory catalog cache.

    Used after a config change (``display.language``) so the next ``t()`` call
    picks up the new catalog without a restart.
    """
    _CATALOG_CACHE.clear()
