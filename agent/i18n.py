"""Lightweight internationalization (i18n) for Hermes static user-facing messages.

Scope (thin slice, by design): only the highest-impact static strings shown
to the user by Hermes itself -- approval prompts, a handful of gateway slash
command replies, restart-drain notices.  Agent-generated output, log lines,
error tracebacks, tool outputs, and slash-command descriptions all stay in
English.

Catalog files live under ``locales/<lang>.yaml`` at the repo root.  Each
catalog is a flat dict keyed by dotted paths (e.g. ``approval.choose`` or
``gateway.approval_expired``).  Missing keys fall back to English; if English
is missing too, the key path itself is returned so a broken catalog never
crashes the agent.

Usage::

    from agent.i18n import t
    print(t("approval.choose_long"))                       # current lang
    print(t("gateway.draining", count=3))                  # {count} formatted
    print(t("approval.choose_long", lang="zh"))            # explicit override

Language resolution order:
    1. Explicit ``lang=`` argument passed to :func:`t`
    2. ``HERMES_LANGUAGE`` environment variable (for tests / quick override)
    3. ``display.language`` from config.yaml
    4. ``"en"`` (baseline)

Supported languages: en, zh, ja, de, es, fr, uk.  Unknown values fall back to en.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "zh", "ja", "de", "es", "fr", "uk")
DEFAULT_LANGUAGE = "en"

# Accept a few natural aliases so users who type "chinese" / "zh-CN" / "jp"
# get the right catalog instead of silently falling back to English.
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-tw": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "japanese": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de",
    "spanish": "es", "español": "es", "espanol": "es", "es-es": "es", "es-mx": "es",
    "french": "fr", "français": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "українська": "uk", "uk-ua": "uk", "ua": "uk",
}

_catalog_cache: dict[str, dict[str, str]] = {}
_catalog_lock = threading.Lock()


def _locales_dir() -> Path:
    """Return the directory containing locale YAML files.

    Lives next to the repo root so both the bundled install and editable
    checkouts find it without PYTHONPATH gymnastics.
    """
    # agent/i18n.py -> agent/ -> repo root
    return Path(__file__).resolve().parent.parent / "locales"


def _normalize_lang(value: Any) -> str:
    """Normalize a user-supplied language value to a supported code.

    Accepts supported codes directly, common aliases (``chinese`` -> ``zh``),
    and case-insensitive regional tags (``zh-CN`` -> ``zh``).  Returns the
    default language for unknown values.
    """
    if not isinstance(value, str):
        return DEFAULT_LANGUAGE
    key = value.strip().lower()
    if not key:
        return DEFAULT_LANGUAGE
    if key in SUPPORTED_LANGUAGES:
        return key
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    # Try stripping a region suffix (e.g. "pt-br" -> "pt" won't be supported,
    # but "zh-CN" -> "zh" will).
    base = key.split("-", 1)[0]
    if base in SUPPORTED_LANGUAGES:
        return base
    return DEFAULT_LANGUAGE


def _load_catalog(lang: str) -> dict[str, str]:
    """Load and flatten one locale YAML file into a dotted-key dict.

    YAML files can be nested for human readability; this produces the flat
    key space :func:`t` expects.  Cached per-language for the process.
    """
    with _catalog_lock:
        cached = _catalog_cache.get(lang)
        if cached is not None:
            return cached

    path = _locales_dir() / f"{lang}.yaml"
    if not path.is_file():
        logger.debug("i18n catalog missing for %s at %s", lang, path)
        with _catalog_lock:
            _catalog_cache[lang] = {}
        return {}

    try:
        import yaml  # PyYAML is already a hermes dependency
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load i18n catalog %s: %s", path, exc)
        with _catalog_lock:
            _catalog_cache[lang] = {}
        return {}

    flat: dict[str, str] = {}
    _flatten_into(raw, "", flat)
    with _catalog_lock:
        _catalog_cache[lang] = flat
    return flat


def _flatten_into(node: Any, prefix: str, out: dict[str, str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten_into(value, child_key, out)
    elif isinstance(node, str):
        out[prefix] = node
    # Non-string, non-dict leaves are ignored -- catalogs are text-only.


@lru_cache(maxsize=1)
def _config_language_cached() -> str | None:
    """Read ``display.language`` from config.yaml once per process.

    Cached because ``t()`` is called in hot paths (every approval prompt,
    every gateway reply) and re-reading YAML each call would be wasteful.
    ``reset_language_cache()`` clears this when config changes at runtime
    (e.g. after the setup wizard).
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        lang = (cfg.get("display") or {}).get("language")
        if lang:
            return _normalize_lang(lang)
    except Exception as exc:
        logger.debug("Could not read display.language from config: %s", exc)
    return None


def reset_language_cache() -> None:
    """Invalidate cached language resolution and catalogs.

    Call after :func:`hermes_cli.config.save_config` if a running process
    needs to pick up a changed ``display.language`` without restart.
    """
    _config_language_cached.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()


def get_language() -> str:
    """Resolve the active language using env > config > default order."""
    env_lang = os.environ.get("HERMES_LANGUAGE")
    if env_lang:
        return _normalize_lang(env_lang)
    cfg_lang = _config_language_cached()
    if cfg_lang:
        return cfg_lang
    return DEFAULT_LANGUAGE


def t(key: str, lang: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted key to the active language.

    Parameters
    ----------
    key
        Dotted path into the catalog, e.g. ``"approval.choose_long"``.
    lang
        Explicit language override.  Takes precedence over env + config.
    **format_kwargs
        ``str.format`` substitution arguments (``t("gateway.drain", count=3)``
        expects a catalog entry with a ``{count}`` placeholder).

    Returns
    -------
    The translated string, or the English fallback if the key is missing in
    the target language, or the bare key if English is also missing.
    """
    target = _normalize_lang(lang) if lang else get_language()
    catalog = _load_catalog(target)
    value = catalog.get(key)

    if value is None and target != DEFAULT_LANGUAGE:
        # Fall through to English rather than showing a key path to the user.
        value = _load_catalog(DEFAULT_LANGUAGE).get(key)

    if value is None:
        # Last-ditch: return the key itself.  A broken catalog should not
        # crash anything; it just looks ugly until someone fixes it.
        logger.debug("i18n miss: key=%r lang=%r", key, target)
        value = key

    if format_kwargs:
        try:
            return value.format(**format_kwargs)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning(
                "i18n format failed for key=%r lang=%r kwargs=%r: %s",
                key, target, format_kwargs, exc,
            )
            return value
    return value


__all__ = [
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "t",
    "get_language",
    "reset_language_cache",
]
