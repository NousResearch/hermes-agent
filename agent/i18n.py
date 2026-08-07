"""Lightweight internationalization (i18n) for Hermes static user-facing messages.

Scope (thin slice, by design): only the highest-impact static strings shown
to the user by Hermes itself -- approval prompts, a handful of gateway slash
command replies, restart-drain notices.  Agent-generated output, log lines,
error tracebacks, tool outputs, and slash-command descriptions all stay in
English.

Catalog files live under ``locales/`` at the repo root.  The primary catalog
for a language is ``<lang>.yaml`` (e.g. ``en.yaml``, ``zh.yaml``).  Each
catalog is a flat dict keyed by dotted paths (e.g. ``approval.choose`` or
``gateway.approval_expired``).  Missing keys fall back to English; if English
is missing too, the key path itself is returned so a broken catalog never
crashes the agent.

Register-aware variants live alongside the primary catalog as
``<lang>-<register>.yaml`` (e.g. ``en-friendly.yaml``).  A register is a
communication style (``"technical"``, ``"friendly"``) that translates the
*same information* into a different tone -- it does not suppress or replace
content.  Register is orthogonal to language: the language fallback chain
runs first, then the register overlay is applied on top.  Missing register
catalogs or missing keys fall back safely to the base catalog for that
language.

Usage::

    from agent.i18n import t
    print(t("approval.choose_long"))                       # current lang + register
    print(t("gateway.draining", count=3))                  # {count} formatted
    print(t("approval.choose_long", lang="zh"))            # explicit language
    print(t("gateway.reset.header_default", register="friendly"))  # friendly register

Language resolution order:
    1. Explicit ``lang=`` argument passed to :func:`t`
    2. ``HERMES_LANGUAGE`` environment variable (for tests / quick override)
    3. ``display.language`` from config.yaml
    4. ``"en"`` (baseline)

Register resolution order:
    1. Explicit ``register=`` argument passed to :func:`t`
    2. ``display.message_register`` from config.yaml
    3. ``"technical"`` (baseline — byte-compatible with pre-register output)

Supported languages: en, zh, zh-hant, ja, de, es, fr, tr, uk, af, ko, it, ga,
pt, ru, hu, ar.  Unknown values fall back to en.

Supported registers: ``"technical"`` (default), ``"friendly"``.  Unknown
values fall back to ``"technical"``.
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
DEFAULT_REGISTER = "technical"
SUPPORTED_REGISTERS: tuple[str, ...] = ("technical", "friendly")

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
    # Portuguese — bare "portuguese" routes to European Portuguese; pt-br
    # is in the same family but rendered identically here (no separate br catalog).
    "portuguese": "pt", "português": "pt", "portugues": "pt",
    "pt-pt": "pt", "pt-br": "pt", "brazilian": "pt", "brasileiro": "pt",
    # Russian
    "russian": "ru", "русский": "ru", "ru-ru": "ru",
    # Hungarian
    "hungarian": "hu", "magyar": "hu", "hu-hu": "hu",
    # Arabic — bare "arabic"/endonym plus the common regional BCP-47 tags.
    "arabic": "ar", "العربية": "ar",
    "ar-sa": "ar", "ar-eg": "ar", "ar-ae": "ar", "ar-ma": "ar", "ar-dz": "ar",
}

_catalog_cache: dict[str, dict[str, str]] = {}
_catalog_lock = threading.Lock()

# Separate cache for register-overlay catalogs (e.g. en-friendly.yaml).
# Keyed by ``"<lang>-<register>"`` so it doesn't collide with the base cache.
_register_catalog_cache: dict[str, dict[str, str]] = {}
_register_catalog_lock = threading.Lock()


def _locales_dir() -> Path:
    """Return the directory containing locale YAML files.

    Resolution order, first existing wins:

    1. ``HERMES_BUNDLED_LOCALES`` env var -- set by the Nix wrapper (or any
       sealed-packaging system) to point at the installed catalog directory.
    2. ``<repo-root>/locales`` -- source checkouts and editable installs,
       where the working tree sits next to ``agent/``.

    Falling through to the source-style path (even when missing) keeps
    ``_load_catalog`` error messages informative -- it logs the path it
    looked at -- rather than raising.
    """
    override = os.getenv("HERMES_BUNDLED_LOCALES", "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_dir():
            return candidate
        logger.warning(
            "HERMES_BUNDLED_LOCALES points to a non-directory path (%s); "
            "falling back to bundled/source locale resolution",
            override,
        )

    # agent/i18n.py -> agent/ -> repo root (source checkout, editable install)
    source_dir = Path(__file__).resolve().parent.parent / "locales"
    return source_dir


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


def _normalize_register(value: Any) -> str:
    """Normalize a user-supplied register value to a supported name.

    Returns the default register (``"technical"``) for unknown or empty
    values so a typo never crashes the agent.
    """
    if not isinstance(value, str):
        return DEFAULT_REGISTER
    key = value.strip().lower()
    if not key:
        return DEFAULT_REGISTER
    if key in SUPPORTED_REGISTERS:
        return key
    return DEFAULT_REGISTER


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


def _load_register_catalog(lang: str, register: str) -> dict[str, str]:
    """Load and flatten a register-overlay catalog (``<lang>-<register>.yaml``).

    Returns an empty dict if the file does not exist -- the caller falls
    back to the base catalog for that language.  Cached per lang+register
    for the process.
    """
    cache_key = f"{lang}-{register}"
    with _register_catalog_lock:
        cached = _register_catalog_cache.get(cache_key)
        if cached is not None:
            return cached

    path = _locales_dir() / f"{cache_key}.yaml"
    if not path.is_file():
        logger.debug("i18n register catalog missing for %s at %s", cache_key, path)
        with _register_catalog_lock:
            _register_catalog_cache[cache_key] = {}
        return {}

    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load i18n register catalog %s: %s", path, exc)
        with _register_catalog_lock:
            _register_catalog_cache[cache_key] = {}
        return {}

    flat: dict[str, str] = {}
    _flatten_into(raw, "", flat)
    with _register_catalog_lock:
        _register_catalog_cache[cache_key] = flat
    return flat


@lru_cache(maxsize=1)
def _config_language_cached() -> str | None:
    """Read ``display.language`` from config.yaml once per process.

    Cached because ``t()`` is called in hot paths (every approval prompt,
    every gateway reply) and re-reading YAML each call would be wasteful.
    ``reset_language_cache()`` clears this when config changes at runtime
    (e.g. after the setup wizard).
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        lang = (cfg.get("display") or {}).get("language")
        if lang:
            return _normalize_lang(lang)
    except Exception as exc:
        logger.debug("Could not read display.language from config: %s", exc)
    return None


def reset_language_cache() -> None:
    """Invalidate cached language/register resolution and catalogs.

    Call after :func:`hermes_cli.config.save_config` if a running process
    needs to pick up a changed ``display.language`` or
    ``display.message_register`` without restart.
    """
    _config_language_cached.cache_clear()
    _config_register_cached.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()
    with _register_catalog_lock:
        _register_catalog_cache.clear()


def get_language() -> str:
    """Resolve the active language using env > config > default order."""
    env_lang = os.environ.get("HERMES_LANGUAGE")
    if env_lang:
        return _normalize_lang(env_lang)
    cfg_lang = _config_language_cached()
    if cfg_lang:
        return cfg_lang
    return DEFAULT_LANGUAGE


@lru_cache(maxsize=1)
def _config_register_cached() -> str | None:
    """Read ``display.message_register`` from config.yaml once per process.

    Cached for the same reason as ``_config_language_cached()`` -- ``t()``
    is on the hot path.  ``reset_language_cache()`` clears this too.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        register = (cfg.get("display") or {}).get("message_register")
        if register:
            return _normalize_register(register)
    except Exception as exc:
        logger.debug("Could not read display.message_register from config: %s", exc)
    return None


def get_register() -> str:
    """Resolve the active register using config > default order."""
    cfg_register = _config_register_cached()
    if cfg_register:
        return cfg_register
    return DEFAULT_REGISTER


def t(key: str, lang: str | None = None, *, register: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted key to the active language and register.

    Parameters
    ----------
    key
        Dotted path into the catalog, e.g. ``"approval.choose_long"``.
    lang
        Explicit language override.  Takes precedence over env + config.
    register
        Explicit communication-register override (e.g. ``"friendly"``).
        Takes precedence over config.  When ``None`` or ``"technical"``,
        output is byte-compatible with pre-register behaviour.
    **format_kwargs
        ``str.format`` substitution arguments (``t("gateway.drain", count=3)``
        expects a catalog entry with a ``{count}`` placeholder).

    Returns
    -------
    The translated string.  Resolution order for the value:

    1. Register overlay for the target language (``<lang>-<register>.yaml``)
    2. Base catalog for the target language (``<lang>.yaml``)
    3. Register overlay for English (``en-<register>.yaml``)
    4. Base English catalog (``en.yaml``)
    5. The bare key (broken catalog fallback)

    When the resolved register is the default (``"technical"``), steps 1
    and 3 are skipped entirely so output is byte-compatible with
    pre-register behaviour.
    """
    target = _normalize_lang(lang) if lang else get_language()
    target_register = _normalize_register(register) if register is not None else get_register()
    use_register_overlay = target_register != DEFAULT_REGISTER

    value: str | None = None

    # Step 1: register overlay for the target language.
    if use_register_overlay:
        value = _load_register_catalog(target, target_register).get(key)

    # Step 2: base catalog for the target language.
    if value is None:
        value = _load_catalog(target).get(key)

    # Step 3: register overlay for English (language fallback).
    if value is None and target != DEFAULT_LANGUAGE and use_register_overlay:
        value = _load_register_catalog(DEFAULT_LANGUAGE, target_register).get(key)

    # Step 4: base English catalog (language fallback).
    if value is None and target != DEFAULT_LANGUAGE:
        value = _load_catalog(DEFAULT_LANGUAGE).get(key)

    # Step 5: last-ditch fallback to the bare key.
    if value is None:
        logger.debug("i18n miss: key=%r lang=%r register=%r", key, target, target_register)
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
    "SUPPORTED_REGISTERS",
    "DEFAULT_REGISTER",
    "t",
    "get_language",
    "get_register",
    "reset_language_cache",
]
