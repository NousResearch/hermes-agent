"""Lightweight i18n for Hermes' static user-facing strings (approval prompts, gateway replies, CLI, tips).

Catalogs are flat dotted-key mappings resolved through layers, top first:

1. plugin language packs (``PluginContext.register_locale``; last registered wins) — ``agent.i18n_layers``
2. the user overlay ``$HERMES_HOME/locales/<lang>.yaml`` of the current profile home
3. the bundled ``locales/<lang>.yaml``
4. the same chain for ``en``
5. the bare key

Every layer may be partial. Language resolution: explicit ``lang=`` > ``HERMES_LANGUAGE`` >
``display.language`` > ``en``; any id that some layer supplies is accepted, so a pack-only language
(``pl``) works the moment its plugin loads. ``t()`` is a hot path: one cached merged dict per
``(home, lang)``, invalidated by :func:`reset_language_cache` (which every pack registration calls).
A bundled file is parsed once per process and reused by every home and after every reset, because
the pure-Python YAML parse takes hundreds of milliseconds and ``t()`` runs on the gateway's event loop.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent import i18n_layers
from agent.i18n_languages import language_options

logger = logging.getLogger(__name__)

# Bundled catalogs (compat: tests and the parity check iterate this). ``supported_languages()`` is the
# live set including overlay and pack languages.
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "ru", "hu", "ar",
)
DEFAULT_LANGUAGE = "en"

# Natural aliases so "chinese" / "zh-CN" / "jp" hit the right catalog instead of
# silently falling back to English. Bare "chinese" defaults to Simplified;
# Taiwan/HK/Macau tags route to the distinct Traditional catalog. pt-br shares
# the pt catalog unless a pack supplies a real pt-br one (a supplied id always wins over an alias).
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

# (home, lang) -> merged catalog (packs over overlay over bundled). home -> supported tuple.
_catalog_cache: dict[tuple[str, str], dict[str, str]] = {}
# (path, st_dev, st_ino, mtime_ns, size) -> one parsed bundled file. Not cleared by reset_language_cache.
_bundled_cache: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
_supported_cache: dict[str, tuple[str, ...]] = {}
_catalog_lock = threading.Lock()


def _locales_dir() -> Path:
    """Locale dir: ``HERMES_BUNDLED_LOCALES`` (sealed packaging, e.g. Nix) if it exists, else ``<repo-root>/locales``.

    The source path is returned even when missing so ``_load_bundled`` can log
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


def _current_home() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def supported_languages(home: str | None = None) -> tuple[str, ...]:
    """Every language some layer supplies for the current profile home: bundled ∪ user overlay ∪ plugin
    packs, ``en`` first then sorted. Cached until :func:`reset_language_cache`."""
    home = home or _current_home()
    with _catalog_lock:
        cached = _supported_cache.get(home)
        if cached is not None:
            return cached
    langs = set(SUPPORTED_LANGUAGES) | i18n_layers.layered_languages(home)
    result = (DEFAULT_LANGUAGE, *sorted(langs - {DEFAULT_LANGUAGE}))
    with _catalog_lock:
        _supported_cache[home] = result
    return result


def resolve_language_id(value: Any, home: str | None = None) -> str | None:
    """Canonical supported id for a user-supplied value (code, alias, regional tag), or ``None`` when no
    layer supplies it — the validation ``hermes config set display.language`` runs."""
    key = i18n_layers.normalize_language_id(value)
    if not key:
        return None
    supported = supported_languages(home)
    if key in supported:
        return key
    alias = _LANGUAGE_ALIASES.get(key)
    if alias in supported:
        return alias
    base = key.split("-", 1)[0]  # strip region suffix
    return base if base in supported else None


def _normalize_lang(value: Any, home: str | None = None) -> str:
    """Map a user-supplied value to a supported code (any layer), else the default."""
    return resolve_language_id(value, home) or DEFAULT_LANGUAGE


def _load_bundled(lang: str) -> dict[str, str]:
    """One bundled locale YAML flattened to dotted keys (empty dict on any failure; never crashes).
    The result is shared by every caller, so callers must not mutate it."""
    path = _locales_dir() / f"{lang}.yaml"
    if not path.is_file():
        logger.debug("i18n catalog missing for %s at %s", lang, path)
        return {}
    try:
        stat = path.stat()
        key = (str(path), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        with _catalog_lock:
            cached = _bundled_cache.get(key)
        if cached is not None:
            return cached
        parsed = i18n_layers.parse_locale_file(path)
    except Exception as exc:
        logger.warning("Failed to load i18n catalog %s: %s", path, exc)
        return {}
    with _catalog_lock:
        _bundled_cache[key] = parsed
    return parsed


def _flatten_into(node: Any, prefix: str, out: dict[str, str]) -> None:
    """Flatten a nested mapping into ``out`` (text leaves only)."""
    i18n_layers.flatten(node, prefix, out)


def _load_catalog(lang: str, home: str | None = None) -> dict[str, str]:
    """Merged catalog for ``lang`` in ``home`` (packs > overlay > bundled); cached per (home, lang)."""
    home = home or _current_home()
    key = (home, lang)
    with _catalog_lock:
        cached = _catalog_cache.get(key)
        if cached is not None:
            return cached
    merged = dict(_load_bundled(lang))
    merged.update(i18n_layers.overlay_layer(home, lang))
    merged.update(i18n_layers.pack_layer(lang))
    with _catalog_lock:
        _catalog_cache[key] = merged
    return merged


def surface_catalog(lang: str, surface: str = i18n_layers.CORE_SURFACE) -> dict[str, str]:
    """What ``i18n.catalog`` serves for one surface: packs > overlay > bundled ``locales/<lang>.<surface>.yaml``.

    The core surface omits the bundled layer (the Python renderer already has it); the TUI ships only an
    English catalog in-tree, so its bundled translations live in ``locales/<lang>.tui.yaml`` and ride
    along here."""
    lang = _normalize_lang(lang)
    merged: dict[str, str] = {}
    if surface != i18n_layers.CORE_SURFACE:
        merged.update(_load_bundled(f"{lang}.{surface}"))
    merged.update(i18n_layers.surface_catalog(_current_home(), lang, surface))
    return merged


@lru_cache(maxsize=8)
def _config_language_cached(hermes_home: str) -> str | None:
    """``display.language`` from config.yaml, read once per profile home (``t()`` is a hot path).
    Keyed by home so a multiplexed gateway serving several profiles doesn't freeze the first
    profile's language for every other profile."""
    try:
        from hermes_cli.config import load_config_readonly
        lang = (load_config_readonly().get("display") or {}).get("language")
        return _normalize_lang(lang, hermes_home) if lang else None
    except Exception as exc:
        logger.debug("Could not read display.language from config: %s", exc)
        return None


def _config_language() -> str | None:
    return _config_language_cached(_current_home())


def reset_language_cache() -> None:
    """Invalidate cached language resolution, merged catalogs and every layer view (call after
    ``save_config`` changes ``display.language``, after a pack registers/unregisters, or after editing
    an overlay file)."""
    _config_language_cached.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()
        _supported_cache.clear()
    i18n_layers.clear_cache()


def _resolve_language(home: str) -> str:
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        env_lang = get_secret("HERMES_LANGUAGE")
    except UnscopedSecretError:
        env_lang = os.environ.get("HERMES_LANGUAGE")  # unscoped default-profile path: environ IS its own value
    return _normalize_lang(env_lang, home) if env_lang else _config_language_cached(home) or DEFAULT_LANGUAGE


def get_language() -> str:
    """Resolve the active language using env > config > default order. ``HERMES_LANGUAGE`` is a
    per-profile ``.env`` value, so it is read through the secret scope: under multiplexing a raw
    environ read would impose the default profile's language on every other profile."""
    return _resolve_language(_current_home())


def warm_catalog() -> str:
    """Load the current profile's catalog for its active language and the English fallback; return the
    language.

    The gateway calls ``t()`` on its event loop. It runs this on a worker thread under each served
    profile's scope before that profile serves, so the loop finds the bundled files already parsed."""
    home = _current_home()
    lang = _resolve_language(home)
    _load_catalog(lang, home)
    if lang != DEFAULT_LANGUAGE:
        _load_catalog(DEFAULT_LANGUAGE, home)
    return lang


def t(key: str, lang: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted catalog key to the active (or explicit ``lang``) language.

    ``format_kwargs`` are applied with ``str.format``. Falls back to English,
    then to the bare key; a format failure returns the unformatted string.
    """
    home = _current_home()
    target = _normalize_lang(lang, home) if lang else _resolve_language(home)
    value = _load_catalog(target, home).get(key)
    if value is None and target != DEFAULT_LANGUAGE:
        value = _load_catalog(DEFAULT_LANGUAGE, home).get(key)
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


__all__ = [
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "get_language",
    "language_options",
    "reset_language_cache",
    "resolve_language_id",
    "supported_languages",
    "surface_catalog",
    "t",
    "warm_catalog",
]
