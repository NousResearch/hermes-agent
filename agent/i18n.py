"""Lightweight internationalization (i18n) for Hermes static user-facing messages.

Scope (thin slice, by design): only the highest-impact static strings shown
to the user by Hermes itself -- approval prompts, a handful of gateway slash
command replies, and restart-drain notices. Agent-generated output, log lines,
error tracebacks, and tool outputs stay outside this catalog; each UI owns its
additional presentation catalog and uses the same locale normalization contract.

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

Supported languages: en, zh, zh-hant, ja, de, es, fr, tr, uk, af, ko, it, ga, pt, ru, hu.  Unknown values fall back to en.
"""

from __future__ import annotations

import logging
import os
import sysconfig
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "ru", "hu",
)
DEFAULT_LANGUAGE = "en"

# Accept a few natural aliases so users who type "chinese" / "simplified-chinese" / "jp"
# get the right catalog instead of silently falling back to English.
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    # Chinese is handled as two explicit language choices here:
    # Simplified Chinese and Traditional Chinese.
    # Bare "chinese" / "mandarin"
    # also default to Simplified since that's the larger user base.
    "chinese": "zh", "mandarin": "zh", "simplified-chinese": "zh",
    # BCP-47 compatibility stays internal; user-facing choices remain
    # "Simplified Chinese" and "Traditional Chinese".
    "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh",
    "traditional-chinese": "zh-hant",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
    "japanese": "ja", "日本語": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de", "de-at": "de", "de-ch": "de",
    "spanish": "es", "español": "es", "espanol": "es", "es-es": "es", "es-mx": "es", "es-ar": "es",
    "french": "fr", "français": "fr", "francais": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "українська": "uk", "uk-ua": "uk", "ua": "uk",
    "turkish": "tr", "turkce": "tr", "türkçe": "tr", "tr-tr": "tr",
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
}

_catalog_cache: dict[str, dict[str, str]] = {}
_catalog_lock = threading.Lock()


def _locales_dir() -> Path:
    """Return the directory containing locale YAML files.

    Resolution order, first existing wins:

    1. ``HERMES_BUNDLED_LOCALES`` env var -- set by the Nix wrapper (or any
       sealed-packaging system) to point at the installed catalog directory.
    2. ``<repo-root>/locales`` -- source checkouts and ``pip install -e .``,
       where the working tree sits next to ``agent/``.
    3. ``<sysconfig data|purelib|platlib>/locales`` -- pip wheel installs.
       setuptools ``data-files`` extracts ``locales/*.yaml`` under the
       interpreter's ``data`` scheme; the other schemes are checked as a
       safety net for nonstandard layouts.

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
    if source_dir.is_dir():
        return source_dir

    # pip wheel install: data-files lands under the interpreter data scheme.
    # ``data`` (== sys.prefix in a venv) is where setuptools data-files extract
    # and is checked first. ``purelib``/``platlib`` (site-packages) are a safety
    # net for nonstandard layouts. NOTE: this does NOT cover ``pip install
    # --user`` (user scheme, ~/.local/locales) or ``pip install --target`` --
    # both are out of scope; see the plan header.
    for scheme in ("data", "purelib", "platlib"):
        raw = sysconfig.get_path(scheme)
        if not raw:
            continue
        candidate = Path(raw) / "locales"
        if candidate.is_dir():
            return candidate

    # Last resort: return the source-style path so _load_catalog's catalog-missing
    # log (logger.debug "i18n catalog missing for %s at %s") stays informative.
    return source_dir


def normalize_language(value: Any) -> str:
    """Normalize a user-supplied language value to a supported code.

    Accepts supported codes directly, common aliases (``chinese`` -> ``zh``),
    and explicit Chinese language choices (``simplified-chinese`` -> ``zh``).
    Returns the
    default language for unknown values.
    """
    if not isinstance(value, str):
        return DEFAULT_LANGUAGE
    key = "-".join(value.strip().lower().replace("_", "-").split())
    if not key:
        return DEFAULT_LANGUAGE
    if key in SUPPORTED_LANGUAGES:
        return key
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    # Chinese is limited to two explicit language choices here. Do not collapse
    # extra zh-* values to Simplified/Traditional.
    if key.startswith("zh-"):
        return DEFAULT_LANGUAGE

    # Try stripping a region suffix (e.g. "pt-br" -> "pt").
    base = key.split("-", 1)[0]
    if base in SUPPORTED_LANGUAGES:
        return base
    return DEFAULT_LANGUAGE


# Backward-compatible private name retained for existing internal tests and
# callers while cross-module consumers use the public boundary above.
_normalize_lang = normalize_language


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


def _configured_language() -> str | None:
    """Read the current ``display.language`` through the central config cache.

    ``load_config_readonly()`` owns the mtime-aware cache and avoids a deepcopy
    on cache hits. Keeping a second process-lifetime cache here would prevent a
    running gateway from observing Dashboard language changes.
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
    """Invalidate locale catalogs loaded by this process.

    Language configuration is already refreshed by the central mtime-aware
    config cache; this hook remains useful for tests and catalog updates.
    """
    with _catalog_lock:
        _catalog_cache.clear()


def get_language() -> str:
    """Resolve the active language using env > config > default order."""
    env_lang = os.environ.get("HERMES_LANGUAGE")
    if env_lang:
        return _normalize_lang(env_lang)
    cfg_lang = _configured_language()
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
    "normalize_language",
    "t",
    "get_language",
    "reset_language_cache",
]
