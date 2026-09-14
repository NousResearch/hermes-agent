"""Static Python messages; UI catalogs own Dashboard and Ink presentation.

Language identities and aliases come from locales/registry.json. Catalogs
fall back directly to English. Python language resolution stays cached per
profile; UI display refresh does not invalidate the agent prompt or tools.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


def _load_locale_registry() -> dict[str, Any]:
    """Load and validate the cross-runtime language identity registry."""
    path = _locales_dir() / "registry.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to load locale registry {path}: {exc}") from exc

    locales = registry.get("locales")
    default = registry.get("default")
    if not isinstance(locales, dict) or not locales:
        raise RuntimeError(f"Locale registry {path} must define a non-empty locales object")
    if default not in locales:
        raise RuntimeError(f"Locale registry {path} default {default!r} is not registered")

    for section in ("aliases", "compatibilityAliases"):
        entries = registry.get(section)
        if not isinstance(entries, dict):
            raise RuntimeError(f"Locale registry {path} must define {section}")
        invalid = {key: value for key, value in entries.items() if value not in locales}
        if invalid:
            raise RuntimeError(f"Locale registry {path} has invalid {section}: {invalid}")
    return registry


_LOCALE_REGISTRY = _load_locale_registry()
SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(_LOCALE_REGISTRY["locales"])
DEFAULT_LANGUAGE: str = _LOCALE_REGISTRY["default"]
_LANGUAGE_ALIASES: dict[str, str] = dict(_LOCALE_REGISTRY["aliases"])

# External protocols and historical configuration may supply region-tagged
# locale values. Keep that compatibility isolated from the canonical product
# language registry and user-facing language choices.
_INTERNAL_COMPATIBILITY_ALIASES: dict[str, str] = dict(
    _LOCALE_REGISTRY["compatibilityAliases"]
)

_catalog_cache: dict[str, dict[str, str]] = {}
_catalog_lock = threading.Lock()


def normalize_language(value: Any) -> str:
    """Normalize a user-supplied language value to a supported code.

    Accepts supported codes directly plus registry-owned aliases and
    compatibility inputs. Primary-subtag fallback is allowed only when one
    registered product language owns that family; ambiguous families require
    an explicit registry mapping. Returns the default language for unknown
    values.
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
    if key in _INTERNAL_COMPATIBILITY_ALIASES:
        return _INTERNAL_COMPATIBILITY_ALIASES[key]
    # Strip a region suffix only when the registry has no sibling product pack
    # in the same language family. This stays language-neutral as new locales
    # are registered and prevents an arbitrary pack from becoming privileged.
    base = key.split("-", 1)[0]
    if base in SUPPORTED_LANGUAGES:
        has_sibling_pack = any(
            locale != base and locale.startswith(f"{base}-")
            for locale in SUPPORTED_LANGUAGES
        )
        if has_sibling_pack:
            return DEFAULT_LANGUAGE
        return base
    return DEFAULT_LANGUAGE


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
    # Non-string, non-dict leaves are ignored -- catalogs are text-only.


@lru_cache(maxsize=8)
def _config_language_cached(hermes_home: str) -> str | None:
    """``display.language`` from config.yaml, read once per profile home (``t()`` is a hot path).
    Keyed by home so a multiplexed gateway serving several profiles doesn't freeze the first
    profile's language for every other profile."""
    try:
        from hermes_cli.config import load_config_readonly
        lang = (load_config_readonly().get("display") or {}).get("language")
        return normalize_language(lang) if lang else None
    except Exception as exc:
        logger.debug("Could not read display.language from config: %s", exc)
        return None


def _config_language() -> str | None:
    from hermes_constants import get_hermes_home
    return _config_language_cached(str(get_hermes_home()))


def reset_language_cache() -> None:
    """Invalidate cached language resolution and locale catalogs.

    Call after a deliberate shared-Python language update or in tests. The
    Dashboard and Ink live-refresh paths do not depend on this cache.
    """
    _config_language_cached.cache_clear()
    with _catalog_lock:
        _catalog_cache.clear()


def get_language() -> str:
    """Resolve the active language using env > config > default order. ``HERMES_LANGUAGE`` is a
    per-profile ``.env`` value, so it is read through the secret scope: under multiplexing a raw
    environ read would impose the default profile's language on every other profile."""
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        env_lang = get_secret("HERMES_LANGUAGE")
    except UnscopedSecretError:
        env_lang = os.environ.get("HERMES_LANGUAGE")  # unscoped default-profile path: environ IS its own value
    return normalize_language(env_lang) if env_lang else _config_language() or DEFAULT_LANGUAGE


def t(key: str, lang: str | None = None, **format_kwargs: Any) -> str:
    """Translate a dotted catalog key to the active (or explicit ``lang``) language.

    ``format_kwargs`` are applied with ``str.format``. Falls back to English,
    then to the bare key; a format failure returns the unformatted string.
    """
    target = normalize_language(lang) if lang else get_language()
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


__all__ = [
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "normalize_language",
    "t",
    "get_language",
    "reset_language_cache",
]
