"""Lightweight localization helpers for Hermes Agent.

The project keeps technical identifiers in English, but user-facing prose can
be localized by loading language-specific YAML catalogs from ``locales/``.
The runtime language comes from:

1. ``HERMES_LANGUAGE`` env var, if present
2. ``display.language`` in ``config.yaml``
3. English default

This module intentionally stays tiny so CLI, gateway, and website code can all
share the same translation keys without pulling in a heavier framework.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_LANGUAGE = "en"
_LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
_LANGUAGE_OVERRIDE: str | None = None


def _normalize_language(language: Any) -> str:
    """Normalize locale tags to a stable short form."""
    if language is None:
        return _DEFAULT_LANGUAGE
    text = str(language).strip().lower()
    if not text:
        return _DEFAULT_LANGUAGE
    return text.replace("_", "-").split("-", 1)[0] or _DEFAULT_LANGUAGE


def set_language(language: str | None) -> None:
    """Override the active language for this process.

    Primarily useful for tests; production callers generally rely on env or
    config-based detection.
    """
    global _LANGUAGE_OVERRIDE
    _LANGUAGE_OVERRIDE = _normalize_language(language) if language else None


def get_language() -> str:
    """Return the active UI language."""
    if _LANGUAGE_OVERRIDE:
        return _LANGUAGE_OVERRIDE

    env_language = os.getenv("HERMES_LANGUAGE")
    if env_language:
        return _normalize_language(env_language)

    try:
        from hermes_cli.config import cfg_get, read_raw_config

        cfg = read_raw_config()
        language = cfg_get(cfg, "display", "language", default=None)
        return _normalize_language(language) if language else _DEFAULT_LANGUAGE
    except Exception:
        return _DEFAULT_LANGUAGE


@lru_cache(maxsize=16)
def _load_catalog(language: str) -> dict[str, Any]:
    """Load a locale YAML file into a nested dictionary."""
    catalog_path = _LOCALES_DIR / f"{language}.yaml"
    if not catalog_path.exists():
        return {}
    try:
        data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _lookup(catalog: dict[str, Any], key: str) -> Any:
    node: Any = catalog
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def t(key: str, *, default: str | None = None, language: str | None = None, **kwargs: Any) -> str:
    """Translate a dotted key with optional formatting.

    Fallback order:
      1. Requested language catalog
      2. English catalog, if present
      3. ``default`` if provided
      4. The key itself
    """
    resolved_language = _normalize_language(language or get_language())
    catalogs = [_load_catalog(resolved_language)]
    if resolved_language != _DEFAULT_LANGUAGE:
        catalogs.append(_load_catalog(_DEFAULT_LANGUAGE))

    value: Any = None
    for catalog in catalogs:
        value = _lookup(catalog, key)
        if isinstance(value, str):
            break
        value = None

    if value is None:
        value = default if default is not None else key

    if kwargs:
        try:
            value = str(value).format(**kwargs)
        except Exception:
            value = str(value)
    else:
        value = str(value)
    return value


def pluralize(
    count: int,
    one: str,
    few: str,
    many: str,
    *,
    language: str | None = None,
) -> str:
    """Return the grammatically appropriate noun form for ``count``.

    The helper keeps Russian plural forms readable without bringing in a
    heavier i18n framework. For non-Russian locales we fall back to the
    English-friendly singular/plural split.
    """
    resolved_language = _normalize_language(language or get_language())
    if resolved_language != "ru":
        return one if abs(int(count)) == 1 else many

    n = abs(int(count))
    mod100 = n % 100
    mod10 = n % 10
    if 11 <= mod100 <= 14:
        return many
    if mod10 == 1:
        return one
    if 2 <= mod10 <= 4:
        return few
    return many
