"""Localized summaries for bundled skills without extending SKILL.md.

The Agent Skills specification reserves ``metadata`` for string-to-string data,
so locale maps cannot safely live in standards-compatible frontmatter. This
sidecar keeps SKILL.md portable while allowing presentation surfaces to request
one localized description at a time.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

_DATA_DIR = Path(__file__).with_name("data")
_ZH_HANT_ALIASES = frozenset({"zh-hant", "zh-tw", "zh-hk", "zh-mo"})


def normalize_skill_description_locale(locale: str | None) -> str:
    """Return a supported catalog locale, falling back to English."""
    normalized = str(locale or "").strip().lower().replace("_", "-")
    if normalized in _ZH_HANT_ALIASES or normalized.startswith("zh-hant-"):
        return "zh-hant"
    return "en"


@lru_cache(maxsize=4)
def _load_normalized_catalog(locale: str) -> Mapping[str, str]:
    if locale == "en":
        return {}
    path = _DATA_DIR / f"skill_descriptions.{locale}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): str(description)
        for name, description in raw.items()
        if isinstance(name, str) and isinstance(description, str) and description.strip()
    }


def load_skill_description_catalog(locale: str | None) -> Mapping[str, str]:
    """Load the sidecar for *locale*; aliases share one cached mapping."""
    return _load_normalized_catalog(normalize_skill_description_locale(locale))


load_skill_description_catalog.cache_clear = _load_normalized_catalog.cache_clear  # type: ignore[attr-defined]


def localize_skill_description(
    name: str,
    description: str,
    locale: str | None,
    *,
    bundled: bool,
) -> str:
    """Return one localized bundled-skill summary, otherwise the original."""
    if not bundled:
        return description
    return load_skill_description_catalog(locale).get(name, description)
