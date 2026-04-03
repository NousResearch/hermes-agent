"""YAML-driven selector loader for site-specific HTML parsers.

Loads regex patterns from ``selectors.yaml`` so that parsers remain
stable even when retailer HTML structures change — users only need
to update the YAML file.

Features:
    * Lazy-load + in-memory cache
    * Hot-reload when YAML file is modified
    * Graceful fallback to empty lists for missing keys
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_selectors_cache: Optional[Dict[str, Any]] = None
_cache_mtime: float = 0.0
_yaml_path: Optional[Path] = None


def _get_yaml_path() -> Path:
    """Return the path to ``selectors.yaml`` next to this package."""
    global _yaml_path
    if _yaml_path is None:
        _yaml_path = Path(__file__).resolve().parent.parent.parent / "selectors.yaml"
    return _yaml_path


def _load_yaml() -> Dict[str, Any]:
    """Load and parse the YAML file, with hot-reload support."""
    global _selectors_cache, _cache_mtime

    yaml_file = _get_yaml_path()

    if not yaml_file.exists():
        logger.warning("selectors.yaml not found at %s — using empty selectors", yaml_file)
        _selectors_cache = {}
        return _selectors_cache

    try:
        current_mtime = yaml_file.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    # Return cached version if file hasn't changed
    if _selectors_cache is not None and current_mtime == _cache_mtime:
        return _selectors_cache

    try:
        import yaml
    except ImportError:
        # PyYAML not available — fall back to basic parsing
        logger.warning("PyYAML not installed — selectors.yaml cannot be loaded")
        _selectors_cache = {}
        return _selectors_cache

    try:
        with open(yaml_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _selectors_cache = data
        _cache_mtime = current_mtime
        logger.debug("Loaded selectors.yaml (%d top-level keys)", len(data))
    except Exception as exc:
        logger.error("Failed to parse selectors.yaml: %s", exc)
        _selectors_cache = _selectors_cache or {}

    return _selectors_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_selectors(site: str, key: str) -> List[str]:
    """Return a list of regex patterns for *site* and *key*.

    Example::

        patterns = get_selectors("amazon", "name")
        # → ['id="productTitle"...', 'id="title"...', ...]

    Returns an empty list if the site or key is missing.
    """
    data = _load_yaml()
    site_data = data.get(site, {})
    value = site_data.get(key, [])

    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def get_selector(site: str, key: str) -> Optional[str]:
    """Return a single selector string for *site* and *key*.

    Useful for one-off patterns like ``price_whole_fraction``.
    Returns ``None`` if not found.
    """
    data = _load_yaml()
    site_data = data.get(site, {})
    value = site_data.get(key)

    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return value[0]
    return None


def get_site_config(site: str) -> Dict[str, Any]:
    """Return the full configuration dict for a given *site*.

    Returns an empty dict if the site is not defined.
    """
    data = _load_yaml()
    return data.get(site, {})


def get_domains(site: str) -> List[str]:
    """Return the list of domains for a site."""
    return get_selectors(site, "domains")


def get_stock_keywords(site: str) -> Dict[str, List[str]]:
    """Return stock keyword lists (out_of_stock, limited, in_stock)."""
    data = _load_yaml()
    site_data = data.get(site, {})
    keywords = site_data.get("stock_keywords", {})
    return {
        "out_of_stock": keywords.get("out_of_stock", []),
        "limited": keywords.get("limited", []),
        "in_stock": keywords.get("in_stock", []),
    }


def invalidate_cache() -> None:
    """Force the next ``get_selectors()`` call to reload from disk.

    Useful in tests or after programmatic edits to ``selectors.yaml``.
    """
    global _selectors_cache, _cache_mtime
    _selectors_cache = None
    _cache_mtime = 0.0
