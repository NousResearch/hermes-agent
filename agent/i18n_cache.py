"""Disk cache of flattened bundled locale catalogs.

Parsing a bundled ``locales/<lang>.yaml`` with the pure-Python YAML loader costs hundreds of
milliseconds, and ``import cli`` pays it for command descriptions. The flattened catalog is cached as
JSON under ``<HERMES_HOME>/cache/i18n/``, keyed by the source file's resolved path and
``(mtime_ns, size)``; a mismatch, a missing entry or any read error re-parses. Writes are best-effort
and atomic, so concurrent processes race harmlessly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _cache_file(source: Path) -> Path:
    from hermes_constants import get_hermes_home
    digest = hashlib.sha256(str(source).encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return Path(get_hermes_home()) / "cache" / "i18n" / f"{source.name}.{digest}.json"


def _read(cache_file: Path, source: Path, stat_key: list[int]) -> dict[str, str] | None:
    try:
        with open(cache_file, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("source") != str(source) or data.get("stat") != stat_key:
        return None
    catalog = data.get("catalog")
    return catalog if isinstance(catalog, dict) else None


def _write(cache_file: Path, source: Path, stat_key: list[int], catalog: dict[str, str]) -> None:
    from hermes_constants import mkdir_under_hermes_home
    from utils import atomic_json_write
    try:
        mkdir_under_hermes_home(cache_file.parent)
        atomic_json_write(cache_file, {"source": str(source), "stat": stat_key, "catalog": catalog}, indent=0)
    except (OSError, ValueError) as exc:
        logger.debug("Could not write i18n catalog cache %s: %s", cache_file, exc)


def load_cached(path: Path, parse: Callable[[Path], dict[str, str]]) -> dict[str, str]:
    """``parse(path)``, served from the disk cache while the source's path, mtime and size are unchanged.
    Parse errors propagate; cache errors never do."""
    source = path.resolve()
    st = source.stat()
    stat_key = [st.st_mtime_ns, st.st_size]
    cache_file = _cache_file(source)
    cached = _read(cache_file, source, stat_key)
    if cached is not None:
        return cached
    catalog = parse(path)
    _write(cache_file, source, stat_key, catalog)
    return catalog
