"""Static Skill shop categories loaded from ``category_map.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

_PKG_DIR = Path(__file__).resolve().parent
_MAP_PATH = _PKG_DIR / "category_map.yaml"


@lru_cache(maxsize=1)
def _raw_categories() -> List[Dict[str, Any]]:
    if not _MAP_PATH.is_file():
        return [
            {"code": "all", "name": "全部", "patterns": []},
        ]
    data = yaml.safe_load(_MAP_PATH.read_text(encoding="utf-8")) or {}
    rows = data.get("categories")
    if not isinstance(rows, list):
        return [{"code": "all", "name": "全部", "patterns": []}]
    return [r for r in rows if isinstance(r, dict) and r.get("code")]


def list_categories() -> List[Dict[str, str]]:
    """Return ``[{code, name}, ...]`` including ``all`` first."""
    out: List[Dict[str, str]] = []
    for row in _raw_categories():
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", code)).strip() or code
        if not code:
            continue
        out.append({"code": code, "name": name})
    if not out or out[0].get("code") != "all":
        out.insert(0, {"code": "all", "name": "全部"})
    return out


def normalize_category_key(value: str) -> str:
    """Normalize category codes/names from UI, Hub metadata, and local scans."""
    return "_".join(str(value or "").strip().lower().replace("&", " ").split())


def category_rules() -> List[Dict[str, Any]]:
    """Rows with ``patterns`` for tagging skills (excludes synthetic-only rows)."""
    return [r for r in _raw_categories() if r.get("code") not in (None, "", "all")]

