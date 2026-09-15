"""Feishu per-group admission rules: ``~/.hermes/feishu_group_rules.json`` (mtime+size cached).

Hot file supplements/overrides boot ``config.extra.group_rules`` per chat_id. Missing
file → no overrides (boot config unchanged). Invalid/unreadable JSON → last
successfully loaded rules (or none if never loaded). Each field, especially
``require_mention``, falls back independently: a missing key means inherit, not false.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

RULES_FILENAME = "feishu_group_rules.json"


def rules_file() -> Path:
    """Path to the hot-reload group-rules file (resolved at call time)."""
    return get_hermes_home() / RULES_FILENAME


class _MtimeCache:
    """JSON file cache: ``stat()`` per access, re-read only when ``(mtime, size)`` changes.

    Size is part of the fingerprint so coarse-timestamp filesystems or same-second
    rewrites still reload. Fail-open: missing file → empty dict; invalid/unreadable
    → last good payload (or empty if nothing has loaded successfully yet).
    """

    def __init__(self, path: Optional[Path] = None):
        self._fixed_path = path
        self._mtime = 0.0
        self._size = -1
        self._data: Optional[dict] = None
        self._bound_path: Optional[Path] = None

    def _path(self) -> Path:
        return self._fixed_path if self._fixed_path is not None else rules_file()

    def _reset_bound(self, path: Path) -> None:
        self._mtime, self._size, self._data, self._bound_path = 0.0, -1, None, path

    def load(self) -> dict:
        path = self._path()
        if self._bound_path != path:
            self._reset_bound(path)
        try:
            st = path.stat()
        except FileNotFoundError:
            self._mtime, self._size, self._data = 0.0, -1, {}
            return {}
        mtime, size = st.st_mtime, st.st_size
        if mtime == self._mtime and size == self._size and self._data is not None:
            return self._data
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[Feishu-GroupRules] Failed to read %s, keeping last loaded rules: %s",
                path,
                exc,
            )
            self._mtime, self._size = mtime, size
            return self._data if self._data is not None else {}
        if not isinstance(data, dict):
            logger.warning(
                "[Feishu-GroupRules] %s is not a JSON object, keeping last loaded rules",
                path,
            )
            self._mtime, self._size = mtime, size
            return self._data if self._data is not None else {}
        self._mtime, self._size, self._data = mtime, size, data
        return self._data


_rules_cache = _MtimeCache()


def load_hot_group_rules() -> Dict[str, dict]:
    """Return per-chat rule dicts from the hot file (mtime-cached)."""
    raw = _rules_cache.load()
    rules = raw.get("group_rules")
    if not isinstance(rules, dict):
        return {}
    return {str(chat_id): rule for chat_id, rule in rules.items() if isinstance(rule, dict)}
