"""Per-plugin persistent storage: ``<rabbit home>/plugin-data/<name>/``.

Plugins must NOT park state in ``<rabbit home>/plugins/<name>/`` (the install dir, deleted by
``remove`` and git-pulled by ``update``). Secrets are deliberately NOT part of this convention —
credential reads go through ``agent.secret_scope`` / ``.env``.
Usage: ``plugin_data_dir("my-plugin") / "state.json"``; ``plugin_db("my-plugin")`` → ``data.db``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

__all__ = ["plugin_data_dir", "plugin_db"]

# Mirrors the plugin-name shape `rabbit plugins install` accepts (no separators/traversal).
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def _validate_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"invalid plugin name for storage: {name!r}")
    return name


def plugin_data_dir(name: str) -> Path:
    """Return (and create) ``<rabbit home>/plugin-data/<name>/``; resolves ``get_rabbit_home()`` on
    every call so it follows the active profile — don't cache across profile switches."""
    from rabbit_constants import get_rabbit_home
    root = get_rabbit_home() / "plugin-data" / _validate_name(name)
    root.mkdir(parents=True, exist_ok=True)
    return root


def plugin_db(name: str, filename: str = "data.db") -> sqlite3.Connection:
    """Open ``<data dir>/<filename>``. WAL so a dashboard reader and a tool writer coexist;
    ``check_same_thread=False`` for the threaded FastAPI/tool env — caller owns transactions."""
    if Path(filename).name != filename or not filename:
        raise ValueError(f"invalid plugin db filename: {filename!r}")
    from rabbit_cli.sqlite_util import open_db

    # WAL via the shared fallback helper: network filesystems degrade to DELETE and WAL-reset-bug
    # builds never enable it, instead of every plugin DB bypassing those rules with a raw PRAGMA.
    return open_db(plugin_data_dir(name) / filename, db_label=f"plugin-data/{name}/{filename}",
                   foreign_keys=True, row_factory=None, check_same_thread=False)
