"""Per-plugin persistent storage convention.

Plugins that want durable state today invent their own paths, and most of
them invent the same wrong one: a scratch directory inside
``<hermes home>/plugins/<name>/``. That tree is the plugin *install* dir —
``hermes plugins remove`` deletes it and ``hermes plugins update`` git-pulls
into it — so user data parked there dies with the code that wrote it.

This module is the sanctioned alternative: one data root per plugin under
``<hermes home>/plugin-data/<name>/``, owned by the user, untouched by
install/update/remove. Agent-built plugins get durable state without
inventing a storage story, and every plugin's data is inspectable in one
predictable place.

Secrets are deliberately NOT part of this convention — credential reads go
through ``agent.secret_scope`` / ``.env`` like everywhere else in Hermes.

Usage::

    from plugins.plugin_storage import plugin_data_dir, plugin_db

    state_file = plugin_data_dir("my-plugin") / "state.json"

    conn = plugin_db("my-plugin")             # <data dir>/data.db
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS ...")
        conn.commit()
    finally:
        conn.close()
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

__all__ = ["plugin_data_dir", "plugin_db"]

# Mirrors the plugin-name shape `hermes plugins install` accepts. Anything
# else could escape the data root via separators or traversal.
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def _validate_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"invalid plugin name for storage: {name!r}")
    return name


def plugin_data_dir(name: str) -> Path:
    """Return (and create) this plugin's durable data directory.

    ``<hermes home>/plugin-data/<name>/`` — survives plugin update and
    removal, and follows the active profile because it resolves through
    :func:`hermes_constants.get_hermes_home` on every call. Don't cache the
    result across profile switches.
    """
    from hermes_constants import get_hermes_home

    root = get_hermes_home() / "plugin-data" / _validate_name(name)
    root.mkdir(parents=True, exist_ok=True)
    return root


def plugin_db(name: str, filename: str = "data.db") -> sqlite3.Connection:
    """Open this plugin's SQLite database (created on first use).

    Lives at ``<data dir>/<filename>``. WAL mode so a dashboard reader and an
    agent-tool writer can coexist; ``check_same_thread=False`` matches the
    multi-threaded FastAPI/tool environment plugins actually run in — the
    caller still owns transaction discipline.
    """
    if Path(filename).name != filename or not filename:
        raise ValueError(f"invalid plugin db filename: {filename!r}")

    conn = sqlite3.connect(plugin_data_dir(name) / filename, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
