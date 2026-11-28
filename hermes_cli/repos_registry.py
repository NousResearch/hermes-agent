"""Derived registry of repo-backed installs (``hermes repos``).

The registry is a **cache, never the authority**. Every row is rebuilt from disk
by :func:`hermes_cli.repos_provenance.collect_installs` — the plugin install
sidecar, the skills manifest, the MCP config block, and git remotes under the
Hermes home. Deleting the file is safe; ``hermes repos sync`` recreates it.

Why a cache at all: the four surfaces disagree about vocabulary (``status``,
``source``, ``enabled``, ``trust``) and about where the repo URL lives, so a
single reader over the raw stores would be re-implemented by every caller. One
normalised shape here, one rebuild command, and the CLI reads that.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_write_text

REGISTRY_VERSION = 1
_REGISTRY_DIR = "state"
_REGISTRY_FILE = "installed-repos.json"


def registry_path() -> Path:
    """Absolute path of the registry file (never created here)."""
    return get_hermes_home() / _REGISTRY_DIR / _REGISTRY_FILE


def empty_registry() -> dict[str, Any]:
    """A fresh, valid registry document."""
    return {"version": REGISTRY_VERSION, "installs": [], "synced_at": None}


def read_registry() -> dict[str, Any]:
    """The registry document, or a fresh one when absent/corrupt/wrong-version.

    Corruption is not an error: the registry is derived, so an unreadable file
    reads as empty and the next ``sync`` repairs it.
    """
    path = registry_path()
    if not path.is_file():
        return empty_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty_registry()
    if not isinstance(data, dict):
        return empty_registry()
    installs = data.get("installs")
    if not isinstance(installs, list):
        return empty_registry()
    version = data.get("version")
    if version != REGISTRY_VERSION:
        return empty_registry()
    return {
        "version": REGISTRY_VERSION,
        "installs": installs,
        "synced_at": data.get("synced_at"),
    }


def write_registry(
    installs: list[dict[str, Any]], *, synced_at: str | None = None
) -> Path:
    """Atomically replace the registry with *installs* (best-effort by callers)."""
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": REGISTRY_VERSION,
        "installs": installs,
        "synced_at": synced_at or datetime.now(UTC).isoformat(),
    }
    atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def upsert(installs: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Replace the row with the same ``id``, or append it. In place."""
    for index, existing in enumerate(installs):
        if existing.get("id") == row.get("id"):
            installs[index] = row
            return
    installs.append(row)
