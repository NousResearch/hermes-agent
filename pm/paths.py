"""Where the store, lockfile, and installed-state file live."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def lockfile_path() -> Path:
    return Path(__file__).resolve().parent / "lock.json"


@lru_cache(maxsize=1)
def _stamp() -> dict:
    for parent in (repo_root(), *repo_root().parents):
        stamp = parent / "install-stamp.json"
        if stamp.is_file():
            try:
                return json.loads(stamp.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                return {}
    return {}


def store_root() -> Path:
    """The byte store. Entries are keyed (package, version, target) and
    hold nothing profile-specific, so the store is MACHINE-scoped: a
    second profile reuses the engine and browser this machine already
    downloaded instead of fetching its own copy.

    A payload overrides it — a sealed bundle carries its own store beside
    its manifest, and that IS per-install by construction.
    """
    env = os.environ.get("HERMES_RUNTIME_DIR")
    if env:
        return Path(env).resolve()
    stamped = _stamp().get("runtimeDir")
    if stamped:
        return Path(stamped).resolve()
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "tools"


def partials_root() -> Path:
    """The downloader's managed partials area: machine-scoped and shared
    (keyed by sha256(url), so two callers or two profiles reuse one
    partial), but anchored to the DEFAULT hermes root — NOT the byte
    store. The store can live inside a read-only sealed payload
    (WindowsApps/agent-payload), and partials are mutable state the
    downloader writes continuously, so they must land somewhere writable
    on every install kind: ``%LOCALAPPDATA%\\hermes\\cache\\partials`` on
    Windows, ``~/.hermes/cache/partials`` on POSIX.
    """
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "cache" / "partials"


def facts_path() -> Path:
    return store_root() / "facts.json"
