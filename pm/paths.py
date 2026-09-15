"""Where the store, lockfile, and installed-state file live."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def lockfile_path() -> Path:
    return Path(__file__).resolve().parent / "lock.json"


def store_root() -> Path:
    from hermes_cli.runtime_paths import store_root as resolve

    return resolve(repo_root())


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


def writable_store_root() -> Path:
    if not (store_root().parent / "manifest.json").is_file():
        return store_root()
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "tools"


def runtime_facts_path() -> Path:
    from hermes_cli.runtime_paths import runtime_facts_path as resolve

    return resolve(repo_root())
