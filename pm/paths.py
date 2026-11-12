"""Where the store, lockfile, and installed-state file live."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def install_root() -> Path:
    """The tree this process runs from. ``HERMES_INSTALL_ROOT`` when a steward
    wrapper sets it (Nix points it at the sealed tree whose install stamp lives
    outside the package dir), else the executing checkout."""
    env = os.environ.get("HERMES_INSTALL_ROOT")
    return Path(env) if env else repo_root()


def install_stamp_path(project_root: Path) -> Path:
    """THE stamp location for ``project_root``, shared by every stamp reader.

    Beside the code in checkouts, Docker and desktop payloads. A Nix package
    bakes the stamp outside the store's package dir and its wrapper carries
    ``HERMES_INSTALL_ROOT`` for the executing tree only — so the executing
    tree resolves through install_root, any other tree is taken literally.
    PM reads it in sealed stages (the Docker runtime base) before hermes_cli
    ships, so it lives here rather than beside the stewards.
    """
    root = Path(project_root)
    if root.resolve() == repo_root():
        root = install_root()
    return root / "install-stamp.json"


def lockfile_path() -> Path:
    return Path(__file__).resolve().parent / "lock.json"


def store_root() -> Path:
    from pm.environments import store_root as resolve

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
    from pm.environments import runtime_facts_path as resolve

    return resolve(repo_root())
