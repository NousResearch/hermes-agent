"""Name resolution for profile homes that live outside ``<root>/profiles/``.

A profile's home may be a symlink to an encrypted or off-disk directory, or the process
home of a scoped gateway. Its resolved path carries none of the ``<root>/profiles/<name>``
shape, so the owning name has to be found by identity: the ``profiles/<name>`` symlink
entry that resolves to the home. NULL over a fabricated owner.

Ported here (from the branch's top-level ``hermes_state_sessions_linked_homes.py``) by the
``#153250`` rebase: ``be3d6affc8`` moved the session store under ``persistence/`` and its
backend-import guard allows no file outside ``persistence/``.

Its roots are HANDED IN, never read here. The guard permits a store in this package no
process-home read of its own ("a store reads beside its own profile folder"), so the one
file ``_HOME_READS`` already lists for ``get_default_hermes_root`` — ``_store/sessions.py``,
whose reason is *the owner name the store stamps* — resolves both roots and passes them in.
This module names no home reader at all, so the rebased branch passes the guard unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def linked_profile_name(home: Path, root: Path, fallback_roots: Iterable[Path] = ()) -> Optional[str]:
    """Name of the ``profiles/<name>`` symlink entry that resolves to *home*, if any.

    A profile home that is a symlink outside its root's ``profiles/`` (an encrypted or
    off-disk home) resolves to a path without the ``<root>/profiles/<name>`` shape — and a
    scoped gateway runs with ``HERMES_HOME`` set to that home, which makes the home its own
    root. In both cases the registered name is the symlink entry's: find it by identity,
    under *root* and under every root in *fallback_roots* — the platform default root, where
    the entry lives once the scoped home has become its own root — alike. NULL over a
    fabricated owner.
    """
    candidates = [root]
    for fallback in fallback_roots:
        if fallback is not None and fallback not in candidates:
            candidates.append(fallback)
    for base in candidates:
        profiles = base / "profiles"
        if not profiles.is_dir():
            continue
        try:
            entries = list(profiles.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_symlink() and entry.resolve(strict=False) == home:
                    return entry.name
            except OSError:
                continue
    return None
