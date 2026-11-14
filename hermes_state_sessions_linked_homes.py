"""Name resolution for profile homes that live outside ``<root>/profiles/``.

A profile's home may be a symlink to an encrypted or off-disk directory, or the process
home of a scoped gateway. Its resolved path carries none of the ``<root>/profiles/<name>``
shape, so the owning name has to be found by identity: the ``profiles/<name>`` symlink
entry that resolves to the home. NULL over a fabricated owner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def linked_profile_name(home: Path, root: Path) -> Optional[str]:
    """Name of the ``profiles/<name>`` symlink entry that resolves to *home*, if any.

    A profile home that is a symlink outside its root's ``profiles/`` (an encrypted or
    off-disk home) resolves to a path without the ``<root>/profiles/<name>`` shape — and a
    scoped gateway runs with ``HERMES_HOME`` set to that home, which makes the home its own
    root. In both cases the registered name is the symlink entry's: find it by identity,
    under the scoped root and the platform default root alike. NULL over a fabricated owner.
    """
    from hermes_constants import get_default_hermes_root
    candidates = [root]
    try:
        platform_default = get_default_hermes_root(home="").resolve()
    except (OSError, RuntimeError, ValueError):
        platform_default = None
    if platform_default is not None and platform_default not in candidates:
        candidates.append(platform_default)
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
