"""Move Hermes-owned disposable data into an MSIX package's uninstall-managed cache."""

from __future__ import annotations

import os
from pathlib import Path

from hermes_constants import _get_platform_default_hermes_home, get_default_hermes_root
from hermes_platform.windows_appdata import local_cache_folder


def managed_cache_dir(relative: str, *, home: Path | None = None) -> Path:
    """Keep ordinary/custom homes; migrate default-home caches for packaged installs.

    Callers name only disposable, app-owned directories. Explicit custom roots and
    links to user-managed storage retain their existing ownership. Profiles keep
    separate subdirectories where their caller already scopes storage by profile.
    """
    home = home if home is not None else get_default_hermes_root()
    source = home / relative
    cache = local_cache_folder()
    if cache is None:
        return source
    native_home = _get_platform_default_hermes_home()
    if get_default_hermes_root().resolve() != native_home.resolve():
        return source
    try:
        scoped = home.relative_to(native_home)
    except ValueError:
        return source
    # Do not adopt a user's symlink/junction or its target into package ownership.
    if source.resolve() != source.absolute():
        return source
    target = cache / "hermes-inference" / scoped / relative
    if target.resolve() != target.absolute():
        raise RuntimeError(f"Hermes package cache points to linked storage: {target}")
    if not source.exists():
        return target

    from pm.filesystem import lock_fd

    native_home.mkdir(parents=True, exist_ok=True)
    fd = os.open(native_home / ".cache-migration.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not lock_fd(fd, wait=True, timeout=30):
            raise RuntimeError("Another Hermes process is moving its cached downloads; retry shortly.")
        if source.exists():
            if _contains_links(source):
                return source
            _move_directory(source, target)
    except OSError as exc:
        raise RuntimeError(f"Could not move Hermes cache from {source} to {target}. "
                           "Close other Hermes processes and retry; existing files were preserved.") from exc
    finally:
        os.close(fd)
    return target


def _move_directory(source: Path, target: Path) -> None:
    """Rename only: never copy a multi-GB model, follow links, or replace a collision.

    A partial migration is resumable: moved files remain at the destination and a
    retry merges the remaining entries. Windows rename refuses existing targets.
    """
    from pm.filesystem import is_junction

    if source.is_symlink() or is_junction(source):
        raise OSError(f"Cache contains user-managed linked storage: {source}")
    if target.is_symlink() or (target.exists() and is_junction(target)):
        raise OSError(f"Cache destination is linked storage: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        source.rename(target)
        return
    for entry in source.iterdir():
        destination = target / entry.name
        if entry.is_dir():
            _move_directory(entry, destination)
        elif entry.is_symlink() or destination.exists() or destination.is_symlink():
            raise OSError(f"Cache migration would replace or follow an existing file: {entry}")
        else:
            entry.rename(destination)
    source.rmdir()


def _contains_links(root: Path) -> bool:
    """A nested junction is also user-selected storage; never adopt its tree."""
    from pm.filesystem import is_junction

    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            entry = Path(parent) / name
            if entry.is_symlink() or is_junction(entry):
                return True
    return False
