"""Select package-owned cache paths without moving existing installation data."""

from __future__ import annotations

from pathlib import Path

from hermes_constants import _get_platform_default_hermes_home, get_default_hermes_root
from hermes_platform.windows_appdata import local_cache_folder


def managed_cache_dir(relative: str, *, home: Path | None = None) -> Path:
    """Use MSIX LocalCache for default-home downloads; preserve explicit custom roots.

    This is a path lookup only. Existing files, including another installation's
    models and linked directories, are neither adopted nor inspected.
    """
    home = home if home is not None else get_default_hermes_root()
    cache = local_cache_folder()
    if cache is None:
        return home / relative
    native_home = _get_platform_default_hermes_home()
    if get_default_hermes_root().resolve() != native_home.resolve():
        return home / relative
    try:
        scoped = home.relative_to(native_home)
    except ValueError:
        return home / relative
    return cache / "hermes-inference" / scoped / relative
