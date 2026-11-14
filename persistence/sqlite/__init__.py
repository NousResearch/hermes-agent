"""SQLite backend for the persistence contracts.

The package's public names load every public module under it on first use, so each repository module's
``register_repository`` call runs before the first store handle is opened. Importing a private module (the holder,
the registry) loads only what that module imports.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._backend import (
        CAPABILITY_ATTRIBUTES, SQLiteBackend, SQLiteStore, register_repository, translated,
    )

__all__ = [
    "CAPABILITY_ATTRIBUTES",
    "SQLiteBackend",
    "SQLiteStore",
    "register_repository",
    "translated",
]

_PACKAGE_DIR = Path(__file__).parent


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module in sorted(_PACKAGE_DIR.rglob("*.py")):
        parts = module.relative_to(_PACKAGE_DIR).with_suffix("").parts
        if not any(part.startswith("_") for part in parts):
            importlib.import_module(".".join((__name__, *parts)))
    from . import _backend

    for public in __all__:
        globals()[public] = getattr(_backend, public)
    return globals()[name]
