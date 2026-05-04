from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def prefer_hermes_package_root() -> None:
    """Keep Hermes' wheel root ahead of Chaquopy dependency modules.

    Some Android dependencies ship a top-level ``utils.py``. Hermes also has a
    top-level ``utils.py`` with shared helpers, so Chaquopy must resolve Hermes'
    package root before its requirements directory when importing that module.
    """
    root = _hermes_package_root()
    if root:
        root_text = str(root)
        sys.path[:] = [item for item in sys.path if item != root_text]
        sys.path.insert(0, root_text)

    loaded_utils = sys.modules.get("utils")
    if loaded_utils is not None and not hasattr(loaded_utils, "atomic_replace"):
        sys.modules.pop("utils", None)


def _hermes_package_root() -> Path | None:
    for module_name in ("hermes_cli", "hermes_android"):
        spec = importlib.util.find_spec(module_name)
        origin = getattr(spec, "origin", None)
        if origin:
            return Path(origin).resolve().parent.parent
    return None
