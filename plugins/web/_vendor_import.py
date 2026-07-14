"""
Helper to import a vendor SDK even when a local plugin directory shadows it.

Web plugin directories like ``plugins/web/firecrawl/`` can end up on ``sys.path``
via the plugin discovery machinery, shadowing PyPI packages with the same name
(``firecrawl``, ``ddgs``).  This helper temporarily removes shadowing paths so
the real SDK import always resolves correctly.
"""

import sys
from pathlib import Path


def _import_vendor(module_name: str, package_dir: str | Path) -> object:
    """Import a vendor SDK bypassing any local plugin directory that may shadow it.

    Args:
        module_name: The top-level module name to import (e.g. ``"firecrawl"``).
        package_dir: The absolute path of the local plugin directory that shadows
                     the vendor package (e.g. ``/path/to/plugins/web/firecrawl``).

    Returns:
        The imported module.
    """
    package_dir_str = str(Path(package_dir).resolve())

    # Remove any sys.path entries that would shadow the real SDK
    removed: list[int] = []
    for i, entry in enumerate(sys.path):
        resolved = str(Path(entry).resolve())
        if resolved == package_dir_str:
            removed.append(i)

    for idx in reversed(removed):
        sys.path.pop(idx)

    try:
        return __import__(module_name)
    finally:
        # Restore removed entries in original order
        for idx in removed:
            sys.path.insert(idx, package_dir_str)
