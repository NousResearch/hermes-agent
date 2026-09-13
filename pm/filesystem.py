"""Filesystem checks used before PM can replace the caller's interpreter."""
from __future__ import annotations

import os
from pathlib import Path
import stat


def is_junction(path: Path) -> bool:
    """Keep junctions opaque even before Python 3.12's Path.is_junction exists."""
    return os.name == "nt" and path.lstat().st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
