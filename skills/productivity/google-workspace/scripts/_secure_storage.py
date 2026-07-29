"""Helpers for persisting Google Workspace credentials securely."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_private_json(path: Path, data: Any) -> None:
    """Write JSON with owner-only permissions, including over existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            fd = -1  # fdopen owns the descriptor from here.
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
    finally:
        if fd >= 0:
            os.close(fd)

    # Windows does not expose fchmod. This is also a final best-effort repair
    # if a platform altered the mode while closing the descriptor.
    try:
        path.chmod(0o600)
    except OSError:
        pass
