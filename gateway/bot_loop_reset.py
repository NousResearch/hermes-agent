"""Epoch file for a bot-loop reset, stored under gateway-state.

Callers decide when a reset applies. This module only stores the epoch.
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
import time
from pathlib import Path

EPOCH_NAME = "bot-loop-reset-epoch"


def epoch_path(root: Path) -> Path:
    """``<root>/gateway-state/<EPOCH_NAME>``."""
    return Path(root) / "gateway-state" / EPOCH_NAME


def read_epoch(root: Path) -> float | None:
    """The stored epoch, or None when the file is missing or unreadable."""
    try:
        value = float(epoch_path(root).read_text(encoding="utf-8").strip())
    except (OSError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def write_epoch(root: Path, when: float | None = None) -> float:
    """Store ``when`` (or now) via a same-directory temp file and ``os.replace``."""
    stamp = time.time() if when is None else when
    path = epoch_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{stamp}\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return stamp
