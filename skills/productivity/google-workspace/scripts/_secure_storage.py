"""Helpers for persisting Google Workspace credentials securely."""

from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any


def write_private_json(path: Path, data: Any) -> None:
    """Atomically write JSON that only the owner can read (mode 0600 from creation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(
            str(tmp_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        # The replaced file keeps the temp file's 0600 mode, so it is never
        # readable by others, even for an instant, and an existing file with
        # looser permissions is fixed as a side effect.
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
