"""Helpers for persisting Google Workspace credentials securely."""

from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any


def write_private_json(path: Path, data: Any) -> None:
    """Atomically write JSON with owner-only permissions."""
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
        os.replace(tmp_path, path)

        # Windows does not enforce the creation mode in the same way as
        # POSIX. This also repairs the final mode if a platform altered it
        # during replacement.
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
