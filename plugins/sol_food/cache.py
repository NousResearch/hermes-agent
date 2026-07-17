"""Owner-private food image cache with terminal deletion guarantees.

- Directory mode 0700, files mode 0600.
- Opaque random file ids; callers never learn or log paths (the cache
  hands back ids, and even exceptions carry reason codes only).
- Every food image is deleted on success, error, cancel, or timeout, and
  no later than 60 seconds after terminal parsing (backstop timer).
- On startup, prior-process food-temp orphans are removed. The host's
  existing 24-hour global sweep remains only as the last crash backstop.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from pathlib import Path
from typing import Dict, Optional

from plugins.sol_food.limits import (
    FOOD_CACHE_DIR_MODE,
    FOOD_CACHE_FILE_MODE,
    FOOD_CACHE_TERMINAL_DELETE_SECONDS,
    FOOD_IMAGE_MAX_BYTES,
)

__all__ = ["FoodImageCache"]


class FoodImageCache:
    def __init__(self, state_dir: Path) -> None:
        self._dir = Path(state_dir) / "food-images"
        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, FOOD_CACHE_DIR_MODE)
        self._backstops: Dict[str, asyncio.TimerHandle] = {}

    def sweep_orphans(self) -> int:
        """Remove ALL cached food images (startup: nothing may survive a
        prior process). Returns the number removed."""
        removed = 0
        for entry in self._dir.iterdir():
            if entry.is_file():
                try:
                    entry.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed

    def store(self, data: bytes) -> str:
        if len(data) > FOOD_IMAGE_MAX_BYTES:
            raise ValueError("food_cache_too_large")
        image_id = secrets.token_hex(16)
        path = self._dir / image_id
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FOOD_CACHE_FILE_MODE)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return image_id

    def path_for(self, image_id: str) -> Optional[Path]:
        if not image_id or "/" in image_id or "." in image_id:
            return None
        path = self._dir / image_id
        return path if path.is_file() else None

    def delete(self, image_id: str) -> None:
        handle = self._backstops.pop(image_id, None)
        if handle is not None:
            handle.cancel()
        path = self._dir / image_id
        try:
            path.unlink()
        except OSError:
            pass

    def arm_terminal_backstop(self, image_id: str) -> None:
        """Guarantee deletion no later than 60 s after terminal parsing,
        even if the primary deletion path is skipped by a bug/crash of
        the caller's task."""
        loop = asyncio.get_running_loop()
        existing = self._backstops.pop(image_id, None)
        if existing is not None:
            existing.cancel()
        self._backstops[image_id] = loop.call_later(
            FOOD_CACHE_TERMINAL_DELETE_SECONDS, self.delete, image_id
        )
