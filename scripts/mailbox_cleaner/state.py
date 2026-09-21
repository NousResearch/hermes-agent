from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .models import Observation


class StateStore:
    """Minimal local dedupe state; never stores message bodies or credentials."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return set(raw.get("urgent_fingerprints", []))
        except (OSError, ValueError, TypeError):
            return set()

    def _atomic_write(self, fingerprints: set[str]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"urgent_fingerprints": sorted(fingerprints)}, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def record_urgent(self, item: Observation) -> bool:
        fingerprints = self._load()
        if item.fingerprint in fingerprints:
            return False
        fingerprints.add(item.fingerprint)
        self._atomic_write(fingerprints)
        return True
