"""DM Pairing System

Code-based approval flow for authorizing new users on messaging platforms.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_dir
from utils import atomic_replace

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8
CODE_TTL_SECONDS = 3600
RATE_LIMIT_SECONDS = 600
LOCKOUT_SECONDS = 3600

MAX_PENDING_PER_PLATFORM = 3
MAX_FAILED_ATTEMPTS = 5

PAIRING_DIR = get_hermes_dir("pairing", "pairing")


class PairingStore:
    """Manages pairing codes and approved user lists.

    Data files per platform:
      - {platform}-pending.json   : pending pairing requests
      - {platform}-approved.json  : approved (paired) users
      - _rate_limits.json         : rate limit tracking

    When constructed with ``profile="<name>"``, storage lives under
    ``<HERMES_HOME>/profiles/<name>/pairing/`` (per-profile, used by
    multiplexing gateways so each profile has its own whitelist).
    Without a profile, storage is the global ``<HERMES_HOME>/pairing/``
    directory (backward-compat for the ``hermes pairing`` CLI).
    """

    def __init__(self, profile: Optional[str] = None):
        # Resolve storage directory lazily — tests use a temp HERMES_HOME
        # and PairingStore may be constructed before the env is set.
        if profile:
            from hermes_constants import get_hermes_home
            self._dir = get_hermes_home() / "profiles" / profile / "pairing"
        else:
            self._dir = PAIRING_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        # Protects all read-modify-write cycles. The gateway runs multiple
        # platform adapters concurrently in threads sharing one PairingStore.
        self._lock = threading.RLock()
        self._profile = profile  # for diagnostics / log lines

    @property
    def profile(self) -> Optional[str]:
        """Profile name this store is scoped to, or None for the global store."""
        return self._profile

    def _pending_path(self, platform: str) -> Path:
        return self._dir / f"{platform}-pending.json"

    def _approved_path(self, platform: str) -> Path:
        return self._dir / f"{platform}-approved.json"

    def _rate_limit_path(self) -> Path:
        return self._dir / "_rate_limits.json"

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
