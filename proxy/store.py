"""In-memory credential store (Phase 1).

Thread-safe, write-only external interface.  The ``_resolve`` method is
internal — used only by the proxy server to substitute placeholders.
No external read/get API is exposed; this is the core security invariant.

Phase 3 will add AES-256-GCM encrypted persistence + Argon2id key derivation.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CredentialStore:
    """Thread-safe, write-only credential store.

    External callers can *store*, *rotate*, *delete*, and *list* credential
    names.  Only the proxy server (via ``_resolve``) can retrieve the actual
    value — and that method is deliberately prefixed with ``_`` to signal
    it is not part of the public API.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._creds: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public (write-only) interface
    # ------------------------------------------------------------------

    def store(self, name: str, value: str) -> None:
        """Store a credential.  Overwrites if the name already exists."""
        with self._lock:
            self._creds[name] = value
            logger.info("credential stored: %s", name)

    def rotate(self, name: str, value: str) -> None:
        """Atomically replace a credential value.

        Behaves identically to ``store`` but logs the operation as a
        rotation for audit clarity.
        """
        with self._lock:
            existed = name in self._creds
            self._creds[name] = value
            if existed:
                logger.info("credential rotated: %s", name)
            else:
                logger.info("credential stored (rotate on new name): %s", name)

    def delete(self, name: str) -> bool:
        """Remove a credential.  Returns True if it existed."""
        with self._lock:
            existed = self._creds.pop(name, None) is not None
            if existed:
                logger.info("credential deleted: %s", name)
            else:
                logger.debug("credential delete (not found): %s", name)
            return existed

    def list_names(self) -> List[str]:
        """Return a sorted list of stored credential names."""
        with self._lock:
            return sorted(self._creds.keys())

    # ------------------------------------------------------------------
    # Internal — proxy server only
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> Optional[str]:
        """Resolve a credential name to its value.

        **Internal use only** — called by the proxy server during header/body
        rewriting.  Returns ``None`` if the name is not in the store.
        """
        with self._lock:
            return self._creds.get(name)

    def __len__(self) -> int:
        with self._lock:
            return len(self._creds)
