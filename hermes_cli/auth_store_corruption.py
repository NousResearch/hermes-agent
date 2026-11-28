"""Corrupt ``auth.json`` handling: keep a unique copy, and never write over the original.

``_load_auth_store`` degrades an unparseable store to an empty one so the CLI keeps working, but
the auth module read-modify-writes in many places. A plain empty store is therefore one
``_save_auth_store()`` away from replacing the corrupt file (and everything a human could still
recover from it) with ``{"providers": {}}``. The degraded store carries a marker instead, and
saving a marked store raises.
"""

from __future__ import annotations

import glob
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOAD_FAILED_KEY = "_auth_store_load_failed"
CORRUPT_COPY_KEY = "_auth_store_corrupt_copy"
_WARNED: set[tuple[str, str]] = set()


class AuthStoreCorruptError(RuntimeError):
    """A store loaded from an unparseable auth.json must not be written back over it."""


def _identical_copy(auth_file: Path, data: bytes) -> Optional[Path]:
    """Return an existing ``<name>.corrupt*`` copy holding exactly *data*, if any.

    ``_load_auth_store`` runs on every credential lookup, so without this a corrupt store would be
    copied again on each call (an unbounded stream of secret-bearing files).
    """
    for copy in auth_file.parent.glob(f"{glob.escape(auth_file.name)}.corrupt*"):
        try:
            if copy.is_file() and copy.stat().st_size == len(data) and copy.read_bytes() == data:
                return copy
        except OSError:
            continue
    return None


def preserve_corrupt_auth_file(auth_file: Path) -> Optional[Path]:
    """Copy *auth_file* to a new, unique ``<name>.corrupt.<utc-stamp>.<pid>.<rand>`` file (0600).

    A fixed ``auth.json.corrupt`` name would overwrite the copy kept from an earlier corruption,
    which may be the last recoverable one. Identical bytes are copied only once. Returns the copy,
    or None when it could not be written.
    """
    try:
        data = auth_file.read_bytes()
    except OSError:
        logger.debug("auth: could not read %s for a corrupt copy", auth_file, exc_info=True)
        return None
    existing = _identical_copy(auth_file, data)
    if existing is not None:
        return existing
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    for _ in range(10):
        candidate = auth_file.with_name(f"{auth_file.name}.corrupt.{stamp}.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        except OSError:
            logger.debug("auth: could not create %s", candidate, exc_info=True)
            return None
        try:
            with os.fdopen(fd, "wb") as dst:
                dst.write(data)  # bytes only: copy2() would copy the source mode too
        except OSError:
            logger.debug("auth: could not write %s", candidate, exc_info=True)
            candidate.unlink(missing_ok=True)
            return None
        return candidate
    return None


def degraded_store_for_corrupt_file(auth_file: Path, exc: Exception, empty_store: dict[str, Any]) -> dict[str, Any]:
    """Preserve a copy of the unparseable *auth_file* and return *empty_store* marked as load-failed.

    The warning is logged once per file and copy; later loads of the same corrupt bytes log at debug.
    """
    copy = preserve_corrupt_auth_file(auth_file)
    seen = (str(auth_file), str(copy))
    log = logger.debug if seen in _WARNED else logger.warning
    _WARNED.add(seen)
    log("auth: failed to parse %s (%s); using an empty store and refusing to overwrite the file. %s",
        auth_file, exc, f"Corrupt file preserved at {copy}" if copy else "A copy could NOT be preserved")
    return {**empty_store, LOAD_FAILED_KEY: str(exc), CORRUPT_COPY_KEY: str(copy) if copy else None}


def refuse_save_of_degraded_store(auth_store: dict[str, Any], auth_file: Path) -> None:
    """Raise AuthStoreCorruptError when *auth_store* came from a corrupt-file fallback."""
    if auth_store.get(LOAD_FAILED_KEY):
        raise AuthStoreCorruptError(
            f"Refusing to overwrite {auth_file}: it could not be parsed when it was loaded. Repair or remove "
            f"it first (preserved copy: {auth_store.get(CORRUPT_COPY_KEY) or 'none'}).")
