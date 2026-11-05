"""Storage interface handles held beside a legacy store handle.

A handle beside a store never changes what the store's path does: an open or close that fails logs at debug, the open
answers None, and the caller carries on with its store alone.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def open_beside(home: Optional[Path] = None) -> Any:
    """A handle on the store of ``home`` (the current home when None), or None when it cannot open."""
    try:
        from persistence_assembly import open_persistence, profile_scope

        return open_persistence(profile_scope(home=home))
    except Exception:  # health: allow BLE001 -- the store beside it decides the result; the handle stays absent
        logger.debug("storage interface unavailable for %s", home or "the current home", exc_info=True)
        return None


def open_reader_beside(target: Path, handles: ExitStack) -> Any:
    """A read-only handle over ``target``, closed with ``handles``, or None when it cannot open."""
    try:
        from persistence_assembly import open_persistence_reader

        return handles.enter_context(open_persistence_reader(target))
    except Exception:  # health: allow BLE001 -- the store beside it decides the result; the handle stays absent
        logger.debug("storage interface reader unavailable for %s", target, exc_info=True)
        return None


def close_beside(handle: Any) -> None:
    """Close ``handle`` (a handle from ``open_beside`` or an ``ExitStack`` of readers); None does nothing."""
    if handle is None:
        return
    try:
        handle.close()
    except Exception:  # health: allow BLE001 -- the store beside it decides the result; a failed release is logged
        logger.debug("storage interface close failed", exc_info=True)
