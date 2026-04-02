"""
Monkey patches for making hermes-agent tools work inside async frameworks (Atropos).

Problem:
    Some tools use asyncio.run() internally (e.g., Modal backend via SWE-ReX,
    web_extract). This crashes when called from inside Atropos's event loop because
    asyncio.run() can't be nested.

Solution:
    The Modal environment (tools/environments/modal.py) now uses a dedicated
    _AsyncWorker thread internally, making it safe for both CLI and Atropos use.
    No monkey-patching is required.

    This module is kept for backward compatibility. apply_patches() is a no-op.

Usage:
    Call apply_patches() once at import time (done automatically by hermes_base_env.py).
    This is idempotent and safe to call multiple times.
"""

import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_patches():
    """Apply all monkey patches needed for Atropos compatibility."""
    global _patches_applied
    if _patches_applied:
        return

    logger.debug("apply_patches() called; no patches needed (async safety is built-in)")
    _patches_applied = True
