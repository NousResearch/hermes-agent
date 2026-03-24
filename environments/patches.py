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

    This module is kept for backward compatibility — apply_patches() is now a no-op.

Usage:
    Call apply_patches() once at import time (done automatically by hermes_base_env.py).
    This is idempotent — calling it multiple times is safe.
"""

import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_patches():
    """Apply all monkey patches needed for Atropos compatibility.

    Now a no-op — Modal async safety is built directly into ModalEnvironment.
    Safe to call multiple times.
    """
    global _patches_applied
    if _patches_applied:
        return

    # Modal async-safety is now built into tools/environments/modal.py
    # via the _AsyncWorker class. No monkey-patching needed.
    logger.debug("apply_patches() called — no patches needed (async safety is built-in)")

    _patches_applied = True
