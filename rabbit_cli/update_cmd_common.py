"""Shared leaf helpers for the ``rabbit update`` modules (no Rabbit imports; no cycle)."""

import logging
from contextlib import contextmanager

# Log-record parity with the origin module.
logger = logging.getLogger("rabbit_cli.update_cmd")


@contextmanager
def _best_effort(message: str):
    """Run a non-critical update step; swallow ``Exception`` and log it at debug.

    The updater must never die on bookkeeping (receipt, notices, cache seeds):
    ``message`` is the ``%s``-style debug line the inline ``try/except`` used.
    """
    try:
        yield
    except Exception as exc:
        logger.debug(message, exc)
