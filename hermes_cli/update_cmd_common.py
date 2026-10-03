"""Shared leaf helpers for the ``hermes update`` modules (no Hermes imports; no cycle)."""

import logging
from contextlib import contextmanager

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


@contextmanager
def _best_effort(message: str, *, failure: dict[str, bool] | None = None):
    """Run a non-critical update step; swallow ``Exception`` and log it at debug.

    The updater must never die on bookkeeping (receipt, notices, cache seeds):
    ``message`` is the ``%s``-style debug line the inline ``try/except`` used.
    """
    try:
        yield
    except Exception as exc:
        if failure is not None:
            failure["failed"] = True
        logger.debug(message, exc)
