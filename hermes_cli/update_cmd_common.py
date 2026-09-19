"""Shared leaf helpers for the ``hermes update`` modules (no Hermes imports; no cycle)."""

import logging
from contextlib import contextmanager

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


@contextmanager
def _best_effort(message: str, level: int = logging.DEBUG):
    """Run a non-critical update step; swallow ``Exception`` and log it at *level*.

    The updater must never die on bookkeeping (receipt, notices, cache seeds):
    ``message`` is the ``%s``-style debug line the inline ``try/except`` used.
    Pass ``logging.WARNING`` for bookkeeping whose loss must be visible at the default log level
    (the update receipt) — a DEBUG-only swallow is what made a missing receipt unrecoverable
    in the field. See #112558.
    """
    try:
        yield
    except Exception as exc:
        logger.log(level, message, exc)
