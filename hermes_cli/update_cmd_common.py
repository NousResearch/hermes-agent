"""Shared leaf helpers for the ``hermes update`` modules (no Hermes imports; no cycle)."""

import logging
from contextlib import contextmanager

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")

#: Backend purposes whose supervision means "Desktop owns the control plane"
#: (serve/dashboard are the control plane, the gateway a detached sibling, #92091).
BACKEND_PURPOSES = ("serve", "dashboard")


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


def _try_call(fn, log_message: str, *log_args, default=None):
    """``fn()``, or *default* after logging the exception at debug (``log_message`` gets ``*log_args, exc``)."""
    try:
        return fn()
    except Exception as exc:
        logger.debug(log_message, *log_args, exc)
        return default


def desktop_owns_gateway_lifecycle() -> bool:
    """True when Desktop currently supervises this install's control plane (updater must not steal gateway start).

    Platform-neutral probe (the spawn ledger is cross-platform by construction): the
    fleet-restart tail consults it on every OS, not just the Windows pause/resume paths
    (#126177 — an external ``hermes update`` must bracket its fleet bounce when a live
    Desktop instance is supervising this install). The Windows module's
    ``_desktop_owns_gateway_lifecycle`` delegates here and adds the venv-holder scan
    fallback, which only exists on Windows.

    Not proof messaging is served: serve is the control plane, the gateway a detached
    sibling. Prefer the spawn ledger. An orphaned control plane (supervisor gone) does
    not count.

    See #76129, #92091, #126177.
    """
    with _best_effort("Desktop-lifecycle ledger probe failed: %s"):
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        if any(
            entry.get("purpose") in BACKEND_PURPOSES and spawner_is_dead(entry) is False
            for entry in ledger_entries()
        ):
            return True
    return False
