"""Import-time replay of buffered provider-discovery failures.

Kept out of the ``hermes_cli.main`` facade (facade line cap): ``main.py``
calls :func:`early_replay_provider_failures` after ``setup_logging()`` and
:func:`bind_provider_replay` unconditionally, so the dispatch replay never
hits ``NameError`` when logging setup fails. The dispatch site replays via
:func:`dispatch_replay_provider_failures`, which stays silent until logging
is known ready (``providers._LOGGING_READY``), so a setup failure never
leaks buffered warnings through ``logging.lastResort`` raw stderr.
"""

import logging

logger = logging.getLogger(__name__)


def bind_provider_replay(current=None):
    """Return a bound discovery-failure replay callable, or ``None``.

    Never raises: ``main.py`` runs this before logging exists, so an
    unimportable ``providers`` package must stay silent (raw stderr stays
    clean for the fullscreen TUI pre-logging).
    """
    if current is not None:
        return current
    try:
        from providers import replay_provider_load_failures
    except Exception:  # health: allow BLE001 -- import-time, no logger yet; stderr must stay clean
        return None
    return replay_provider_load_failures


def early_replay_provider_failures():
    """Replay failures buffered before ``setup_logging()``; never raises.

    Returns the bound replay so ``main.py`` can flush late-buffered failures
    again at dispatch. A failed flush stays buffered for the dispatch retry.
    """
    replay = bind_provider_replay()
    if replay is None:
        return None
    try:
        replay()
    except Exception:
        logger.debug("early provider-failure replay failed; dispatch will retry", exc_info=True)
    return replay


def dispatch_replay_provider_failures(bound=None):
    """Dispatch-time replay of buffered provider-discovery failures.

    No-op until logging is known ready: ``providers._LOGGING_READY`` flips
    only inside :func:`replay_provider_load_failures` after ``setup_logging()``
    ran. The bound replay may still be armed when setup failed (so dispatch
    never hits ``NameError``), but calling it there would emit
    ``logger.warning`` with zero handlers attached, falling through to
    ``logging.lastResort`` raw stderr. Gating here keeps raw stderr clean and
    leaves failures buffered for a later ready replay. Never raises.
    """
    if bound is None:
        return 0
    try:
        from providers import _LOGGING_READY
    except Exception:  # health: allow BLE001 -- dispatch pre-logging; stderr must stay clean
        return 0
    if not _LOGGING_READY:
        return 0
    try:
        return bound() or 0
    except Exception:
        logger.exception("buffered provider-failure replay failed")
        return 0
