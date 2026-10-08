"""Import-time replay of buffered provider-discovery failures.

Kept out of the ``hermes_cli.main`` facade (facade line cap): ``main.py``
calls :func:`early_replay_provider_failures` after ``setup_logging()`` and
:func:`bind_provider_replay` unconditionally, so the dispatch replay never
hits ``NameError`` when logging setup fails.
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
