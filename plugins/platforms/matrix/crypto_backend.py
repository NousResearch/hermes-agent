"""Choose the Olm/Megolm implementation that ``mautrix.crypto`` runs on.

vodozemac (the ``vodozemac`` package, via ``olm_vodozemac``) whenever it is installed; the
``matrix`` extra always installs it. Legacy python-olm/libolm, which the Matrix.org Foundation
deprecated in July 2024, is only used when vodozemac is missing but python-olm is still
present (an environment that has not been re-synced since this change), and logs a warning.

The selection must happen before anything imports ``mautrix.crypto`` (including
``mautrix.crypto.attachments``), because mautrix binds ``import olm`` at import time.
:func:`select_backend` is idempotent and cheap; the adapter calls it at import time and again
before every E2EE entry point (the ``matrix`` extra may be lazily installed in between).

Rollback after a crypto store was migrated to vodozemac (``crypto_migration``): libolm can no
longer read it, so downgrade Hermes and restore the ``crypto.db.libolm-backup-*`` copy written
before the migration.
"""

from __future__ import annotations

import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)

VODOZEMAC = "vodozemac"
LIBOLM = "libolm"
NONE = "none"

_selected: str | None = None


def _real_python_olm_available() -> bool:
    module = sys.modules.get("olm")
    if module is not None:
        return getattr(module, "BACKEND_NAME", None) != VODOZEMAC
    try:
        return importlib.util.find_spec("olm") is not None
    except (ImportError, ValueError):
        return False


def select_backend() -> str:
    """Install vodozemac as ``olm`` if possible; return ``vodozemac``, ``libolm`` or ``none``.

    A successful selection is sticky for the process. ``none`` is not cached, so a later call
    (after the ``matrix`` extra was installed on first use) can still pick a backend.
    """
    global _selected
    if _selected is not None:
        return _selected
    if importlib.util.find_spec("vodozemac") is not None:
        from plugins.platforms.matrix import olm_vodozemac
        try:
            olm_vodozemac.install()
        except RuntimeError as exc:  # python-olm was imported before we got here
            logger.warning("Matrix: cannot switch E2EE backend to vodozemac: %s", exc)
        else:
            _selected = VODOZEMAC
            logger.debug("Matrix: E2EE backend: vodozemac %s", olm_vodozemac.VODOZEMAC_VERSION)
            return _selected

    if _real_python_olm_available():
        _selected = LIBOLM
        logger.warning(
            "Matrix: E2EE is running on python-olm/libolm, which the Matrix.org Foundation "
            "deprecated in July 2024 and no longer maintains. Re-sync the matrix extra to switch "
            "to vodozemac; the crypto store is migrated automatically.")
        return _selected
    return NONE


def selected_backend() -> str | None:
    """The backend chosen so far in this process, or None if none was selected yet."""
    return _selected
