"""Choose the Olm/Megolm implementation that ``mautrix.crypto`` runs on.

``MATRIX_E2EE_BACKEND`` selects it:

* ``auto`` (default): vodozemac (the ``vodozemac`` package, via ``olm_vodozemac``) whenever it
  is installed; the ``matrix`` extra always installs it. Legacy python-olm/libolm, which the
  Matrix.org Foundation deprecated in July 2024, is used only when vodozemac is missing but
  python-olm is present, and logs a warning.
* ``vodozemac``: vodozemac or nothing. Never falls back to libolm.
* ``libolm``: python-olm or nothing (the ``matrix-libolm`` extra, Linux only). The fallback for
  a vodozemac regression; the vodozemac layer is never installed.

An unknown value selects nothing, so E2EE fails closed (``MATRIX_E2EE_MODE=required`` refuses
to connect). The choice is process-wide by construction -- the backend is registered as the
``olm`` module in ``sys.modules`` -- so it is read from the process environment, not per profile.

The selection must happen before anything imports ``mautrix.crypto`` (including
``mautrix.crypto.attachments``), because mautrix binds ``import olm`` at import time.
:func:`select_backend` is idempotent and cheap; the adapter calls it at import time and again
before every E2EE entry point (the ``matrix`` extra may be lazily installed in between).

Rollback after a crypto store was migrated to vodozemac (``crypto_migration``): libolm can no
longer read it, so switching to ``libolm`` also means restoring the
``crypto.db.libolm-backup-*`` copy written before the migration. The adapter refuses to open a
store libolm cannot read rather than letting mautrix fail half-way (see
``crypto_migration.libolm_store_problem``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

ENV_VAR = "MATRIX_E2EE_BACKEND"
AUTO = "auto"
VODOZEMAC = "vodozemac"
LIBOLM = "libolm"
NONE = "none"
CHOICES = (AUTO, VODOZEMAC, LIBOLM)

_selected: str | None = None


def requested_backend() -> str:
    """The configured backend (lower-cased), ``auto`` when unset. Not validated."""
    return os.environ.get(ENV_VAR, "").strip().lower() or AUTO


def _real_python_olm_available() -> bool:
    module = sys.modules.get("olm")
    if module is not None:
        return getattr(module, "BACKEND_NAME", None) != VODOZEMAC
    try:
        return importlib.util.find_spec("olm") is not None
    except (ImportError, ValueError):
        return False


def _install_vodozemac() -> bool:
    if importlib.util.find_spec("vodozemac") is None:
        return False
    from plugins.platforms.matrix import olm_vodozemac
    try:
        olm_vodozemac.install()
    except RuntimeError as exc:  # python-olm was imported before we got here
        logger.warning("Matrix: cannot switch E2EE backend to vodozemac: %s", exc)
        return False
    logger.debug("Matrix: E2EE backend: vodozemac %s", olm_vodozemac.VODOZEMAC_VERSION)
    return True


def select_backend() -> str:
    """Register the requested backend as ``olm``; return ``vodozemac``, ``libolm`` or ``none``.

    A successful selection is sticky for the process. ``none`` is not cached, so a later call
    (after the ``matrix`` extra was installed on first use) can still pick a backend.
    """
    global _selected
    if _selected is not None:
        return _selected
    requested = requested_backend()
    if requested not in CHOICES:
        logger.error("Matrix: %s=%r is not one of %s; no E2EE backend selected",
                     ENV_VAR, requested, ", ".join(CHOICES))
        return NONE

    if requested in (AUTO, VODOZEMAC) and _install_vodozemac():
        _selected = VODOZEMAC
        return _selected
    if requested == VODOZEMAC:
        return NONE  # explicitly requested: never fall back to libolm

    if _real_python_olm_available():
        _selected = LIBOLM
        if requested == LIBOLM:
            logger.warning(
                "Matrix: %s=libolm: E2EE runs on python-olm/libolm, which the Matrix.org "
                "Foundation deprecated in July 2024 and no longer maintains (unfixed "
                "CVE-2024-45191/45192/45193). Use it only as a temporary fallback.", ENV_VAR)
        else:
            logger.warning(
                "Matrix: vodozemac is not installed, so E2EE falls back to python-olm/libolm, "
                "which the Matrix.org Foundation deprecated in July 2024 and no longer "
                "maintains. Re-sync the matrix extra to switch to vodozemac; the crypto store "
                "is migrated automatically.")
        return _selected
    return NONE


def selected_backend() -> str | None:
    """The backend chosen so far in this process, or None if none was selected yet."""
    return _selected
