"""
Resolve where the Photon sidecar runs from and where its Node deps live.

The rungs and the reason a read-only tree has to copy the sidecar are in
``gateway.sidecar_runtime``, which this module wraps. What belongs here is
the Photon-specific part: where the sidecar ships, and the
``PHOTON_SIDECAR_DIR`` override name that predates the shared resolver.

This module is import-light on purpose: both ``adapter.py`` (gateway) and
``cli.py`` (``hermes photon ...``) use it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from gateway.sidecar_runtime import (
    dir_writable,
    override_env_var,
    resolve_sidecar,
)

logger = logging.getLogger(__name__)

SOURCE_SIDECAR_DIR = Path(__file__).parent / "sidecar"

_SIDECAR_NAME = "photon"

# Backwards-friendly private alias for module-internal use.
_dir_writable = dir_writable


def resolve_sidecar_dir(source_dir: Optional[Path] = None) -> Path:
    """Return the directory the sidecar should run from.

    ``source_dir`` defaults to the installed plugin tree; tests and callers
    that monkeypatch the adapter's ``_SIDECAR_DIR`` pass it through so the
    override keeps working.

    ``PHOTON_SIDECAR_DIR`` is read as well as the shared
    ``HERMES_PHOTON_SIDECAR_DIR``. Operators set the short name before the
    resolver was shared, and it costs one line to keep working.
    """
    source = Path(source_dir) if source_dir is not None else SOURCE_SIDECAR_DIR

    legacy = os.getenv("PHOTON_SIDECAR_DIR")
    if legacy and not os.getenv(override_env_var(_SIDECAR_NAME)):
        return Path(legacy)

    return resolve_sidecar(_SIDECAR_NAME, source)
