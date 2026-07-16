"""Helpers for the writable Photon sidecar runtime directory.

The bundled plugin source tree may be installed under an immutable prefix
(`/opt/hermes` in containers). The Node sidecar's mutable npm install state
therefore lives under HERMES_HOME, while the committed JS entrypoints are
mirrored from the bundled source on demand.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from hermes_constants import get_hermes_home


_SOURCE_SIDECAR_DIR = Path(__file__).parent / "sidecar"
_RUNTIME_FILES = (
    "index.mjs",
    "package.json",
    "package-lock.json",
    "patch-spectrum-mixed-attachments.mjs",
)


def get_source_sidecar_dir() -> Path:
    """Return the bundled, read-only Photon sidecar source directory."""
    return _SOURCE_SIDECAR_DIR


def get_runtime_sidecar_dir() -> Path:
    """Return the writable sidecar directory for the active HERMES_HOME."""
    return get_hermes_home() / "platforms" / "photon" / "sidecar"


def ensure_runtime_sidecar_files() -> Path:
    """Mirror the committed sidecar sources into the writable runtime dir."""
    runtime_dir = get_runtime_sidecar_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name in _RUNTIME_FILES:
        shutil.copy2(_SOURCE_SIDECAR_DIR / name, runtime_dir / name)
    return runtime_dir
