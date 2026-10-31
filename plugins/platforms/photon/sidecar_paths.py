"""Resolve where the Photon sidecar runs from and where its Node deps live.

Hosted images keep the plugin tree read-only (EROFS), so, mirroring
``resolve_whatsapp_bridge_dir``: (1) ``PHOTON_SIDECAR_DIR`` override as-is; (2) writable
source dir → run in place; (3) read-only with baked, current ``node_modules`` → in place;
(4) read-only and deps missing/stale → mirror the source files to
``$HERMES_HOME/photon/sidecar``. The mirror is refreshed by content compare; ``node_modules``
is left alone so the lockfile-vs-install-marker check triggers ``npm ci`` inside the mirror.
Resolution never happens at import time (it probes/copies on disk).
"""
from __future__ import annotations

import filecmp
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SOURCE_SIDECAR_DIR = Path(__file__).parent / "sidecar"
# Files that define the sidecar; node_modules is deliberately absent (baked on managed
# images or installed by npm in the mirror).
_MIRROR_FILES = ("index.mjs", "package.json", "package-lock.json", "patch-spectrum-mixed-attachments.mjs")
# Tests monkeypatch these module globals directly; the accessors honor a non-None value.
_SIDECAR_DIR: Optional[Path] = None
# Written by `hermes photon install-sidecar` on npm failure so check_requirements() can
# surface the root cause later; cleared on success.
_NPM_ERROR_LOG: Optional[Path] = None
_NPM_ERROR_LOG_MAX_CHARS = 300


def dir_writable(path: Path) -> bool:
    """True when we can create files in ``path`` (probe, not stat — stat lies on root-squash
    / read-only bind mounts)."""
    probe = path / ".hermes-write-probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


_dir_writable = dir_writable


def _lock_newer_than_install(sidecar_dir: Path) -> bool:
    """True when the committed lockfile postdates npm's install marker
    (``node_modules/.package-lock.json``) — the same signal ``npm ci`` uses. False on any
    stat failure so an odd filesystem never blocks start."""
    lockfile = sidecar_dir / "package-lock.json"
    marker = sidecar_dir / "node_modules" / ".package-lock.json"
    try:
        return lockfile.stat().st_mtime > marker.stat().st_mtime
    except OSError:
        return False


def resolve_sidecar_dir(source_dir: Optional[Path] = None) -> Path:
    """Return the directory the sidecar should run from (see module doc)."""
    source = Path(source_dir) if source_dir is not None else SOURCE_SIDECAR_DIR
    override = os.getenv("PHOTON_SIDECAR_DIR")
    if override:
        return Path(override)
    if _dir_writable(source):
        return source
    # Read-only tree with baked, current deps: run in place (the sidecar never writes there).
    if (source / "node_modules").exists() and not _lock_newer_than_install(source):
        return source
    from hermes_constants import get_hermes_home
    mirror = get_hermes_home() / "photon" / "sidecar"
    try:
        mirror.mkdir(parents=True, exist_ok=True)
        for name in _MIRROR_FILES:
            src, dst = source / name, mirror / name
            if src.exists() and (not dst.exists() or not filecmp.cmp(str(src), str(dst), shallow=False)):
                shutil.copy2(str(src), str(dst))
        return mirror
    except OSError as exc:
        logger.warning(
            "[photon] install tree is read-only and mirroring the sidecar "
            "to %s failed (%s) — falling back to the read-only source dir; "
            "dependency installs will not be possible",
            mirror, exc)
        return source


def _sidecar_dir() -> Path:
    """Sidecar runtime dir, resolved once on first use (never at import)."""
    global _SIDECAR_DIR
    if _SIDECAR_DIR is None:
        _SIDECAR_DIR = resolve_sidecar_dir()
    return _SIDECAR_DIR


def _npm_error_log() -> Path:
    """Path of the persisted npm-failure log (derived from the sidecar dir)."""
    return _NPM_ERROR_LOG if _NPM_ERROR_LOG is not None else _sidecar_dir() / ".photon-npm-error.log"
