"""
Resolve where the Photon sidecar runs from and where its Node deps live.

The sidecar source ships inside the installed plugin tree
(``plugins/platforms/photon/sidecar/``). On dev/source installs that tree is
writable and everything — ``npm ci``, the spectrum patch, the sidecar itself —
happens in place. Hosted/managed images instead keep the whole install tree
under an immutable ``/opt/hermes`` (read-only for the hermes user), which
broke every install/self-heal path with EROFS (NS-606).

Resolution order (mirrors ``resolve_whatsapp_bridge_dir`` for the Baileys
bridge, which hit the same wall):

1. ``PHOTON_SIDECAR_DIR`` env override — operator escape hatch, used as-is.
2. Source dir writable → run in place (dev installs, unchanged behavior).
3. Source dir read-only but ``node_modules`` is baked and current → run in
   place. This is the managed-image happy path: the Dockerfile bakes the
   sidecar deps with ``npm ci`` at build time (deterministic installs,
   NS-559), so no runtime install is ever needed.
4. Source dir read-only and deps missing or stale → mirror the sidecar
   source files to ``$HERMES_HOME/photon/sidecar`` (the durable data volume,
   e.g. ``/opt/data`` on hosted) and return that. The caller's normal
   install/self-heal machinery then works there because it is writable.

The mirror is refreshed on every resolve: when an image update changes a
sidecar source file, the changed file is re-copied (content compare, not
mtime) while ``node_modules`` is left in place — the adapter's existing
lockfile-vs-install-marker staleness check then triggers the ``npm ci``
self-heal inside the mirror.

This module is import-light on purpose: both ``adapter.py`` (gateway) and
``cli.py`` (``hermes photon ...``) use it.
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

# The files that define the sidecar. Mirrored into the writable runtime dir
# when the install tree is read-only. node_modules is deliberately absent —
# it is either baked (managed image) or installed by npm in the mirror.
_MIRROR_FILES = (
    "index.mjs",
    "package.json",
    "package-lock.json",
    "patch-spectrum-mixed-attachments.mjs",
)


def dir_writable(path: Path) -> bool:
    """True when we can create files in ``path`` (probe-based, not stat).

    A stat-mode check lies on containers (root-squash, read-only bind
    mounts), so probe with a real create+unlink like the WhatsApp bridge
    resolver does.
    """
    probe = path / ".hermes-write-probe"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


# Backwards-friendly private alias for module-internal use.
_dir_writable = dir_writable


def _lock_newer_than_install(sidecar_dir: Path) -> bool:
    """True when the committed lockfile postdates npm's install marker.

    Same signal as ``adapter._sidecar_deps_stale`` — duplicated here (three
    lines) rather than imported so this module stays import-light for the
    CLI. Returns False on any stat failure so an odd filesystem never forces
    the mirror path.
    """
    lockfile = sidecar_dir / "package-lock.json"
    marker = sidecar_dir / "node_modules" / ".package-lock.json"
    try:
        return lockfile.stat().st_mtime > marker.stat().st_mtime
    except OSError:
        return False


def resolve_sidecar_dir(source_dir: Optional[Path] = None) -> Path:
    """Return the directory the sidecar should run from (see module doc).

    ``source_dir`` defaults to the installed plugin tree; tests and callers
    that monkeypatch the adapter's ``_SIDECAR_DIR`` pass it through so the
    override keeps working.
    """
    source = Path(source_dir) if source_dir is not None else SOURCE_SIDECAR_DIR

    override = os.getenv("PHOTON_SIDECAR_DIR")
    if override:
        return Path(override)

    if _dir_writable(source):
        return source

    # Read-only install tree (hosted/managed image). If the image baked the
    # deps at build time and they match the lockfile, run in place — the
    # sidecar itself never writes inside its own directory.
    if (source / "node_modules").exists() and not _lock_newer_than_install(source):
        return source

    # Deps missing or stale inside a read-only tree: mirror to the durable
    # data volume so the normal install/self-heal machinery has somewhere
    # writable to work.
    from hermes_constants import get_hermes_home

    mirror = get_hermes_home() / "photon" / "sidecar"
    try:
        mirror.mkdir(parents=True, exist_ok=True)
        for name in _MIRROR_FILES:
            src = source / name
            if not src.exists():
                continue
            dst = mirror / name
            if not dst.exists() or not filecmp.cmp(str(src), str(dst), shallow=False):
                shutil.copy2(str(src), str(dst))
        return mirror
    except OSError as exc:
        logger.warning(
            "[photon] install tree is read-only and mirroring the sidecar "
            "to %s failed (%s) — falling back to the read-only source dir; "
            "dependency installs will not be possible",
            mirror,
            exc,
        )
        return source
