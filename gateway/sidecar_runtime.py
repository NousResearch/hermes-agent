"""Resolve the directory a Node sidecar runs from.

Hermes ships two Node sidecars: the Photon iMessage bridge
(``plugins/platforms/photon/sidecar``) and the WhatsApp Baileys bridge
(``scripts/whatsapp-bridge``). Both need a ``node_modules`` beside their entry
file, and some installs put the source tree somewhere nothing can write.

Node decides the shape of the answer. Its ESM resolver reads ``node_modules``
only from directories above the importing file. ``NODE_PATH`` applies to
CommonJS and not to ESM, and both sidecars declare ``"type": "module"``, so
there is no way to leave the code in a read-only tree and point Node at
packages held elsewhere. The entry file and the packages have to share a tree.
That is why a read-only install with no usable deps copies the sidecar to a
writable directory: it is the only arrangement Node accepts, not a preference.

Order:

1. ``HERMES_<NAME>_SIDECAR_DIR`` — an operator override, used as given.
2. Source writable → run in place. Dev checkouts, and any plugin a user
   installs under ``$HERMES_HOME/plugins``.
3. Source read-only, deps present and matching the lockfile → run in place.
   The Nix store and the container image both arrive this way, and the
   sidecar never writes inside its own directory.
4. Source read-only, deps missing or stale → copy the sidecar to
   ``$HERMES_HOME/sidecars/<name>`` and hand back that path, where the
   caller's usual npm install can work.

Rung 4 copies the whole tree apart from ``node_modules``. It does not consult
a list of files to copy: such a list has to name every module the entry file
imports, and the one this replaces was wrong twice, each time in a way that
only appears on a read-only install.
"""
from __future__ import annotations

import filecmp
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_DEPS_DIR = "node_modules"
# npm writes this inside node_modules to record the state of the last install.
# Comparing it with the committed lockfile is the check `npm ci` runs.
_INSTALL_MARKER = ".package-lock.json"


def override_env_var(name: str) -> str:
    """Return the override variable for sidecar ``name``."""
    return f"HERMES_{name.upper().replace('-', '_')}_SIDECAR_DIR"


def dir_writable(path: Path) -> bool:
    """Can Hermes create a file in ``path``?

    Probe with a real create and delete. A stat of the mode bits gives the
    wrong answer under root-squash and on a read-only bind mount, which are
    the cases this function exists to detect.
    """
    probe = path / ".hermes-write-probe"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def deps_are_current(sidecar_dir: Path) -> bool:
    """Does the install in ``sidecar_dir`` match its lockfile?

    False when either file is absent or unreadable, so a first run and an odd
    filesystem both resolve to "install needed" rather than to an error.

    ``plugins.platforms.photon.adapter._sidecar_deps_stale`` reads the same
    two files with the opposite missing-file answer, on purpose: there the
    missing case belongs to ``sidecar_deps_installed``.
    """
    lockfile = sidecar_dir / "package-lock.json"
    marker = sidecar_dir / _DEPS_DIR / _INSTALL_MARKER
    try:
        return marker.stat().st_mtime >= lockfile.stat().st_mtime
    except OSError:
        return False


def _refresh_mirror(source: Path, mirror: Path) -> None:
    """Copy ``source`` into ``mirror``, without ``node_modules``.

    Runs on each resolve, so an image update reaches a mirror that already
    exists. Compares content rather than mtime, because a copy has the mtime
    of the copy. ``node_modules`` stays out: npm owns the mirror's copy, and
    replacing it would make each update a fresh install.
    """
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        mirror,
        ignore=shutil.ignore_patterns(_DEPS_DIR),
        copy_function=_copy_if_changed,
        dirs_exist_ok=True,
    )


def _copy_if_changed(src: str, dst: str) -> None:
    """Copy, skipped when the destination already holds the same bytes.

    ``shutil.copy``, not ``copy2``. copy2 gives the destination the mtime
    of the SOURCE, and a Nix store source has mtime = epoch. A refreshed
    mirror lockfile with an epoch mtime always looks older than npm's
    install marker, so ``deps_are_current`` keeps stale node_modules
    through every upgrade. A plain copy stamps the file with the copy
    time, so a content change always postdates the previous install.

    npm's ``node_modules/.package-lock.json`` cannot replace this content
    comparison. It is a different document, and npm matches it against the
    committed lockfile semantically, not byte for byte.
    """
    if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
        return
    shutil.copy(src, dst)


def resolve_sidecar(name: str, source_dir: Path) -> Path:
    """Return the directory sidecar ``name`` should run from.

    ``source_dir`` is where the sidecar ships in the install tree.
    """
    source = Path(source_dir)

    override = os.getenv(override_env_var(name))
    if override:
        return Path(override)

    if dir_writable(source):
        return source

    if (source / _DEPS_DIR).exists() and deps_are_current(source):
        return source

    from hermes_constants import get_hermes_home

    mirror = get_hermes_home() / "sidecars" / name
    try:
        _refresh_mirror(source, mirror)
        return mirror
    except OSError as exc:
        logger.warning(
            "[%s] the install tree is read-only and the copy to %s failed "
            "(%s). Falling back to the read-only source directory, where a "
            "dependency install cannot run.",
            name,
            mirror,
            exc,
        )
        return source
