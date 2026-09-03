"""Unified server-side media-roots policy (one source of truth for every
media-serving dashboard endpoint).

Historically the three media fetch surfaces enforced *different* root
restrictions (see the ``lib/media.ts`` comment this module retires):

* ``GET /api/media`` — image-extension allowlist + a private root list
  (``~/.hermes/{images,screenshots,cache}``).
* ``GET /api/files/stream`` / the audio-video branch of ``GET /api/files/download``
  — streamable-extension allowlist, no root confinement at all.
* ``GET /api/fs/read-data-url`` — any path on disk (16 MB cap), no confinement.

That divergence is the "finicky" class: a file the gateway happily serves as an
inline attachment is refused by the data-URL preview (or vice versa) depending
on which endpoint the desktop picked for its extension, and the rules are not
written down anywhere a user can adjust.

This module defines ONE policy:

* The allowed roots come from ``media.roots`` in config.yaml. Each entry is a
  ``~``-expanded absolute directory; the safe default (``None`` / unset /
  empty) is the gateway's workspace plus the Hermes-managed media subtrees —
  ``images/``, ``screenshots/``, and ``cache/`` (where generated images,
  browser screenshots, inbound platform media, and relay-staged media all
  live).
* ``media_roots()`` resolves those to symlink-safe absolute paths at request
  time (config changes apply without a restart).
* ``path_in_media_roots()`` is the single containment test every media-serving
  endpoint consults. Symlinks are resolved before the check, so a link escaping
  a root is judged by its target.

Scope: media-serving fetch surfaces only. The dashboard Files tab
(``/api/files`` list/read/upload under :class:`ManagedFilesPolicy`) keeps its
own operator-facing contract — a root-level filesystem browser and a media
player are different surfaces with different threat models. Explicit Save-as
via ``GET /api/fs/download`` stays ungated for the same reason: it is the
desktop's fallback path when an inline fetch is denied (never-silent cards),
and it already enforces the sensitive-file denylist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_config() -> Dict[str, Any]:
    """Best-effort config load; policy must survive a broken config.yaml."""
    try:
        from hermes_cli.config import load_config

        return load_config() or {}
    except Exception:  # noqa: BLE001 - policy falls back to safe defaults
        return {}


def default_media_roots() -> List[Path]:
    """The safe default roots, as *unresolved* paths.

    Workspace (the gateway process's cwd — agent-produced artifacts land here
    by default) plus the Hermes-managed media subtrees under
    :func:`hermes_constants.get_hermes_home` (generated images, screenshots,
    cached platform/relay media). Resolved per-request by :func:`media_roots`
    so profile HERMES_HOME overrides are honored at call time.
    """
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return [
        Path.cwd(),
        home / "images",
        home / "screenshots",
        home / "cache",
    ]


def configured_roots(config: Optional[Dict[str, Any]] = None) -> Optional[List[Path]]:
    """Raw ``media.roots`` entries from config.yaml, or None for the default.

    Returns None (meaning "use :func:`default_media_roots`") when the key is
    absent, empty, or malformed — an operator clearing the list gets the safe
    default rather than a policy that serves nothing. Invalid entries are
    skipped individually so one bad path can't disable the whole policy.
    """
    if config is None:
        config = _load_config()
    media_cfg = config.get("media")
    if not isinstance(media_cfg, dict):
        return None
    raw = media_cfg.get("roots")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None

    out: List[Path] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            continue
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            logger.warning("media.roots entry %r is not absolute; ignoring", entry)
            continue
        out.append(candidate)
    return out or None


def media_roots(config: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Resolved, symlink-safe media roots. Never raises; never empty.

    Resolution failures (unresolvable symlink loops, permission errors) drop
    the offending root rather than failing the request — a partially
    restrictive policy is the graceful degradation, matching how
    ``/api/media``'s private root builder behaved. Drops of *default* roots
    (e.g. ``~/.hermes/images`` before the first generated image creates it)
    are silent — no file can live under a not-yet-existing directory, so the
    policy outcome is identical — while drops of *configured* entries are
    logged so an operator typo can't become silent 403s.

    Concurrency note: the policy is re-read and re-resolved on every request,
    so a config edit can flip a root set between two requests. Each request
    is judged against one internally-consistent snapshot (this function's
    return value), which is the guarantee callers need; there is no
    cross-request atomicity, by design — the config file is not a transaction
    surface.
    """
    raw_roots = configured_roots(config)
    explicit = raw_roots is not None
    if raw_roots is None:
        raw_roots = default_media_roots()

    out: List[Path] = []
    for root in raw_roots:
        try:
            resolved = root.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            if explicit:
                logger.warning("media.roots entry %r could not be resolved; ignoring", root)
            continue
        if resolved.is_dir():
            out.append(resolved)
        elif explicit:
            logger.warning(
                "media.roots entry %r is not an existing directory; media under it "
                "will be denied until it exists",
                root,
            )
    return out


def path_in_media_roots(
    path: Path,
    config: Optional[Dict[str, Any]] = None,
    extra_roots: Optional[List[Path]] = None,
) -> bool:
    """True when *path* (resolved, symlink-safe) is under an allowed root.

    The single containment test for every media-serving endpoint. ``extra_roots``
    lets a caller extend the policy with a root it already trusts for full read
    access — the dashboard's operator-locked managed files root
    (``ManagedFilesPolicy.locked_root``): when an operator pins the Files tab to
    a root, every file under it is already downloadable, so media playback from
    the same tree must not be denied by the media policy. A root itself counts
    as inside (a root directory is not readable as media — the callers check
    ``is_file`` — but containment-wise ``target == root`` keeps the predicate
    total).
    """
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    roots = media_roots(config)
    for extra in extra_roots or ():
        try:
            extra_resolved = Path(extra).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if extra_resolved.is_dir():
            roots.append(extra_resolved)
    for root in roots:
        if resolved == root or root in resolved.parents:
            return True
    return False
